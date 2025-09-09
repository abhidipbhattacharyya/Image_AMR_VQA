# coding=utf-8
# Copyleft 2019 project LXRT.

import json
import os
from os import listdir
from os.path import isfile, join
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset
from tsv_utils import load_obj_tsv
from amr_processor import amr_process
from AMRBART.AMRBartTokenizer import AMRBartTokenizer, AMRRobertaTokenizer
from collections import Counter, namedtuple

# Load part of the dataset for fast checking.
# Notice that here is the number of images instead of the number of data,
# which means all related data to the images would be used.
TINY_IMG_NUM = 512
FAST_IMG_NUM = 5000

# The path to data and image features.
VQA_DATA_ROOT = '/srv/data1/abhidipbhatt/data/coco/lxmert_data/'
MSCOCO_IMGFEAT_ROOT = '/srv/data1/abhidipbhatt/data/coco/mscoco_imgfeat/'
SPLIT2NAME = {
    'train': 'train2014',
    'valid': 'val2014',
    'minival': 'val2014',
    'nominival': 'val2014',
    'test': 'test2015',
}

batch_fields = [
    'input_ids',
    'input_attention_masks',
    'vision_feats',
    'visual_pos',
    'ques_ids',
    'amr_ids',
    'amr_attention_masks',
    'target_ids',
    
]
Batch = namedtuple('Batch', field_names=batch_fields,
                   defaults=[None] * len(batch_fields)) 

class VQADataset:
    """
    A VQA data example in json file:
        {
            "answer_type": "other",
            "img_id": "COCO_train2014_000000458752",
            "label": {
                "net": 1
            },
            "question_id": 458752000,
            "question_type": "what is this",
            "sent": "What is this photo taken looking through?"
        }
    """
    def __init__(self, splits: str, datapath):
        self.name = splits
        self.splits = splits.split(',')

        # Loading datasets
        self.data = []
        for split in self.splits:
            self.data.extend(json.load(open(os.path.join(datapath,'{}.json'.format(split)),"r")))
        print("Load %d data from split(s) %s." % (len(self.data), self.name))

        # Convert list to dict (for evaluation)
        self.id2datum = {
            datum['question_id']: datum
            for datum in self.data
        }

        # Answers
        self.ans2label = json.load(open(datapath+"trainval_ans2label.json","r"))
        self.label2ans = json.load(open(datapath+"trainval_label2ans.json","r"))
        assert len(self.ans2label) == len(self.label2ans)

    @property
    def num_answers(self):
        return len(self.ans2label)

    def __len__(self):
        return len(self.data)


"""
An example in obj36 tsv:
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
FIELDNAMES would be keys in the dict returned by load_obj_tsv.
"""
class VQATorchDataset(Dataset):
    def __init__(self, dataset: VQADataset, MSCOCO_IMGFEAT_ROOT, model_name_or_path=None):
        super().__init__()
        self.raw_dataset = dataset

        if args.tiny:
            topk = TINY_IMG_NUM
        elif args.fast:
            topk = FAST_IMG_NUM
        else:
            topk = None

        # Loading detection features to img_data
        img_data = []
        
        if model_name_or_path==None:
            model_name_or_path = "unc-nlp/lxmert-base-uncased"
        lxmert_tokenizer = LxmertTokenizer.from_pretrained(model_name_or_path)
        
        for split in dataset.splits:
            # Minival is 5K images in MS COCO, which is used in evaluating VQA/LXMERT-pre-training.
            # It is saved as the top 5K features in val2014_***.tsv
            load_topk = 5000 if (split == 'minival' and topk is None) else topk
            img_data.extend(load_obj_tsv(
                os.path.join(MSCOCO_IMGFEAT_ROOT, '%s_obj36.tsv' % (SPLIT2NAME[split])),
                topk=load_topk))

        # Convert img list to dict
        self.imgid2img = {}
        for img_datum in img_data:
            self.imgid2img[img_datum['img_id']] = img_datum

        # Only kept the data with loaded image features
        self.data = []
        for datum in self.raw_dataset.data:
            if datum['img_id'] in self.imgid2img:
                self.data.append(datum)
        print("Use %d data in torch dataset" % (len(self.data)))
        print()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        datum = self.data[item]

        img_id = datum['img_id']
        ques_id = datum['question_id']
        ques = datum['sent']

        # Get image info
        img_info = self.imgid2img[img_id]
        obj_num = img_info['num_boxes']
        feats = img_info['features'].copy()
        boxes = img_info['boxes'].copy()
        assert obj_num == len(boxes) == len(feats)

        # Normalize the boxes (to 0 ~ 1)
        img_h, img_w = img_info['img_h'], img_info['img_w']
        boxes = boxes.copy()
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1+1e-5)
        np.testing.assert_array_less(-boxes, 0+1e-5)

        # Provide label (target)
        if 'label' in datum:
            label = datum['label']
            target = torch.zeros(self.raw_dataset.num_answers)
            for ans, score in label.items():
                target[self.raw_dataset.ans2label[ans]] = score
            return ques_id, feats, boxes, ques, target
        else:
            return ques_id, feats, boxes, ques

def collate_fn(batch, lxmert_tokenizer):
    ques_ids = [b[0] for b in batch]
    feats= [torch.FloatTensor(b[1]) for b in batch]
    boxes = [torch.FloatTensor(b[2]) for b in batch]
    ques = [b[3] for b in batch]
    labels = None
    
    if len(batch[0])==5:
        labels = [b[4] for b in batch]
        labels = torch.stack(labels)
    
    inputs = lxmert_tokenizer(
        ques,
        padding="max_length",
        max_length=20,
        truncation=True,
        return_token_type_ids=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt"
    )
    visual_feats = torch.stack(feats)
    boxes = torch.stack(boxes)
    input_ids = inputs.input_ids
    input_attention = inputs.inputs.attention_mask
    
    if labels:
        return input_ids, input_attention, visual_feats, boxes, ques_ids, labels
    else:
        return input_ids, input_attention, visual_feats, boxes, ques_ids
    
    
class VQATorchAMRDataset(Dataset):
    def __init__(self, dataset: VQADataset, MSCOCO_IMGFEAT_ROOT, AMR_model_path='', AMR_path='/srv/data1/abhidipbhatt/data/coco/amr/ibm_single_amr', mode_training=True ):
        super().__init__()
        self.raw_dataset = dataset
        self.AMR_path_map = {}
        for folder in ['train2014','val2014']:
            path = os.path.join(AMR_path, folder)
            for file_amr in listdir(path):
                self.AMR_path_map[file_amr.replace('.amr','')] = os.path.join(path,file_amr)
                
        #if args.tiny:
            #topk = TINY_IMG_NUM
        #elif args.fast:
            #topk = FAST_IMG_NUM
        #else:
        topk = None

        # Loading detection features to img_data
        img_data = []
        
        
        for split in dataset.splits:
            # Minival is 5K images in MS COCO, which is used in evaluating VQA/LXMERT-pre-training.
            # It is saved as the top 5K features in val2014_***.tsv
            load_topk = 5000 if (split == 'minival' and topk is None) else topk
            img_data.extend(load_obj_tsv(
                os.path.join(MSCOCO_IMGFEAT_ROOT, '%s_obj36.tsv' % (SPLIT2NAME[split])),
                topk=load_topk))

        # Convert img list to dict
        self.imgid2img = {}
        for img_datum in img_data:
            self.imgid2img[img_datum['img_id']] = img_datum
            
        #ignorelist
        ignore_list = []
        ignore_list_name = os.path.join(AMR_path,'ignore_list.txt')
        if os.path.exists(ignore_list_name):
            with open(ignore_list_name,'r') as f:
                ignore_list = f.readlines()
            ignore_list = [ig.split('/')[1].replace('.jpg','').strip() for ig in ignore_list]
            
        # Only kept the data with loaded image features
        self.data = []
        data_temp= []
        for datum in self.raw_dataset.data :
            if datum['img_id'] in self.imgid2img and datum['img_id'] in self.AMR_path_map and datum['img_id'] not in ignore_list:
                data_temp.append(datum)

        if mode_training:
            training_list = '/srv/data1/abhidipbhatt/data/coco/lxmert_data/training_sample_10000.txt'
            training_qids = None
            if os.path.exists(training_list):
                with open(training_list,'r') as f:
                    training_qids = f.readlines()
                training_qids = [int(tq.strip()) for tq in training_qids]
        
            if training_qids:
                for datum in data_temp:
                    if datum['question_id'] in training_qids:
                        self.data.append(datum)
            else:
                self.data = data_temp
        else:
            self.data = data_temp
        print("Use %d data in torch dataset" % (len(self.data)))
        print("label dict size:{}".format(self.raw_dataset.num_answers))
        print()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        datum = self.data[item]

        img_id = datum['img_id']
        ques_id = datum['question_id']
        ques = datum['sent']
        
        # Get AMR
        amr_file = self.AMR_path_map[img_id]
        amr = amr_process(amr_file, mode='path')
        # Get image info
        img_info = self.imgid2img[img_id]
        obj_num = img_info['num_boxes']
        feats = img_info['features'].copy()
        boxes = img_info['boxes'].copy()
        assert obj_num == len(boxes) == len(feats)

        # Normalize the boxes (to 0 ~ 1)
        img_h, img_w = img_info['img_h'], img_info['img_w']
        boxes = boxes.copy()
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1+1e-5)
        np.testing.assert_array_less(-boxes, 0+1e-5)

        # Provide label (target)
        if 'label' in datum:
            label = datum['label']
            target = torch.zeros(self.raw_dataset.num_answers)
            for ans, score in label.items():
                target[self.raw_dataset.ans2label[ans]] = score
            return ques_id, feats, boxes, ques, amr, target
        else:
            return ques_id, feats, boxes, ques, amr
        
        
        
def get_amr_ids(stripped_graphs, tokenizer, max_graph_len=512):
    input_text = ['%s' % graph for graph in stripped_graphs]
    input_encodings = [
        [tokenizer.bos_token_id, tokenizer.mask_token_id, tokenizer.eos_token_id] +
        [tokenizer.amr_bos_token_id] + tokenizer.tokenize_amr(itm.split())[:max_graph_len -5] +
        [tokenizer.amr_eos_token_id] for itm in input_text]

    # padding
    max_batch_length = max(len(x) for x in input_encodings)
    attention_mask = [[1]*len(x) + [0]*(max_batch_length - len(x)) for x in input_encodings]
    input_ids = [x + [tokenizer.pad_token_id]*(max_batch_length - len(x)) for x in input_encodings]

    # truncation
    if max_batch_length > max_graph_len:
        input_ids = [x[:max_graph_len] for x in input_ids]
        attention_mask = [x[:max_graph_len] for x in attention_mask]
        print("overlength")

    # Convert to tensors
    input_ids = torch.LongTensor(input_ids)
    attention_mask = torch.LongTensor(attention_mask)
    return input_ids,attention_mask
    


def collate_fn_AMR(batch, lxmert_tokenizer, amr_tokenizer):
    ques_ids = [b[0] for b in batch]
    feats= [torch.FloatTensor(b[1]) for b in batch]
    boxes = [torch.FloatTensor(b[2]) for b in batch]
    ques = [b[3] for b in batch]
    amrs = [b[4] for b in batch]
    labels = None
    if len(batch[0])==6:
        labels = [b[5] for b in batch]
        labels = torch.stack(labels)
    
    inputs = lxmert_tokenizer(
        ques,
        padding="max_length",
        max_length=20,
        truncation=True,
        return_token_type_ids=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt"
    )
    amr_ids, amr_Attns = get_amr_ids(amrs,amr_tokenizer)
    
    visual_feats = torch.stack(feats)
    boxes = torch.stack(boxes)
    input_ids = inputs.input_ids
    input_attention = inputs.attention_mask
    
    if labels is not None:
        return Batch(
            input_ids= input_ids,
            input_attention_masks= input_attention,
            vision_feats= visual_feats,
            visual_pos= boxes,
            amr_ids= amr_ids,
            amr_attention_masks= amr_Attns,
            ques_ids= ques_ids,
            target_ids= labels,
        )   
    
        #return input_ids, input_attention, visual_feats, boxes, ques_ids, amr_ids, labels
    else:
        return Batch(
            input_ids= input_ids,
            input_attention_masks= input_attention,
            vision_feats= visual_feats,
            visual_pos= boxes,
            amr_ids= amr_ids,
            amr_attention_masks= amr_Attns,
            ques_ids= ques_ids,
            target_ids= None,
        )
        #return input_ids, input_attention, visual_feats, boxes, ques_ids, amr_ids
    
    
class VQAEvaluator:
    def __init__(self, dataset: VQADataset):
        self.dataset = dataset

    def evaluate(self, quesid2ans: dict):
        score = 0.
        for quesid, ans in quesid2ans.items():
            datum = self.dataset.id2datum[quesid]
            label = datum['label']
            if ans in label:
                score += label[ans]
        return score / len(quesid2ans)

    def dump_result(self, quesid2ans: dict, path):
        """
        Dump results to a json file, which could be submitted to the VQA online evaluation.
        VQA json file submission requirement:
            results = [result]
            result = {
                "question_id": int,
                "answer": str
            }

        :param quesid2ans: dict of quesid --> ans
        :param path: The desired path of saved file.
        """
        with open(path, 'w') as f:
            result = []
            for ques_id, ans in quesid2ans.items():
                result.append({
                    'question_id': ques_id,
                    'answer': ans
                })
            json.dump(result, f, indent=4, sort_keys=True)

