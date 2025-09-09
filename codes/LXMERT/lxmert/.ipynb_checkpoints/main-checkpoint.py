import transformers
print('transformer version:{}'.format(transformers.__version__))
from modeling_lxmert_AMR import LxmertForQuestionAnswering
from vqa_data import VQADataset, VQATorchDataset, VQATorchAMRDataset, collate_fn, collate_fn_AMR
from transformers import LxmertTokenizer
from modeling_lxmert_AMR import LxmertForQuestionAnswering
from AMRBART.AMRBartTokenizer import AMRBartTokenizer, AMRRobertaTokenizer
from transformers import BartForConditionalGeneration, AutoConfig, AutoModel
from transformers import AdamW # WarmupLinearSchedule, WarmupConstantSchedule
import argparse
#from transformers.optimization import WarmupLinearSchedule
from transformers import get_linear_schedule_with_warmup
import torch.nn as nn

import errno
import os
import os.path as op
import yaml
import random
import torch
import numpy as np
import torch.distributed as dist
from functools import partial
import logging
from tsv_utils import dump_result, concat_all
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
#PyTorch version 1.6.0 available.
#transformer version:3.5.1

def mkdir(path):
    # if it is the current folder, skip.
    if path == '':
        return
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def set_seed(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)
        
def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0

def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()
    
def ensure_init_process_group(local_rank=None, port=12345):
    # init with env
    world_size = torch.cuda.device_count()#int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    if world_size > 1 and not dist.is_initialized():
        assert local_rank is not None
        print("Init distributed training on local rank {}".format(local_rank))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend='nccl', init_method='env://'#, world_size=world_size, rank = local_rank
        )
    return local_rank

def make_data_sampler(dataset, shuffle, distributed):
    if distributed:
        return torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler

def make_data_loader(dataset, partial_collate_fn, args, is_train=True):
    if is_train:
        shuffle = True
        images_per_gpu = args.per_gpu_train_batch_size
        images_per_batch = images_per_gpu * get_world_size()
        iters_per_batch = len(dataset) // images_per_batch
        num_iters = iters_per_batch * args.num_train_epochs
        logging.info("len dataset:{}".format(len(dataset)))
        logging.info("Train with {} images per GPU.".format(images_per_gpu))
        logging.info("Total batch size {}".format(images_per_batch))
        logging.info("Total training steps {}".format(num_iters))
    else:
        shuffle = False
        images_per_gpu = args.per_gpu_eval_batch_size

    sampler = make_data_sampler(dataset, shuffle, args.distributed)
    data_loader = torch.utils.data.DataLoader(
        dataset, num_workers=args.num_workers, sampler=sampler,
        batch_size=images_per_gpu,
        pin_memory=True,
        collate_fn=partial_collate_fn,
    )
    return data_loader
    
def save_checkpoint(model, args, epoch, iteration, num_trial=10):
    checkpoint_dir = op.join(args.output_dir, 'checkpoint-{}-{}'.format(
        epoch, iteration))
    if not is_main_process():
        return checkpoint_dir
    mkdir(checkpoint_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    for i in range(num_trial):
        try:
            model_to_save.save_pretrained(checkpoint_dir)
            torch.save(args, op.join(checkpoint_dir, 'training_args.bin'))
            #tokenizer.save_pretrained(checkpoint_dir)
            logging.info("Save checkpoint to {}".format(checkpoint_dir))
            break
        except:
            pass
    else:
        logging.info("Failed to save checkpoint after {} trails.".format(num_trial))
    return checkpoint_dir


def train(args, train_dataloader, model, vset=None, val_loader=None):
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    t_total = len(train_dataloader) // args.gradient_accumulation_steps \
                * args.num_train_epochs

    # Prepare optimizer and scheduler
    
    no_decay = ['bias', 'LayerNorm.weight']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not \
            any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if \
            any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
                optimizer,  num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    

    logging.info("***** Running training *****")
    logging.info("  Num Epochs = %d", args.num_train_epochs)
    logging.info("  Batch size per GPU = %d", args.per_gpu_train_batch_size)
    logging.info("  Total train batch size (w. parallel, & accumulation) = %d",
                   args.per_gpu_train_batch_size * get_world_size() * args.gradient_accumulation_steps)
    logging.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logging.info("  Total optimization steps = %d", t_total)

    global_step, global_loss, global_acc =0,  0.0, 0.0
    model.zero_grad()
    eval_log = []
    best_score = 0
    for epoch in range(int(args.num_train_epochs)):
        for step,  batch in enumerate(train_dataloader):
            #batch = tuple(t.to(args.device) for t in batch)
            
            input_ids = batch.input_ids.to(args.device)
            input_attention_masks= batch.input_attention_masks.to(args.device)
            vision_feats= batch.vision_feats.to(args.device)
            visual_pos=  batch.visual_pos.to(args.device)
            amr_ids= batch.amr_ids.to(args.device)
            #amr_attention_masks= batch.amr_attention_masks.to(args.device)
            target_ids = batch.target_ids.to(args.device)
            
            
            model.train()
            outputs = model(input_ids=input_ids,
                            amr_ids = amr_ids,
                            visual_feats = vision_feats,
                            visual_pos = visual_pos,
                            attention_mask=input_attention_masks,
                            visual_attention_mask=None,
                            token_type_ids=None,
                            inputs_embeds=None,
                            labels=target_ids,
                            output_attentions=None,
                            output_hidden_states=None,
                            return_dict=None,)
            
            loss, logits = outputs[:2]
           
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            global_loss += loss.item()
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                global_step += 1
                scheduler.step()
                optimizer.step()
                model.zero_grad()
                if global_step % args.logging_steps == 0:
                    logging.info("Epoch: {}, global_step: {}, lr: {:.6f}, loss: {:.4f} ({:.4f}))".format(epoch, global_step, optimizer.param_groups[0]["lr"], loss, global_loss / global_step,))

                if (args.save_steps > 0 and global_step % args.save_steps == 0) or \
                        global_step == t_total:
                    checkpoint_dir = save_checkpoint(model, args, epoch, global_step)

    if vset and val_loader:
        predict(model, vset, val_loader, args)
    return checkpoint_dir


def predict(model, dset, loader, args):
    """
    Predict the answers to questions in a data split.

    :param eval_tuple: The data tuple to be evaluated.
    :param dump: The path of saved file to dump results.
    :return: A dict of question_id to answer.
    """
    model.eval()
    #dset, loader = eval_tuple
    quesid2ans = {}
    predict_file = os.path.join(args.output_dir,'result.json')
    world_size = get_world_size()
    if world_size == 1:
        cache_file = predict_file
    else:
        cache_file = predict_file.replace('.json','_{}_{}.json'.format(get_rank(),
                world_size))
    
    for i, batch in enumerate(loader):
        input_ids = batch.input_ids.to(args.device)
        input_attention_masks= batch.input_attention_masks.to(args.device)
        vision_feats= batch.vision_feats.to(args.device)
        visual_pos=  batch.visual_pos.to(args.device)
        amr_ids= batch.amr_ids.to(args.device)   
        ques_ids = batch.ques_ids
            
        with torch.no_grad():
            outputs = model(input_ids=input_ids,
                        amr_ids = amr_ids,
                        visual_feats = vision_feats,
                        visual_pos = visual_pos,
                        attention_mask=input_attention_masks,
                        visual_attention_mask=None,
                        token_type_ids=None,
                        inputs_embeds=None,
                        labels=None,
                        output_attentions=None,
                        output_hidden_states=None,
                        return_dict=None,)
            
            logit = outputs[0]
            score, label = logit.max(1)
            for qid, l in zip(ques_ids, label.cpu().numpy()):
                ans = dset.label2ans[l]
                quesid2ans[qid] = ans
    
    dump_result(quesid2ans, cache_file)
    if get_world_size() > 1:
        torch.distributed.barrier()
    if get_world_size() > 1 and is_main_process():
        files = [ predict_file.replace('.josn','_{}_{}.json'.format(i,
                world_size)) for i in range(world_size)]
        concat_all(files, predict_file)
    if get_world_size() > 1:
        torch.distributed.barrier()
    return quesid2ans

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

    
#python -m torch.distributed.launch --nproc_per_node=1 --master_port=18121 main.py --do_train'
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='/srv/data1/abhidipbhatt/data/coco/lxmert_data/', type=str, required=False, help="The input data dir with all required files.")
    parser.add_argument("--tsv_feat_path", default='/srv/data1/abhidipbhatt/data/coco/mscoco_imgfeat/', type=str, required=False, help="The input data dir with all required files.")
    parser.add_argument("--model_name_or_path", default="/srv/data1/abhidipbhatt/models/lxmert/lxmert-base-uncased", type=str, required=False,help="lxmert model") #lxmert-base-uncased
    parser.add_argument("--resume_training_path", default= '/srv/data1/abhidipbhatt/models/lxmert/lxmert_amr/doc_amr_base/checkpoint-2-110940', type=str, required=False,help="lxmert model")
    parser.add_argument("--AMR_model_path", default='/srv/data1/abhidipbhatt/models/AMRBART-large-finetuned-AMR3.0-AMR2Text-v2', type=str, required=False,
                        help="amr model")
    parser.add_argument("--AMR_path", default='/srv/data1/abhidipbhatt/llava/all_doc_amrs_mu3sig', type=str, required=False, help="amr graph files path") #/srv/data1/abhidipbhatt/data/coco/amr/ibm_single_amr
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--freeze_amr", action='store_true', help="Whether to run with AMR.")
    
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for eval.")
    parser.add_argument("--num_workers", default=4, type=int, help="Workers in dataloader.")
    parser.add_argument("--num_train_epochs", default=3, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--logging_steps', type=int, default=2000, help="Log every X steps.")
    parser.add_argument('--save_steps', type=int, default=55470,
                        help="Save checkpoint every X steps. Will also perform evaluatin.")
    
    parser.add_argument("--local_rank", type=int, default=0,
                        help="For distributed training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before backward.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial lr.")
    parser.add_argument("--weight_decay", default=0.05, type=float, help="Weight deay.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam.")
    parser.add_argument("--max_grad_norm", default=5.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup.")
    parser.add_argument("--scheduler", default='linear', type=str, help="constant or linear or")
    parser.add_argument("--output_dir", default='/srv/data1/abhidipbhatt/models/lxmert/lxmert_amr/doc_amr_10ep_scratch_10k_2', type=str, required=False,
                        help="The input data dir with all required files.")
   
    
    #parser.add_argument()
    
    args = parser.parse_args()
     # Setup CUDA, GPU & distributed training
    local_rank = ensure_init_process_group(local_rank=args.local_rank)
    args.local_rank = local_rank
    args.num_gpus = get_world_size()
    args.distributed = args.num_gpus > 1
    args.device = torch.device('cuda')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    synchronize()
    
    lxmert_tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = LxmertForQuestionAnswering(config, num_qa_labels = 3129)#.from_pretrained(args.model_name_or_path) #(config, num_qa_labels = 3129)
    
    if model.num_qa_labels!=3129 or "lxmert-base-uncased" in args.model_name_or_path: 
        model.answer_head.logit_fc[3] = nn.Linear(1536, 3129)
        model.num_qa_labels = 3129
        model.config.num_qa_labels = 3129
        
    amr_tokenizer = AMRBartTokenizer.from_pretrained(args.AMR_model_path)
    partial_collate = partial(collate_fn_AMR, lxmert_tokenizer=lxmert_tokenizer, amr_tokenizer=amr_tokenizer)
    AMR_model = BartForConditionalGeneration.from_pretrained(args.AMR_model_path)
    amr_vocab_size = len(amr_tokenizer)
    #print("frm main amr_vocab_size:{}".format(amr_vocab_size))
    AMR_model.resize_token_embeddings(amr_vocab_size)
    AMR_model = AMR_model.model.encoder
    #print("from main:{}".format(AMR_model.embed_tokens.weight.size()))
    if args.freeze_amr:
        for param in AMR_model.parameters():
            param.requires_grad=False
        AMR_model = AMR_model.eval()
        AMR_model.train = disabled_train
        logging.info("freeze amr model")
    else:
        logging.info("trainable amr model")
    model.set_AMR_model(AMR_model)

    if args.resume_training_path:
        state_dict = torch.load(args.resume_training_path+'/pytorch_model.bin', map_location="cpu")
        logging.info('loading weights from:{}'.format(args.resume_training_path+'/pytorch_model.bin'))
        model.load_state_dict(state_dict)
    #model.lxmert. =AMR_model
    model.to(args.device)
    
    if args.do_train:
        dset = VQADataset('train', args.data_dir)
        tset = VQATorchAMRDataset(dset,args.tsv_feat_path, AMR_model_path=args.AMR_model_path, AMR_path=args.AMR_path, mode_training=True)
        data_loader = make_data_loader(tset, partial_collate, args, is_train=True)
        #train(args, data_loader, model)
        #print("-------=====================================--------------")

        vset = VQADataset('valid', args.data_dir)
        vtset = VQATorchAMRDataset(vset,args.tsv_feat_path, AMR_model_path=args.AMR_model_path, AMR_path=args.AMR_path, mode_training=False)
        val_loader = make_data_loader(vtset, partial_collate, args, is_train=False)
        #predict(model, vset, val_loader, args)
        train(args, data_loader, model, vset, val_loader)
        print("-------=====================================--------------")
        
        
        
        
        
    
# python -m torch.distributed.launch --nproc_per_node=1 --master_port=18121 main.py    
if __name__ == "__main__":
    main()