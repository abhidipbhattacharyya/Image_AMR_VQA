import penman
from penman.models.noop import NoOpModel
from penman import loads as loads_
from penman import load as load_
import copy
import re
import copy
import numpy as np


#/srv/data1/abhidipbhatt/lxmert/lib/python3.8/site-packages/penman/layout.py comment out line #153 logger.info.
# 143 if block. comment out logger and just have pass. Same for 179

def amr_preprocess(gstring, amr_model_type="None", mode = 'path'):

    #if amr_model_type.startswith('xfbai/AMRBART') or amr_model_type.startswith('roberta'):
    model = NoOpModel()
    if mode == 'path':
        out = load_(source=gstring, model=model)
    else:
        out = loads_(string=gstring, model=model)
    lin_tokens, adjacency_matrix = dfs_linearize(out[0])
    gstring = " ".join(lin_tokens)
    return gstring, lin_tokens, adjacency_matrix

def get_position(linearized_nodes, token):
    #print('{}'.format(linearized_nodes))
    return linearized_nodes.index(token)

def tokenize_encoded_graph(encoded):
    linearized = re.sub(r"(\".+?\")", r" \1 ", encoded)
    pieces = []
    for piece in linearized.split():
        if piece.startswith('"') and piece.endswith('"'):
            pieces.append(piece)
        else:
            piece = piece.replace("(", " ( ")
            piece = piece.replace(")", " ) ")
            piece = piece.replace(":", " :")
            piece = piece.replace("/", " / ")
            piece = piece.strip()
            pieces.append(piece)
    linearized = re.sub(r"\s+", " ", " ".join(pieces)).strip()
    return linearized.split(" ")

def dfs_linearize(graph):
    #print('---------------------------')
    graph_ = copy.deepcopy(graph)
    graph_.metadata = {}
    linearized = penman.encode(graph_)
    #print(linearized)
    linearized_nodes = tokenize_encoded_graph(linearized)
    #print(linearized_nodes)
    remap = {}
    for i in range(1, len(linearized_nodes)):
        nxt = linearized_nodes[i]
        lst = linearized_nodes[i - 1]
        if nxt == "/":
            remap[lst] = f"<pointer:{len(remap)}>"
    i = 1
    linearized_nodes_ = [linearized_nodes[0]]
    while i < (len(linearized_nodes)):
        nxt = linearized_nodes[i]
        lst = linearized_nodes_[-1]
        if nxt in remap:
            if lst == "(" and linearized_nodes[i + 1] == "/":
                nxt = remap[nxt]
                i += 1
            elif lst.startswith(":"):
                nxt = remap[nxt]
        linearized_nodes_.append(nxt)
        i += 1
    linearized_nodes = linearized_nodes_
    #print('=={}'.format(linearized_nodes))
    """
    Add: create graph on linearized nodes
    Will only be used for GNN
    """
    id2text = {ins.source: ins.target for ins in graph.instances()}
    id2position = {idx: get_position(linearized_nodes, text) for idx, text in id2text.items()}
    adjacency_matrix = np.zeros((len(linearized_nodes), len(linearized_nodes)))
    return linearized_nodes, adjacency_matrix

def amr_process(amr, mode='path'):
    if not isinstance(amr, str):
        print('not string#####: {}'.format(amr))
    lineared_amr, lineared_amr_token, amr_adjacency = amr_preprocess(amr,mode)
    return lineared_amr
    