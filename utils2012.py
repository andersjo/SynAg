from collections import Counter
import re
numberRegex = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+");
import shutil
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.init import *
from torch import optim

from operator import itemgetter

import numpy as np


class ConllEntry:
    def __init__(self, word, lemma, pos, args):

        self.word = word
        self.norm = normalize(word)
        self.lemma = lemma
        self.pos = pos
        self.args = args
        
def vocab(conll_path):
    wordsCount = Counter()
    lemCount = Counter()
    posCount = Counter()
    argsCount = Counter()


    with open(conll_path, 'r') as conllFP:
        for sentence in read_conll(conllFP):
            for node in sentence:
                if isinstance(node, ConllEntry):
                    wordsCount.update([node.norm])
                    lemCount.update([node.lemma])
                    posCount.update([node.pos ])
                    argsCount.update([node.args[x] for x in range(len(node.args))])

    return ({w: i for i, w in enumerate(wordsCount.keys())}, {p: i for i, p in enumerate(posCount.keys())},
            {l: i for i, l in enumerate(lemCount.keys())}, {r: i for i, r in enumerate(argsCount.keys())}, wordsCount, list(posCount.keys()), list(lemCount.keys()))

        
def read_conll(fh):
    root = ConllEntry('*root*', '*root*', 'ROOT-POS', '-')
    tokens = [root]
    fh.readline()
    for line in fh:
        tok = line.split()

        if len(tok) > 10:
            del(tok[-1])
            #tokens.append(ConllEntry(tok[3], tok[6], tok[4], tok[11:]))
            tokens.append(ConllEntry(tok[3], tok[6], tok[4], [x.strip('()') for x in tok[11:]]))

    if len(tokens) > 1:
        yield tokens
        
def write_conll(fn, conll_gen):
    with open(fn, 'w') as fh:
        for sentence in conll_gen:
            for entry in sentence[1:]:
                fh.write(str(entry) + '\n')
            fh.write('\n')


def normalize(word):
    return 'NUM' if numberRegex.match(word) else word.lower()


def extract_sent(data, emb_dict):

    h = open(data, 'r')
    sent_list = []
    sent = []    
    pred_flag = 0  # Ensures only sentences with a predicate in them are added to sent_list
    
    h.readline()
    for line in h:   
        tok = line.split()
        if len(tok)>1:
            del(tok[-1])
            
        if len(tok)>11:
            if tok[3] in emb_dict.keys() or tok[3] == '/.':
                pre_emb = "KNOWN"
            else:
                pre_emb = 'UNK'
        
            if tok[11] =='(V*)':
                pred_flag = 1
 
            sent.append(((tok[3], pre_emb), tok[6], tok[4], [x.strip('()') for x in tok[11:]]))

        elif pred_flag == 1:
            sent_list.append(sent)
            sent = []
            pred_flag = 0

    return sent_list
    h.close()
    
def extract_targets(sent, preds, roles):
    target_tensor = torch.zeros(len(preds), len(sent))
    for i in range(len(preds)):
        for j in range(len(sent)):
            target_tensor[i][j] = roles[sent[j][3][i]]
    return torch.tensor(target_tensor, dtype=torch.long)
