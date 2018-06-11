from collections import Counter
from pathlib import Path
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
    def __init__(self, word, lemma, pos, pred_args):

        self.word = word
        self.norm = normalize(word)
        self.lemma = lemma
        self.pos = pos
        self.pred = pred_args[:2]
        self.args = pred_args[2:]

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
    root = ConllEntry('*root*', '*root*', 'ROOT-POS', '_')
    tokens = [root]
    for line in fh:
        tok = line.split()

        if len(tok) > 0:
            tokens.append(ConllEntry(tok[1], tok[2], tok[4], tok[12:]))
            #print(tok[14:])
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

    f = open(data, 'r')

    sent_list = []
    sent_dict = {}
    sent = []
    
    num_preds = 0

    for line in f:
        line = line.split()
 
        if len(line) > 11:
            if line[12] == 'Y':
                num_preds = num_preds + 1
            
            if line[1] in emb_dict.keys() or line[1] == '.':
                sent.append(((line[1], "KNOWN"), line[2], line[4], line[12], line[14:]))
            else:
                sent.append(((line[1], "UNK"), line[2], line[4], line[12], line[14:]))
        else:
            if num_preds > 0:
                sent_list.append(sent)
                num_preds = 0
            sent = []
        
        
    f.close()
    
    return sent_list
