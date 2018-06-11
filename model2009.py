import re
from collections import Counter
from pathlib import Path
import shutil
import torch.nn as nn
import torch.nn.functional as F
# from torch.nn import Parameter
from torch.nn.init import *
from torch import optim
from operator import itemgetter
import numpy as np
import utils2009
import imp


numberRegex = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+")
def normalize(word):
    return 'NUM' if numberRegex.match(word) else word.lower()


    
def extract_targets(sent, preds, roles):
    target_tensor = torch.zeros(len(preds), len(sent))
    for i in range(len(preds)):
        for j in range(len(sent)):
            target_tensor[i][j] = roles[sent[j][4][i]]
    return torch.tensor(target_tensor, dtype=torch.long)
    
w2i, p2i, l2i, r2i, words, pos, lems = list(utils2009.vocab(train_data))

external_embedding_fp = open('../sskip.100.vectors', 'r')
external_embedding_fp.readline()
external_embedding = {line.split(' ')[0]: [float(f) for f in line.strip().split(' ')[1:]] for line in
                            external_embedding_fp}
external_embedding_fp.close()



class SynAg(nn.Module):
    def __init__(self, w2i, p2i, l2i, r2i, word_vocab, pos_vocab, lem_vocab):
        super().__init__()
        
        self.w2i = w2i
        self.p2i = p2i
        self.l2i = l2i
        self.r2i = r2i
        
        self.word_vocab = word_vocab
        self.pos_vocab = pos_vocab
        self.lem_vocab = lem_vocab
        
        self.word_emb = nn.Embedding(len(word_vocab), 100)
        self.pos_emb = nn.Embedding(len(pos_vocab), 16)
        self.lem_emb = nn.Embedding(len(lem_vocab), 100)
        
        self.arch = nn.LSTM(input_size=316, hidden_size = 512, num_layers = 1, bidirectional = True, batch_first= True )
        
        self.cell_to_role = nn.Linear(2048, len(r2i))
        
        
    def repn(self, sent, preds):
        m = 0
        pred_idxs = []
        rep_tensor = torch.randn(len(preds), len(sent), 316)
        for pred in preds:
            n = 0
            for word in sent:
                ran_word_emb = self.word_emb(torch.tensor(self.w2i[normalize(word[0][0])]))
            
                if word[0][1] == 'KNOWN':
                    if word[0][0] == '.':
                        pre_emb = torch.tensor(external_embedding['</s>'])
                    else:
                        pre_emb = torch.tensor(external_embedding[word[0][0]])
                else:
                    pre_emb = self.word_emb(torch.tensor(self.w2i[normalize(word[0][0])]))
            
                if word[0][0] == pred[0][0]:
                    ran_lem_emb = self.lem_emb(torch.tensor(self.l2i[word[1]]))
                    pred_idxs.append(sent.index(word))
                
                else:
                    ran_lem_emb = torch.zeros((100))
                    
                ran_pos_emb = self.pos_emb(torch.tensor(self.p2i[word[2]]))
            
                new_ent = torch.cat((ran_word_emb, pre_emb, ran_lem_emb, ran_pos_emb))
                rep_tensor[m][n] = new_ent

                n = n + 1
            m = m + 1
            
        return rep_tensor, pred_idxs
    
    def forward(self, sent, preds):

        sent_tensor, pred_idxs = self.repn(sent, preds)
        #print("shape of sent_tensor = ", sent_tensor.shape)        
        self.hidden = torch.randn(2, len(preds), 512)
        self.cell = torch.randn(2, len(preds), 512)           

        lstm_out, (self.hidden, self.cell) = self.arch(sent_tensor, (self.hidden, self.cell))
        #print("lstm_output shape is ", lstm_out.shape)
       
        pred_tensor = torch.randn(len(preds), len(sent), 2048)
        for m in range(len(preds)):
            for n in range(len(sent)):
                pred_tensor[m][n] = torch.cat((lstm_out[m][n], lstm_out[m][pred_idxs[m]]))
        
        #print("shape of pred_tensor", pred_tensor.shape)

        
        role_space = self.cell_to_role(pred_tensor)
        #print("role_space shape is", role_space.shape)

        role_scores = F.log_softmax(role_space, dim = 2)
        #print("role_scores shape is", role_scores.shape)
        #print("checkpoint")

        return role_scores 
        
        
        
loss_function = nn.NLLLoss()
optimizer = optim.SGD(my_model.parameters(), lr=0.01)
my_model = SynAg(w2i, p2i, l2i, r2i, words, pos, lems)

for sent in train_sents[0:100]:
    sent_preds = [x for x in sent if x[3] == 'Y']
    
    targs = extract_targets(sent, sent_preds, r2i)
    
    my_model.zero_grad()
    scores = my_model(sent, sent_preds)
    n_scores = scores.permute(0, 2, 1)
    
    loss = loss_function(n_scores, targs)
    print(loss)
    loss.backward()
    optimizer.step()
