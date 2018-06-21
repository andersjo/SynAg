from collections import Counter
import re
import shutil
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import *
from torch import optim
from operator import itemgetter
import imp
import utils2012
import numpy as np
import torch.nn.utils.rnn as rnn_utils
import time
import operator

numberRegex = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+");
def normalize(word):
    return 'NUM' if numberRegex.match(word) else word.lower()

torch.set_default_tensor_type('torch.cuda.FloatTensor')

external_embedding_fp = open('/homes/jds76/virtualenv/Project/data/sskip.100.vectors', 'r')
external_embedding_fp.readline()
external_embedding = {line.split(' ')[0]: [float(f) for f in line.strip().split(' ')[1:]] for line in
                            external_embedding_fp}
external_embedding_fp.close()

ext_train_data = #Path to file
ext_dev_data = #Path to file
ext_test_data = #Path to file

train_data = #Path to file


imp.reload(utils2012)
w2i, p2i, l2i, r2i, words, pos, lems = list(utils2012.vocab(ext_train_data))
train_sentences = utils2012.extract_sent(ext_train_data, external_embedding)
dev_sentences = utils2012.extract_sent(ext_dev_data, external_embedding)
role_list = list(r2i.keys())

class SynAg(nn.Module):
    def __init__(self, w2i, p2i, l2i, r2i, word_vocab, pos_vocab, lem_vocab, hidden_size, cell_size, num_layers):
        super().__init__()
        
        self.w2i = w2i
        self.p2i = p2i
        self.l2i = l2i
        self.r2i = r2i
        
        self.word_vocab = word_vocab
        self.pos_vocab = pos_vocab
        self.lem_vocab = lem_vocab
        
        self.word_emb = nn.Embedding(len(self.word_vocab), 100)
        self.pos_emb = nn.Embedding(len(self.pos_vocab), 16)
        self.lem_emb = nn.Embedding(len(self.lem_vocab), 100)
        
        self.hidden_size = hidden_size
        self.cell_size = cell_size
        self.num_layers = num_layers
        
        self.arch = nn.LSTM(input_size=316, hidden_size = self.hidden_size, num_layers = self.num_layers, bidirectional = True, batch_first= True )
        
        self.cell_to_role = nn.Linear(4*self.hidden_size, len(r2i), bias = False)     

    def repn(self, sents):
        count = 0
        pred_count = 0
        sent_count = 0
        rep_dict = {}
        rep_len = {}
  
        self.pred_idxs_dict = {}
    
        sent_order_dict = {}
        sent_order_count = 0
        
        for sent in sents:         
            sent_preds = [x for x in sent if 'V*' in x[3]]
            ord_pred_count = 0
            
            for pred in sent_preds:
                rep_list = []
                ord_pred_count +=1

                for word in sent:
                    if normalize(word[0][0]) in self.w2i:
                        ran_word_emb = self.word_emb(torch.tensor(self.w2i[normalize(word[0][0])]))
        
                        if word[0][1] == 'KNOWN':
                            if word[0][0] == '/.':
                                pre_emb = torch.tensor(external_embedding['</s>'])
                            else:
                                pre_emb = torch.tensor(external_embedding[word[0][0]])
                        else:
                            pre_emb = self.word_emb(torch.tensor(self.w2i[normalize(word[0][0])]))
            
                    else:
                        ran_word_emb = torch.randn(100)
                        pre_word_emb = torch.randn(100)
            
                    if word[0][0] == pred[0][0]:
                        
                        if word[1] in self.l2i:
                            ran_lem_emb = self.lem_emb(torch.tensor(self.l2i[word[1]]))
                        else:
                            ran_lem_emb = torch.randn(100)
                        self.pred_idxs_dict[pred_count] = (sent.index(word), sent_count)
                        pred_count += 1
                    
                    else:
                        ran_lem_emb = torch.zeros((100))
                    
                    ran_pos_emb = self.pos_emb(torch.tensor(self.p2i[word[2]]))            
                    new_ent = torch.cat((ran_word_emb, pre_emb, ran_lem_emb, ran_pos_emb))

                    rep_list.append(new_ent)
                
                rep_tensor = torch.stack([x for x in rep_list])
                rep_dict[count] = rep_tensor

                rep_len[count] = rep_tensor.shape[0]
                sent_order_dict[count] = (sent_order_count, ord_pred_count-1)
                count += 1
                
            sent_count +=1
            sent_order_count +=1
            
        sorted_lengths = sorted(rep_len.items(), key=operator.itemgetter(1), reverse=True)
        sorted_sentences = [rep_dict[x[0]] for x in sorted_lengths]
        sent_lens = [x.shape[0] for x in sorted_sentences]
        

        packed_seq = rnn_utils.pack_sequence(sorted_sentences)
        
        self.sorted_idxs = [self.pred_idxs_dict[x[0]] for x in sorted_lengths]
        sent_order_list = [sent_order_dict[x[0]] for x in sorted_lengths]
        return packed_seq, sent_lens, sent_order_list    

    def forward(self, sents):
        packed_sequence, packed_lengths, order_list = self.repn(sents)     
        
        self.hidden = torch.randn(2*self.num_layers, len(packed_lengths), self.hidden_size)
        self.cell = torch.randn(2*self.num_layers, len(packed_lengths), self.cell_size) 
        
        lstm_out, (self.hidden, self.cell) = self.arch(packed_sequence, (self.hidden, self.cell))
        lstm_out = rnn_utils.pad_packed_sequence(lstm_out)[0]
        lstm_out = lstm_out.permute(1, 0, 2)

        col_list = []
        for m in range(len(packed_lengths)):
            row_list = []
            for n in range(max(packed_lengths)):
                if torch.sum(lstm_out[m][n]) != 0:
                    row_list.append(torch.cat((lstm_out[m][n], lstm_out[m][self.sorted_idxs[m][0]])))
                else:
                    row_list.append(torch.zeros(4*self.hidden_size))
            row_tensor = torch.stack([x for x in row_list])
 
            col_list.append(row_tensor)
        pred_tensor = torch.stack([x for x in col_list])
        
        #pred_tensor = torch.randn(len(packed_lengths), max(packed_lengths), 4*self.hidden_size)
        #for m in range(len(packed_lengths)):
            #for n in range(max(packed_lengths)):
                #if torch.sum(lstm_out[m][n]) != 0:
                    #pred_tensor[m][n] = torch.cat((lstm_out[m][n], lstm_out[m][self.sorted_idxs[m][0]]))
                
                
        role_space = self.cell_to_role(pred_tensor)
        role_scores = F.log_softmax(role_space, dim = 2)
        
      
        return role_scores, packed_lengths, order_list#, self.sorted_idxs
        
        

      

def get_targets(sent, roles, pred_num):
    target = []
    for i in range(len(sent)):
        target.append(roles[sent[i][3][pred_num]])
    return torch.tensor(target)
      
        
my_model = SynAg(w2i, p2i, l2i, r2i, words, pos, lems, 512, 512, 4)
my_model.cuda()

optimizer = optim.SGD(my_model.parameters(), lr=0.01)
loss_function = nn.NLLLoss()

start = time.time()


for epoch in range(10):

    t_loss = 0

    for k in range(5): 

        batch = train_sentences[50*k:50*(k+1)]

        optimizer.zero_grad()
    
        scores, p_lengths, order_list = my_model(batch)
    
        sen_list = []
        for i in range(len(p_lengths)):
            sen_list.append(scores[i][:p_lengths[i]])  

        targ_list = []
        
        for x in order_list:
            #print(x)
            targets = get_targets(batch[x[0]], r2i, x[1])
            targ_list.append(targets)
        
        targ_batch = rnn_utils.pack_sequence([x for x in targ_list])
        sen_batch = rnn_utils.pack_sequence([x for x in sen_list])
    
        loss = loss_function(sen_batch.data, targ_batch.data)
    
        #print(loss)
        t_loss += loss.item()
        loss.backward()
        optimizer.step()
    
    print(t_loss)
    
print(time.time() - start)












