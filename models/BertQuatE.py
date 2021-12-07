from IPython import embed
import torch
import torch.nn.functional as F
from tqdm import tqdm
import itertools
import sys
import numpy as np
import random
import torch
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from IPython import embed
import os
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from .Model import Model
from IPython import embed
import math
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from .Model import Model
from numpy.random import RandomState


class bertEncoder(nn.Module):
    def __init__(self, config):
        super(bertEncoder, self).__init__()
        vocab_file = "bert-base-chinese"
        model_file = "bert-base-chinese"
        self.tokenizer = BertTokenizer.from_pretrained(vocab_file, cache_dir='embedding')
        self.to_idx = {}
        idx_to_tokens = self.tokenizer.ids_to_tokens
        for i in idx_to_tokens:
            self.to_idx[idx_to_tokens[i]] = i
            
        self.bert = BertModel.from_pretrained(model_file, cache_dir='embedding')
        self.UNK = 100
        self.PAD = 0
        
    def sentence_to_tensor(self, s):
        B = len(s)
        ws = []
        L = 0
        for i in range(B):
            r = s[i]
            # replace number with blank space
            # for ch in '0123456789_':
            #     r = r.replace(ch, ' ')
            words = [i for i in self.tokenizer.tokenize(r)]
            words = ['[CLS]'] + words + ['[SEP]']
            ws.append(words)
            if len(words) > L:
                L = len(words)

        if torch.cuda.is_available():
            tokens_tensor = torch.zeros(B, L).long().cuda()
            seg_tensor = torch.zeros(B, L).long().cuda()
        else:
            tokens_tensor = torch.zeros(B, L).long()
            seg_tensor = torch.zeros(B, L).long()
        for i in range(B):
            for j in range(L):
                l = len(ws[i])
                if j < l:
                    tokens_tensor[i][j] = self.to_idx.get(ws[i][j], self.UNK)
                else:
                    tokens_tensor[i][j] = self.PAD

        return tokens_tensor, seg_tensor
        
    def forward(self, s):
        tokens_tensor, seg_tensor = self.sentence_to_tensor(s)
        _, y = self.bert(tokens_tensor, seg_tensor)
        return y
        

class BertQuatE(Model):
    def __init__(self, config):
        super(BertQuatE, self).__init__(config)
        if torch.cuda.is_available():
            self.emb_s_a = nn.Embedding(self.config.entTotal, self.config.hidden_size).cuda()
            self.emb_x_a = nn.Embedding(self.config.entTotal, self.config.hidden_size).cuda()
            self.emb_y_a = nn.Embedding(self.config.entTotal, self.config.hidden_size).cuda()
            self.emb_z_a = nn.Embedding(self.config.entTotal, self.config.hidden_size).cuda()
            self.rel_s_b = nn.Embedding(self.config.relTotal, self.config.hidden_size).cuda()
            self.rel_x_b = nn.Embedding(self.config.relTotal, self.config.hidden_size).cuda()
            self.rel_y_b = nn.Embedding(self.config.relTotal, self.config.hidden_size).cuda()
            self.rel_z_b = nn.Embedding(self.config.relTotal, self.config.hidden_size).cuda()
        else:
            self.emb_s_a = nn.Embedding(self.config.entTotal, self.config.hidden_size)
            self.emb_x_a = nn.Embedding(self.config.entTotal, self.config.hidden_size)
            self.emb_y_a = nn.Embedding(self.config.entTotal, self.config.hidden_size)
            self.emb_z_a = nn.Embedding(self.config.entTotal, self.config.hidden_size)
            self.rel_s_b = nn.Embedding(self.config.relTotal, self.config.hidden_size)
            self.rel_x_b = nn.Embedding(self.config.relTotal, self.config.hidden_size)
            self.rel_y_b = nn.Embedding(self.config.relTotal, self.config.hidden_size)
            self.rel_z_b = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.emb_s_a.weight.requires_grad=False
        self.emb_x_a.weight.requires_grad=False
        self.emb_y_a.weight.requires_grad=False
        self.emb_z_a.weight.requires_grad=False
        self.rel_s_b.weight.requires_grad=False
        self.rel_x_b.weight.requires_grad=False
        self.rel_y_b.weight.requires_grad=False
        self.rel_z_b.weight.requires_grad=False
        
        self.bert = bertEncoder(self.config)
        self.project_ent = nn.Linear(768, self.config.hidden_size * 4)
        self.project_rel = nn.Linear(768, self.config.hidden_size * 4)
        self.criterion = nn.Softplus()
        self.init()

    def init(self):
        self.idx2ent = {}
        self.idx2rel = {}

        # Pre-trained Embedding
        if 'word' in self.config.bert_mode:
            file_name = 'word2id.txt'
        elif 'definition' in self.config.bert_mode:
            file_name = 'definition2id.txt'
        elif 'all' in self.config.bert_mode:
            file_name = 'all2id.txt'

        # Dataset
        if 'WN' in self.config.in_path:
            if 'WN18RR' in self.config.in_path:
                self.print_iter = 500
                self.save_iter = 5000
            elif 'WN18' in self.config.in_path:
                self.print_iter = 2000
                self.save_iter = 20000
            rel_name = 'relation2id.txt'
        elif 'FB' in self.config.in_path:
            if 'FB15K237' in self.config.in_path:
                self.print_iter = 2000
                self.save_iter = 20000
            elif 'FB15K' in self.config.in_path:
                self.print_iter = 2000
                self.save_iter = 20000
            rel_name = 'rel2id.txt'
        elif 'army' in self.config.in_path:
            self.print_iter = 2000
            self.save_iter = 20000
            rel_name = 'rel2id.txt'
        elif 'bid' in self.config.in_path:
            self.print_iter = 2000
            self.save_iter = 20000
            rel_name = 'relation2id.txt'
            
        for line in open("%s/%s" % (self.config.in_path, file_name)):
            line = line.strip()
            if len(line.split('\t')) != 2:
                continue
            e, id = line.split('\t')
            self.idx2ent[int(id)] = e
        print('idx2ent: ' + str(len(self.idx2ent)))

        for line in open("%s/%s" % (self.config.in_path, rel_name)):
            line = line.strip()
            if len(line.strip().split('\t')) != 2:
                continue
            e, id = line.strip().split('\t')
            self.idx2rel[int(id)] = e
        self.accum_loss = 0.0
        self.updates = 0
        self.alpha = 0.99

    def save(self):
        self.eval()
        for i in tqdm(range(self.config.entTotal)):
            h = torch.LongTensor([i])
            s_a, x_a, y_a, z_a = self.get_ent_embedding(h)
            self.emb_s_a.weight.data[i: i+1, :] = s_a
            self.emb_x_a.weight.data[i: i+1, :] = x_a
            self.emb_y_a.weight.data[i: i+1, :] = y_a
            self.emb_z_a.weight.data[i: i+1, :] = z_a
        for i in tqdm(range(self.config.relTotal)):
            r = torch.LongTensor([i])
            s_b, x_b, y_b, z_b = self.get_rel_embedding(r)
            self.rel_s_b.weight.data[i: i+1, :] = s_b
            self.rel_x_b.weight.data[i: i+1, :] = x_b
            self.rel_y_b.weight.data[i: i+1, :] = y_b
            self.rel_z_b.weight.data[i: i+1, :] = z_b
        torch.save((self.emb_s_a.cpu(), self.emb_x_a.cpu(), self.emb_y_a.cpu(), self.emb_z_a.cpu(),\
            self.rel_s_b.cpu(), self.rel_x_b.cpu(), self.rel_y_b.cpu(), self.rel_z_b.cpu()), \
            "%s/BertQuatE_%s_%d.pt" % \
            (self.config.in_path, self.config.bert_mode, self.updates))
        self.emb_s_a = self.emb_s_a.cuda()
        self.emb_x_a = self.emb_x_a.cuda()
        self.emb_y_a = self.emb_y_a.cuda()
        self.emb_z_a = self.emb_z_a.cuda()
        self.rel_s_b = self.rel_s_b.cuda()
        self.rel_x_b = self.rel_x_b.cuda()
        self.rel_y_b = self.rel_y_b.cuda()
        self.rel_z_b = self.rel_z_b.cuda()
        
    def _calc(self, s_a, x_a, y_a, z_a, s_c, x_c, y_c, z_c, s_b, x_b, y_b, z_b):
        denominator_b = torch.sqrt(s_b ** 2+ x_b ** 2+ y_b ** 2+ z_b ** 2)
        s_b = s_b / denominator_b
        x_b = x_b / denominator_b
        y_b = y_b / denominator_b
        z_b = z_b / denominator_b

        A = s_a * s_b - x_a * x_b - y_a * y_b - z_a * z_b
        B = s_a * x_b + s_b * x_a + y_a * z_b - y_b * z_a
        C = s_a * y_b + s_b * y_a + z_a * x_b - z_b * x_a
        D = s_a * z_b + s_b * z_a + x_a * y_b - x_b * y_a

        score_r = (A * s_c + B * x_c + C * y_c + D * z_c)
        return -torch.sum(score_r, -1)

    def loss(self, score, regul, regul2):
        return (
                torch.mean(self.criterion(score * self.batch_y)) + self.config.lmbda * regul + self.config.lmbda * regul2
        )

    def forward(self):
        self.train()
        s_a, x_a, y_a, z_a = self.get_ent_embedding(self.batch_h)
        s_c, x_c, y_c, z_c = self.get_ent_embedding(self.batch_t)
        s_b, x_b, y_b, z_b = self.get_rel_embedding(self.batch_r)
        
        score = self._calc(s_a, x_a, y_a, z_a, s_c, x_c, y_c, z_c, s_b, x_b, y_b, z_b)

        regul = (torch.mean(torch.abs(s_a) ** 2)
                 + torch.mean(torch.abs(x_a) ** 2)
                 + torch.mean(torch.abs(y_a) ** 2)
                 + torch.mean(torch.abs(z_a) ** 2)
                 + torch.mean(torch.abs(s_c) ** 2)
                 + torch.mean(torch.abs(x_c) ** 2)
                 + torch.mean(torch.abs(y_c) ** 2)
                 + torch.mean(torch.abs(z_c) ** 2))

        regul2 = (torch.mean(torch.abs(s_b) ** 2)
                  + torch.mean(torch.abs(x_b) ** 2)
                  + torch.mean(torch.abs(y_b) ** 2)
                  + torch.mean(torch.abs(z_b) ** 2))

        loss = self.loss(score, regul, regul2)
        self.accum_loss = self.alpha * self.accum_loss + (1 - self.alpha) * loss.item()
        self.updates += 1
        if self.updates % self.print_iter == 0:
            print('%7d updates, loss: %10.8f ' % \
                (self.updates, self.accum_loss / (1 - self.alpha ** (self.updates))), flush=True)
        if self.updates % self.save_iter == 0:
            self.save()
        return loss

    def predict(self):
        self.eval()
        s_a = self.emb_s_a(self.batch_h)
        x_a = self.emb_x_a(self.batch_h)
        y_a = self.emb_y_a(self.batch_h)
        z_a = self.emb_z_a(self.batch_h)
        s_b = self.rel_s_b(self.batch_r)
        x_b = self.rel_x_b(self.batch_r)
        y_b = self.rel_y_b(self.batch_r)
        z_b = self.rel_z_b(self.batch_r)
        s_c = self.emb_s_a(self.batch_t)
        x_c = self.emb_x_a(self.batch_t)
        y_c = self.emb_y_a(self.batch_t)
        z_c = self.emb_z_a(self.batch_t)
        score = self._calc(s_a, x_a, y_a, z_a, s_c, x_c, y_c, z_c, s_b, x_b, y_b, z_b)
        return score.cpu().data.numpy()   
  
    def get_ent_embedding(self, x):
        h = self.bert([self.idx2ent[i.item()] for i in x])
        h = self.project_ent(h)
        h = F.tanh(h)
        s, x, y, z = torch.split(h, self.config.hidden_size, dim=-1)
        return s, x, y, z    
          
    def get_rel_embedding(self, x):
        h = self.bert([self.idx2rel[i.item()] \
            for i in x])
        h = F.tanh(self.project_rel(h))
        s, x, y, z = torch.split(h, self.config.hidden_size, dim=-1)
        return s, x, y, z
        