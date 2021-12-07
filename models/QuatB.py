import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from .Model import Model
from numpy.random import RandomState


class QuatB(Model):
    def __init__(self, config):
        super(QuatB, self).__init__(config)
        self.criterion = nn.Softplus()
        self.init_weights()

    def init_weights(self):
        self.emb_s_a, self.emb_x_a, self.emb_y_a, self.emb_z_a, \
        self.rel_s_b, self.rel_x_b, self.rel_y_b, self.rel_z_b = \
            torch.load('%s' % self.config.load_path)

        self.emb_s_a.weight.requires_grad = True
        self.emb_x_a.weight.requires_grad = True
        self.emb_y_a.weight.requires_grad = True
        self.emb_z_a.weight.requires_grad = True
        self.rel_s_b.weight.requires_grad = True
        self.rel_x_b.weight.requires_grad = True
        self.rel_y_b.weight.requires_grad = True
        self.rel_z_b.weight.requires_grad = True
        
    def _calc(self, s_a, x_a, y_a, z_a, s_c, x_c, y_c, z_c, s_b, x_b, y_b, z_b):
    
        denominator_b = torch.sqrt(s_b ** 2 + x_b ** 2 + y_b ** 2 + z_b ** 2)
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
                torch.mean(self.criterion(score * self.batch_y)) + self.config.lmbda * regul +   self.config.lmbda * regul2
        )

    def forward(self):
        s_a = self.emb_s_a(self.batch_h)
        x_a = self.emb_x_a(self.batch_h)
        y_a = self.emb_y_a(self.batch_h)
        z_a = self.emb_z_a(self.batch_h)

        s_c = self.emb_s_a(self.batch_t)
        x_c = self.emb_x_a(self.batch_t)
        y_c = self.emb_y_a(self.batch_t)
        z_c = self.emb_z_a(self.batch_t)

        s_b = self.rel_s_b(self.batch_r)
        x_b = self.rel_x_b(self.batch_r)
        y_b = self.rel_y_b(self.batch_r)
        z_b = self.rel_z_b(self.batch_r)
        

        score = self._calc(s_a, x_a, y_a, z_a, s_c, x_c, y_c, z_c, s_b, x_b, y_b, z_b)
        regul = (torch.mean( torch.abs(s_a) ** 2)
                 + torch.mean( torch.abs(x_a) ** 2)
                 + torch.mean( torch.abs(y_a) ** 2)
                 + torch.mean( torch.abs(z_a) ** 2)
                 + torch.mean( torch.abs(s_c) ** 2)
                 + torch.mean( torch.abs(x_c) ** 2)
                 + torch.mean( torch.abs(y_c) ** 2)
                 + torch.mean( torch.abs(z_c) ** 2)
                 )
        regul2 =  (torch.mean( torch.abs(s_b) ** 2 )
                 + torch.mean( torch.abs(x_b) ** 2 )
                 + torch.mean( torch.abs(y_b) ** 2 )
                 + torch.mean( torch.abs(z_b) ** 2 ))
        
        return self.loss(score, regul, regul2)

    def predict(self):
    
        s_a = self.emb_s_a(self.batch_h)
        x_a = self.emb_x_a(self.batch_h)
        y_a = self.emb_y_a(self.batch_h)
        z_a = self.emb_z_a(self.batch_h)

        s_c = self.emb_s_a(self.batch_t)
        x_c = self.emb_x_a(self.batch_t)
        y_c = self.emb_y_a(self.batch_t)
        z_c = self.emb_z_a(self.batch_t)

        s_b = self.rel_s_b(self.batch_r)
        x_b = self.rel_x_b(self.batch_r)
        y_b = self.rel_y_b(self.batch_r)
        z_b = self.rel_z_b(self.batch_r)
        
        score = self._calc(s_a, x_a, y_a, z_a, s_c, x_c, y_c, z_c, s_b, x_b, y_b, z_b)
        return score.cpu().data.numpy()
