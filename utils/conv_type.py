import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

import math

from args import args as parser_args

DenseConv = nn.Conv2d

class ProbMaskConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))  #Probability
        self.subnet = None                                            #Mask
        self.train_weights = False
        if parser_args.score_init_constant is not None:
            self.scores.data = (
                    torch.ones_like(self.scores) * parser_args.score_init_constant
            )
        else:
            nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

    @property
    def clamped_scores(self):
        return self.scores

    def fix_subnet(self):
        self.subnet = (torch.rand_like(self.scores) < self.clamped_scores).float()

    def forward(self, x):
        if not self.train_weights:                                      #training
            if not parser_args.discrete:
                eps = 1e-20
                temp = parser_args.T
                uniform0 = torch.rand_like(self.scores)
                uniform1 = torch.rand_like(self.scores)
                noise = -torch.log(torch.log(uniform0 + eps) / torch.log(uniform1 + eps) + eps)
                self.subnet = torch.sigmoid((torch.log(self.clamped_scores + eps) - torch.log(1.0 - self.clamped_scores + eps) + noise) * temp)
            else:
                self.subnet = (torch.rand_like(self.scores) < self.clamped_scores).float()
            w = self.weight * self.subnet
            x = F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
        else:                                                           #testing
            w = self.weight * self.subnet
            x = F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x



class ProbMaskConvChannnel(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()[0], 1, 1, 1))
        self.subnet = None
        self.train_weights = False
        self.rescaling_para = nn.Parameter(torch.Tensor(1))
        if parser_args.score_init_constant is not None:
            self.scores.data = (
                    torch.ones_like(self.scores) * parser_args.score_init_constant
            )
        else:
            nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

    def fix_subnet(self):
        self.subnet = (torch.rand_like(self.scores) < self.clamped_scores).float()

    def forward(self, x):
        if not self.train_weights:
            if not parser_args.discrete:
                eps = 1e-20
                temp = parser_args.T
                uniform0 = torch.rand_like(self.scores)
                uniform1 = torch.rand_like(self.scores)
                noise = -torch.log(torch.log(uniform0 + eps) / torch.log(uniform1 + eps) + eps)
                self.subnet = torch.sigmoid((torch.log(self.clamped_scores + eps) - torch.log(1.0 - self.clamped_scores + eps) + noise) * temp)
            else:
                self.subnet = (torch.rand_like(self.scores) < self.clamped_scores).float()
            w = self.weight * self.subnet
            x = F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            w = self.weight * self.subnet
            x = F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x


class ProbMaskConvChannnelDiscrete(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()[0], 1, 1, 1))
        self.subnet = None
        self.train_weights = False
        self.rescaling_para = nn.Parameter(torch.Tensor(1))
        if parser_args.score_init_constant is not None:
            self.scores.data = (
                    torch.ones_like(self.scores) * parser_args.score_init_constant
            )
        else:
            nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

    def fix_subnet(self):
        self.subnet = (torch.rand_like(self.scores) < self.clamped_scores).float()

    def forward(self, x):
        if not self.train_weights:
            eps = 1e-20
            temp = parser_args.T
            uniform0 = torch.rand_like(self.scores)
            uniform1 = torch.rand_like(self.scores)
            noise = -torch.log(torch.log(uniform0 + eps) / torch.log(uniform1 + eps) + eps)
            self.subnet = GetMaskDiscrete.apply(torch.sigmoid((torch.log(self.clamped_scores + eps) - torch.log(1.0 - self.clamped_scores + eps) + noise) * temp))
            print(self.subnet.size(), self.weight.size(), self.subnet)
            w = self.weight * self.subnet
            x = F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            w = self.weight * self.subnet
            x = F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x


class GetMaskDiscrete(autograd.Function):
    @staticmethod
    def forward(ctx, m_cont):
        m_dis = (m_cont > 0.5).float()
        return m_dis

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs