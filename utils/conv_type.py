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



class ProbMaskConvChannel(nn.Conv2d):
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

    @property
    def clamped_scores(self):
        return self.scores

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


class ProbMaskConvChannelDiscrete(nn.Conv2d):
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

    @property
    def clamped_scores(self):
        return self.scores

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
            # print(self.subnet.size(), self.weight.size(), self.subnet)
            w = self.weight * self.subnet
            x = F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            w = self.weight * self.subnet
            x = F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x

class ProbMaskConvChannelDiscrete(nn.Conv2d):
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

    @property
    def clamped_scores(self):
        return self.scores

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
            #self.subnet = GetMaskDiscreteBS.apply(self.scores)
            # print(self.subnet.size(), self.weight.size(), self.subnet)
            w = self.weight * self.subnet
            x = F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            w = self.weight * self.subnet
            x = F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x


class ProbMaskConvChannelDiscreteSpeedUp(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()[0], 1, 1, 1))
        self.subnet = None
        self.train_weights = False
        self.rescaling_para = nn.Parameter(torch.Tensor(1))
        self.temp_w = None
        self.count = 0
        if self.weight.size()[2] > 1:
            self.prune = True
        else:
            self.prune = False
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

    def forward(self, *inputs):
        # print(type(inputs), len(inputs))
        if len(inputs) > 1:
            x, mask = inputs
        else:
            x = inputs[0]
        if not self.train_weights:
            if self.count % 1 == 0:
                self.count += 1
                w = self.weight
                if self.prune:
                    eps = 1e-20
                    temp = parser_args.T
                    uniform0 = torch.rand_like(self.scores)
                    uniform1 = torch.rand_like(self.scores)
                    noise = -torch.log(torch.log(uniform0 + eps) / torch.log(uniform1 + eps) + eps)
                    self.subnet = GetMaskDiscrete.apply(torch.sigmoid((torch.log(self.clamped_scores + eps) - torch.log(1.0 - self.clamped_scores + eps) + noise) * temp))
                    self.subnet = self.subnet.bool()
                    if self.subnet.sum() == 0:
                        self.subnet[0] = True
                    # print("self.subnet", self.subnet)
                    # size = list(self.weight.size()[1:])
                    # size.insert(0, self.subnet.sum())
                    # w = torch.masked_select(self.weight, self.subnet).view(size)
                    w = self.weight[self.subnet.squeeze()]
                    # print("input:, weight:, self.weight.ori:", x.size(), w.size(), self.weight.size())
                if len(inputs) > 1:
                    # size = [w.size()[0], mask.sum(), w.size()[2], w.size()[3]]
                    # # print("size of final w, size of input mask", size, mask.size())
                    # w = torch.masked_select(w, mask.view(1, mask.nelement(), 1, 1)).view(size)
                    w = w[:, mask.squeeze()]
                self.temp_w = w
                # print("input:, weight:, self.weight.ori:", x.size(), w.size(), self.weight.size())
            else:
                w = self.temp_w
            x = F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
            # print("output:", x.size())
        else:
            w = self.weight * self.subnet
            x = F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x, self.subnet

class GrowEfficientConvProbReal(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()[0], 1, 1, 1))
        self.subnet = None
        self.train_weights = False
        if parser_args.score_init_constant is not None:
            self.scores.data = (
                    torch.ones_like(self.scores) * parser_args.score_init_constant
            )
        else:
            nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
        if self.out_channels == 10 or self.out_channels == 100:
            self.prune = False
            self.subnet = torch.ones_like(self.scores).cuda(parser_args.multigpu[0])
        else:
            self.prune = True
        self.rescaling_para = nn.Parameter(torch.Tensor(1))
        self.rescaling_para.data = torch.mean(self.clamped_scores).data

    @property
    def clamped_scores(self):
        return self.scores

    def fix_subnet(self):
        self.subnet = self.scores*GetMaskDiscreteBSReal.apply(self.scores)

    def forward(self, x):
        if self.prune:
            if not self.train_weights:
                self.subnet = self.scores*GetMaskDiscreteBSReal.apply(self.scores)
                w = self.weight * self.subnet
                x = F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
            else:
                w = self.weight * self.subnet
                x = F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x


class GetMaskDiscreteBS(autograd.Function):
    @staticmethod
    def forward(ctx, m_cont):
        output = (torch.rand_like(m_cont) <= m_cont).float()
        return output

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs

class GetMaskDiscreteBSReal(autograd.Function):
    @staticmethod
    def forward(ctx, m_cont):
        output = (torch.rand_like(m_cont) <= m_cont).float()
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_outputs):
        m_dis, = ctx.saved_variables
        grad_inputs = grad_outputs.clone()
        grad_inputs[m_dis == 0.0] = 0.0
        return grad_inputs

class GetMaskDiscrete(autograd.Function):
    @staticmethod
    def forward(ctx, m_cont):
        m_dis = (m_cont > 0.5).float()
        return m_dis

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs