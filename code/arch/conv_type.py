import sys

from torch import tensor
sys.path.insert(0, './')

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

import math

import args as Args


DenseConv = nn.Conv2d


class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # Get the subnetwork by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())

        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None


# Not learning weights, finding subnet
class SubnetConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def init_scores(self, fan_scaled_score):
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        bound = Args.args.score_init_scale
        if fan_scaled_score:
            fan = nn.init._calculate_correct_fan(self.scores, Args.args.mode)
            bound *= 1 / math.sqrt(fan)
        with torch.no_grad():
            self.scores.data.uniform_(-bound, bound)

    def set_prune_rate(self, prune_rate):
        self.prune_rate = prune_rate

    @property
    def clamped_scores(self):
        return self.scores.abs()

    def forward(self, x):
        subnet = GetSubnet.apply(self.clamped_scores, self.prune_rate)
        w = self.weight * subnet
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x

class BinMask(autograd.Function):
    @staticmethod
    def forward(ctx, tensor_in, stochastic):
        if stochastic:
            prob = tensor_in.view(-1,1)
            selection = torch.multinomial(torch.cat((1-prob, prob),1), 1).float()
            return 2 * selection.reshape(tensor_in.shape) - 1.0
        else:
            signed_weights = tensor_in.sign()
            signed_weights[signed_weights == 0] = 1.0
            return signed_weights.float()

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None

class BinConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.freeze_rand_flag = False
        self.binweight = nn.Parameter(torch.Tensor(self.weight.size()), requires_grad=False)

    @property
    def hard_sigmoid(self):
        return torch.clamp((self.weight + 1.0) / 2.0, min=0, max=1)

    def freeze_rand(self):
        self.freeze_rand_flag = True

    def unfreeze_rand(self):
        self.freeze_rand_flag = False

    def set_stochastic_flag(self, stochastic):
        self.stochastic = stochastic

    def forward(self, x):
        self.weight.data = torch.clamp(self.weight.data, min=-1,max=1)
        if self.stochastic:
            if not self.freeze_rand_flag:
                w = BinMask.apply(self.hard_sigmoid, self.stochastic)
                self.binweight.data = w.detach()
            else:
                w = self.binweight
        else:
            w = BinMask.apply(self.weight, self.stochastic)
            self.binweight.data = w.detach()
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x


"""
Fixed subnets 
"""


class FixedSubnetConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

    def set_prune_rate(self, prune_rate):
        self.prune_rate = prune_rate
        print("prune_rate_{}".format(self.prune_rate))

    def set_subnet(self):
        output = self.clamped_scores().clone()
        _, idx = self.clamped_scores().flatten().abs().sort()
        p = int(self.prune_rate * self.clamped_scores().numel())
        flat_oup = output.flatten()
        flat_oup[idx[:p]] = 0
        flat_oup[idx[p:]] = 1
        self.scores = torch.nn.Parameter(output)
        self.scores.requires_grad = False

    def clamped_scores(self):
        return self.scores.abs()

    def get_subnet(self):
        return self.weight * self.scores

    def forward(self, x):
        w = self.get_subnet()
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x

