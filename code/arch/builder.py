import args as Args
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import arch.conv_type
import arch.bn_type


class Builder(object):
    def __init__(self, conv_layer, bn_layer, first_layer=None, last_layer=None):
        self.conv_layer = conv_layer
        self.bn_layer = bn_layer
        self.first_layer = first_layer or conv_layer
        self.last_layer = last_layer or conv_layer

    def conv(self, kernel_size, in_planes, out_planes, stride=1, first_layer=False, last_layer=False):
        assert(not(first_layer and last_layer))
        conv_layer = self.conv_layer

        if first_layer:
            print(f"==> Building first layer with {str(self.first_layer)}")
            conv_layer = self.first_layer

        if last_layer:
            print(f"==> Building last layer with {str(self.last_layer)}")
            conv_layer = self.last_layer

        if kernel_size in [1, 3, 5, 7]:
            conv = conv_layer(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size-1)//2,
                bias=False,
            )
        else:
            return None

        if conv_layer == arch.conv_type.SubnetConv:
            if Args.args.fan_scaled_score_mode == "except_last" and last_layer == False:
                conv.init_scores(fan_scaled_score=True)
            elif Args.args.fan_scaled_score_mode == "only_last" and last_layer == True:
                conv.init_scores(fan_scaled_score=True)
            elif Args.args.fan_scaled_score_mode == "all":
                conv.init_scores(fan_scaled_score=True)
            else:
                conv.init_scores(fan_scaled_score=False)
        elif conv_layer == arch.conv_type.BinConv:
            conv.set_stochastic_flag(Args.args.binconv_stochastic)
            if Args.args.binconv_freeze_rand:
                conv.freeze_rand()

        self._init_conv(conv)

        return conv

    def conv3x3(self, in_planes, out_planes, stride=1, first_layer=False, last_layer=False):
        """3x3 convolution with padding"""
        c = self.conv(3, in_planes, out_planes, stride=stride, first_layer=first_layer, last_layer=last_layer)
        return c

    def conv1x1(self, in_planes, out_planes, stride=1, first_layer=False, last_layer=False):
        """1x1 convolution with padding"""
        c = self.conv(1, in_planes, out_planes, stride=stride, first_layer=first_layer, last_layer=last_layer)
        return c

    def conv7x7(self, in_planes, out_planes, stride=1, first_layer=False, last_layer=False):
        """7x7 convolution with padding"""
        c = self.conv(7, in_planes, out_planes, stride=stride, first_layer=first_layer, last_layer=last_layer)
        return c

    def conv5x5(self, in_planes, out_planes, stride=1, first_layer=False, last_layer=False):
        """5x5 convolution with padding"""
        c = self.conv(5, in_planes, out_planes, stride=stride, first_layer=first_layer, last_layer=last_layer)
        return c

    def batchnorm(self, planes, last_bn=False, last_layer=False):
        return self.bn_layer(planes)

    def activation(self, x):
        if Args.args.nonlinearity == "relu":
            return F.relu(x, inplace=False)
        elif Args.args.nonlinearity == "leaky_relu":
            return F.leaky_relu(x, negative_slope=Args.args.leaky_relu_slope, inplace=False)
        else:
            raise ValueError(f"{Args.args.nonlinearity} is not an initialization option!")

    def _init_conv(self, conv):
        if Args.args.init == "binary":
            # +1 or -1 for all weights
            conv.weight.data = conv.weight.data.sign() * torch.ones_like(conv.weight.data)

        elif Args.args.init == "lottery_binary":
            # +1 or -1 for all weights
            conv.weight.data = nn.init.uniform_(conv.weight, -1., 1.).sign() * torch.ones_like(conv.weight.data)

        elif Args.args.init == "uniform_pm1":
            # uniform in [-1, 1] for all weights:
            conv.weight.data = nn.init.uniform_(conv.weight, -1., 1.)

        elif Args.args.init == "signed_constant_correct":
            fan = nn.init._calculate_correct_fan(conv.weight, Args.args.mode)
            if Args.args.scale_fan:
                fan = fan * Args.args.prune_rate
            gain = nn.init.calculate_gain(Args.args.nonlinearity)
            std = gain / math.sqrt(fan)
            conv.weight.data = conv.weight.data.sign() * std

        elif Args.args.init == "unsigned_constant_correct":
            fan = nn.init._calculate_correct_fan(conv.weight, Args.args.mode)
            if Args.args.scale_fan:
                fan = fan * Args.args.prune_rate
            gain = nn.init.calculate_gain(Args.args.nonlinearity)
            std = gain / math.sqrt(fan)
            conv.weight.data = torch.ones_like(conv.weight.data) * std

        elif Args.args.init == "kaiming_normal_correct":

            if Args.args.scale_fan:
                fan = nn.init._calculate_correct_fan(conv.weight, Args.args.mode)
                fan = fan * Args.args.prune_rate
                gain = nn.init.calculate_gain(Args.args.nonlinearity)
                std = gain / math.sqrt(fan)
                with torch.no_grad():
                    conv.weight.data.normal_(0, std)
            else:
                nn.init.kaiming_normal_(
                    conv.weight, mode=Args.args.mode, nonlinearity=Args.args.nonlinearity
                )
                
        elif Args.args.init == "signed_constant":

            fan = nn.init._calculate_correct_fan(conv.weight, Args.args.mode)
            if Args.args.scale_fan:
                fan = fan * (1 - Args.args.prune_rate)
            gain = nn.init.calculate_gain(Args.args.nonlinearity)
            std = gain / math.sqrt(fan)
            conv.weight.data = conv.weight.data.sign() * std

        elif Args.args.init == "unsigned_constant":

            fan = nn.init._calculate_correct_fan(conv.weight, Args.args.mode)
            if Args.args.scale_fan:
                fan = fan * (1 - Args.args.prune_rate)

            gain = nn.init.calculate_gain(Args.args.nonlinearity)
            std = gain / math.sqrt(fan)
            conv.weight.data = torch.ones_like(conv.weight.data) * std

        elif Args.args.init == "kaiming_normal":

            if Args.args.scale_fan:
                fan = nn.init._calculate_correct_fan(conv.weight, Args.args.mode)
                fan = fan * (1 - Args.args.prune_rate)
                gain = nn.init.calculate_gain(Args.args.nonlinearity)
                std = gain / math.sqrt(fan)
                with torch.no_grad():
                    conv.weight.data.normal_(0, std)
            else:
                nn.init.kaiming_normal_(
                    conv.weight, mode=Args.args.mode, nonlinearity=Args.args.nonlinearity
                )

        elif Args.args.init == "kaiming_uniform":
            nn.init.kaiming_uniform_(
                conv.weight, mode=Args.args.mode, nonlinearity=Args.args.nonlinearity
            )
        elif Args.args.init == "xavier_normal":
            nn.init.xavier_normal_(conv.weight)
        elif Args.args.init == "xavier_constant":

            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(conv.weight)
            std = math.sqrt(2.0 / float(fan_in + fan_out))
            conv.weight.data = conv.weight.data.sign() * std

        elif Args.args.init == "standard":

            nn.init.kaiming_uniform_(conv.weight, a=math.sqrt(5))

        else:
            raise ValueError(f"{Args.args.init} is not an initialization option!")


def get_builder():

    print("==> Conv Type: {}".format(Args.args.conv_type))
    print("==> BN Type: {}".format(Args.args.bn_type))
    print("==> Init Type: {}".format(Args.args.init))

    conv_layer = getattr(arch.conv_type, Args.args.conv_type)
    bn_layer = getattr(arch.bn_type, Args.args.bn_type)

    if Args.args.first_layer_type is not None:
        first_layer = getattr(arch.conv_type, Args.args.first_layer_type)
        print(f"==> First Layer Type: {Args.args.first_layer_type}")
    else:
        first_layer = None

    if Args.args.last_layer_type is not None:
        last_layer = getattr(arch.conv_type, Args.args.last_layer_type)
        print(f"==> Last Layer Type: {Args.args.last_layer_type}")
    else:
        last_layer = None

    builder = Builder(conv_layer=conv_layer, bn_layer=bn_layer, first_layer=first_layer, last_layer=last_layer)

    return builder
