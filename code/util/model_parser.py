import os
import sys
sys.path.insert(0, './')

import torch.nn as nn

from arch.preprocess import DataNormalizeLayer
import arch
from arch.net_utils import (
    set_model_prune_rate,
    freeze_model_weights
)

cifar10_normalize = {'bias': [0.4914, 0.4822, 0.4465], 'scale': [0.2023, 0.1994, 0.2010]}
cifar100_normalize = {'bias': [0.5071, 0.4867, 0.4408], 'scale': [0.2675, 0.2565, 0.2761]}
imagenet_normalize = {'bias': [0.485, 0.456, 0.406], 'scale': [0.229, 0.224, 0.225]}

def parse_model(dataset, model_type, normalize = None, options = None, args = None, **kwargs):

    if isinstance(normalize, str):
        if normalize.lower() in ['cifar10', 'cifar10_normalize',]:
            normalize_layer = DataNormalizeLayer(bias = cifar10_normalize['bias'], scale = cifar10_normalize['scale'])
        elif normalize.lower() in ['cifar100', 'cifar100_normalize']:
            normalize_layer = DataNormalizeLayer(bias = cifar100_normalize['bias'], scale = cifar100_normalize['scale'])
        elif normalize.lower() in ['imagenet', 'imagenet100', 'imagenet_normalize', 'imagenet100_normalize']:
            normalize_layer = DataNormalizeLayer(bias = imagenet_normalize['bias'], scale = imagenet_normalize['scale'])
        else:
            raise ValueError('Unrecognized normalizer: %s' % normalize)
    elif normalize is not None:
        normalize_layer = DataNormalizeLayer(bias = normalize['bias'], scale = normalize['scale'])
    else:
        normalize_layer = DataNormalizeLayer(bias = 0., scale = 1.)

    if dataset.lower() in ['cifar10', 'cifar100', 'imagenet100']:
        if model_type in arch.__all__:
            if args.first_layer_dense:
                args.first_layer_type = "DenseConv"
            if args.last_layer_dense:
                args.last_layer_type = "DenseConv"

            print("=> Creating model '{}'".format(model_type))
            net = arch.__dict__[model_type]()
            # applying sparsity to the network
            if args.conv_type not in ["DenseConv"]:
                if args.prune_rate < 0 or args.prune_rate > 1:
                    raise ValueError(f"Need to set a valid prune rate within [0,1]. Got {args.prune_rate} instead.")

                set_model_prune_rate(net, prune_rate=args.prune_rate, p=args.pr_scale)
                print(
                    f"=> Rough estimate model params {sum(int(par.numel() * args.prune_rate) for n, par in net.named_parameters() if not n.endswith('scores'))}"
                )
            # freezing the weights if we are only doing subnet training
            if args.freeze_weights:
                freeze_model_weights(net)
        else:
            raise ValueError('Unrecognized architecture: %s' % model_type)
    else:
        raise ValueError('Unrecognized dataset: %s' % dataset)

    return nn.Sequential(normalize_layer, net) if normalize_layer else net

