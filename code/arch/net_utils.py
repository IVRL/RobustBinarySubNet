from functools import partial
import os
import pathlib
import shutil
import math
import numpy as np

import torch
import torch.nn as nn


def save_checkpoint(state, is_best, filename="checkpoint.pth", save=False):
    filename = pathlib.Path(filename)

    if not filename.parent.exists():
        os.makedirs(filename.parent)

    torch.save(state, filename)

    if is_best:
        shutil.copyfile(filename, str(filename.parent / "model_best.pth"))

        if not save:
            os.remove(filename)


def get_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def freeze_model_weights(model):
    print("=> Freezing model weights")

    for n, m in model.named_modules():
        if hasattr(m, "weight") and m.weight is not None:
            print(f"==> No gradient to {n}.weight")
            m.weight.requires_grad = False
            if m.weight.grad is not None:
                print(f"==> Setting gradient of {n}.weight to None")
                m.weight.grad = None

            if hasattr(m, "bias") and m.bias is not None:
                print(f"==> No gradient to {n}.bias")
                m.bias.requires_grad = False

                if m.bias.grad is not None:
                    print(f"==> Setting gradient of {n}.bias to None")
                    m.bias.grad = None


def freeze_model_subnet(model):
    print("=> Freezing model subnet")

    for n, m in model.named_modules():
        if hasattr(m, "scores"):
            m.scores.requires_grad = False
            print(f"==> No gradient to {n}.scores")
            if m.scores.grad is not None:
                print(f"==> Setting gradient of {n}.scores to None")
                m.scores.grad = None


def unfreeze_model_weights(model):
    print("=> Unfreezing model weights")

    for n, m in model.named_modules():
        if hasattr(m, "weight") and m.weight is not None:
            print(f"==> Gradient to {n}.weight")
            m.weight.requires_grad = True
            if hasattr(m, "bias") and m.bias is not None:
                print(f"==> Gradient to {n}.bias")
                m.bias.requires_grad = True


def unfreeze_model_subnet(model):
    print("=> Unfreezing model subnet")

    for n, m in model.named_modules():
        if hasattr(m, "scores"):
            print(f"==> Gradient to {n}.scores")
            m.scores.requires_grad = True


def set_model_prune_rate(model, prune_rate, p=1):
    print(f"==> Setting prune rate of network to {prune_rate}, with parameter {p}.")

    # 2 functions:
    # x1' / x1^d = x2' / x2^d = ... = xn' / xn^d = c
    # sum(x1', x2', ..., xn') = prune_rate * sum(x1, x2, ..., xn)

    assert(0 <= p <= 1)

    # sort the layers with their #params
    layer_names = []
    layer_counts = []
    for n, m in model.named_modules():
        if hasattr(m, "set_prune_rate"):
            layer_names.append(n)
            layer_counts.append(m.scores.numel())
    if len(layer_counts) == 0:
        print("no prunable parameters.")
        return  
    sorted_index = sorted(range(len(layer_counts)), key=lambda k: layer_counts[k])
    sorted_names = [layer_names[idx] for idx in sorted_index]
    sorted_counts = [layer_counts[idx] for idx in sorted_index]

    # start from the smallest layer, check if number of kept parameters exceeds the maximum
    total_kept_params = int(prune_rate * np.sum(sorted_counts))
    c = prune_rate * np.sum(sorted_counts) / np.sum(np.power(sorted_counts, p))
    pr_dict = {}
    check_counts = 0
    for idx, name, counts in zip(range(len(sorted_counts)), sorted_names, sorted_counts):
        layer_prune_rate = c * np.power(counts, p-1.0)
        if layer_prune_rate > 1:
            print(f"layer {name}: prune rate > 1. Re-calculating prune rate for other layers")
            layer_prune_rate = 1.0
        total_kept_params -= int(layer_prune_rate * counts)
        check_counts += int(layer_prune_rate * counts)
        c = total_kept_params / np.sum(np.power(sorted_counts[idx+1:], p))
        pr_dict[name] = layer_prune_rate

    # apply the prune rate
    for n, m in model.named_modules():
        if hasattr(m, "set_prune_rate"):
            layer_prune_rate = pr_dict[n]
            m.set_prune_rate(layer_prune_rate)
            print(f"==> Setting prune rate of {n} to {layer_prune_rate}")
    print(f"==> check overall prune rate: {check_counts / np.sum(layer_counts)}; expected: {prune_rate}.")


def accumulate(model, f):
    acc = 0.0

    for child in model.children():
        acc += accumulate(child, f)

    acc += f(model)

    return acc


class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.

        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

