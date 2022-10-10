import os
import sys
sys.path.insert(0, './')

import time
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

from .utility import project
from .augmentation import update_delta, apply_aug, load_delta

h_message = '''
>>> PGD(step_size, threshold, iter_num, order = np.inf)
>>> TradesKL(step_size, threshold, iter_num, order=np.inf)
>>> FGSM-RS(step_size, threshold, order = np.inf, 
        log = 0, delta_reset=None, aug_policy=['crop', 'vflip'], n_class=10, warmup=0)
'''

def parse_attacker(name, mode = 'test', **kwargs):
    if name.lower() in ['h', 'help']:
        print(h_message)
        exit(0)
    elif name.lower() in ['pgd',]:
        return PGD(**kwargs)
    elif name.lower() in ['tradeskl',]:
        return TradesKL(**kwargs)
    elif name.lower() in ['fgsm-rs']:
        return FGSM_RS(**kwargs)
    else:
        raise ValueError('Unrecognized name of the attacker: %s' % name)

def zip_tensor(tensor, rate):

    is_single = False
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(dim = 0)
        is_single = True
    elif tensor.dim() == 4:
        pass
    else:
        raise ValueError('Wrong dimensionality %d' % tensor.dim())

    batch_size, channel_num, height, width = tensor.shape
    post_height = (height - 1) // rate + 1
    post_width = (width - 1) // rate + 1

    padded_height = (rate - (height % rate)) % rate
    padded_width = (rate - (width % rate)) % rate

    if padded_height != 0:
        tensor_pad_height = torch.zeros([batch_size, channel_num, padded_height, width], device = tensor.device)
        tensor = torch.cat([tensor, tensor_pad_height], dim = 2)
    if padded_width != 0:
        tensor_pad_width = torch.zeros([batch_size, channel_num, height + padded_height, padded_width], device = tensor.device)
        tensor = torch.cat([tensor, tensor_pad_width], dim = 3)

    tensor = tensor.view(batch_size, channel_num, post_height, rate, post_width, rate).mean(dim = (3, 5))
    height_scale = 1. - padded_height / rate
    width_scale = 1. - padded_width / rate

    tensor[:, :, -1, :].mul_(1. / height_scale)
    tensor[:, :, :, -1].mul_(1. / width_scale)

    if is_single == True:
        tensor = tensor.squeeze(dim = 0)
    tensor = tensor.detach()

    return tensor

def unzip_tensor(tensor, rate, post_height = None, post_width = None):

    is_single = False
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(dim = 0)
        is_single = True
    elif tensor.dim() == 4:
        pass
    else:
        raise ValueError('Wong dimensionality %d' % tensor.dim())

    batch_size, channel_num, height, width = tensor.shape
    post_tensor = tensor.unsqueeze(dim = 3).repeat(1, 1, 1, rate, 1).view(batch_size, channel_num, height * rate, width)
    post_tensor = post_tensor.unsqueeze(dim = 4).repeat(1, 1, 1, 1, rate).view(batch_size, channel_num, height * rate, width * rate)

    if post_height is not None:
        post_tensor = post_tensor[:, :, :post_height, :]
    if post_width is not None:
        post_tensor = post_tensor[:, :, :, :post_width]
    if is_single == True:
        post_tensor = post_tensor.squeeze(dim = 0)
    post_tensor = post_tensor.detach()

    return post_tensor

class PGD(object):

    def __init__(self, step_size, threshold, iter_num, order = np.inf, log = 0):

        self.step_size = step_size if step_size < 1. else step_size / 255.
        self.threshold = threshold if threshold < 1. else threshold / 255.
        self.step_size_threshold_ratio = self.step_size / (self.threshold + 1e-6)
        self.iter_num = int(iter_num)
        self.order = order if order > 0 else np.inf

        self.meta_threshold = self.threshold
        self.meta_step_size = self.step_size

        self.log = False if int(log) == 0 else True

        print('Create a PGD attacker')
        print('step_size = %1.2e, threshold = %1.2e, iter_num = %d, order = %f' % (
            self.step_size, self.threshold, self.iter_num, self.order))

    def get_name(self):
        return "PGD"

    def adjust_threshold(self, threshold, log = True):

        threshold = threshold if threshold < 1. else threshold / 255.

        self.step_size = self.meta_step_size * threshold / (self.meta_threshold + 1e-6)
        self.threshold = threshold

        if log == True:
            print('Attacker adjusted, threshold = %1.2e, step_size = %1.2e' % (self.threshold, self.step_size))

    def attack(self, model, data_batch, label_batch, criterion = None):

        data_batch = data_batch.detach()
        label_batch = label_batch.detach()
        device = data_batch.device

        batch_size = data_batch.size(0)
        step_size, threshold = self.step_size, self.threshold

        criterion = criterion.cuda(device)

        if np.max(threshold) < 1e-6:
            return data_batch, label_batch

        ori_batch = data_batch.detach()

        # Initial perturbation
        noise = project(ori_pt = (torch.rand_like(data_batch) * 2 - 1) * threshold, threshold = threshold, order = self.order)
        data_batch = torch.clamp(data_batch + noise, min = 0., max = 1.)
        data_batch = data_batch.detach().requires_grad_()

        worst_loss_per_instance = None
        correctness_bits = [1, ] * data_batch.size(0)
        adv_data_batch = deepcopy(data_batch.detach())

        for iter_idx in range(self.iter_num):

            logits = model(data_batch)
            loss = criterion(logits, label_batch)
            _, prediction = logits.max(dim = 1)

            # Gradient
            loss.backward()
            grad = data_batch.grad.data

            # Update worst loss
            loss_per_instance = - F.log_softmax(logits).gather(dim = 1, index = label_batch.view(-1, 1)).view(-1)
            if worst_loss_per_instance is None:
                worst_loss_per_instance = torch.zeros_like(loss_per_instance)

            loss_update_bits = (loss_per_instance > worst_loss_per_instance).float()
            mask_shape = [-1,] + [1,] * (data_batch.dim() - 1)
            loss_update_bits = loss_update_bits.view(*mask_shape)
            adv_data_batch = deepcopy((adv_data_batch * (1. - loss_update_bits) + data_batch * loss_update_bits).detach())
            worst_loss_per_instance = torch.max(worst_loss_per_instance, loss_per_instance)

            if self.order == np.inf:
                next_point = data_batch + torch.sign(grad) * step_size
            elif self.order == 2:
                ori_shape = data_batch.size()
                grad_norm = torch.norm(grad.view(ori_shape[0], -1), dim = 1, p = 2)
                perb = (grad.view(ori_shape[0], -1) + 1e-8) / (grad_norm.view(-1, 1) + 1e-8) * step_size
                next_point = data_batch + perb.view(ori_shape)
            else:
                raise ValueError('Invalid norm: %s' % str(self.order))

            next_point = ori_batch + project(ori_pt = next_point - ori_batch, threshold = threshold, order = self.order)
            next_point = torch.clamp(next_point, min = 0., max = 1.)

            data_batch = next_point.detach().requires_grad_()

            model.zero_grad()

        return adv_data_batch, label_batch


class TradesKL(object):

    def __init__(self, step_size, threshold, iter_num, order=np.inf, log=False):

        self.step_size = step_size if step_size < 1. else step_size / 255.
        self.threshold = threshold if threshold < 1. else threshold / 255.
        self.iter_num = int(iter_num)
        self.order = order if order > 0 else np.inf

        self.log = False if int(log) == 0 else True

        print('Create a TRADES attacker')
        print('step_size = %1.2e, threshold = %1.2e, iter_num = %d, order = %f' % (
            self.step_size, self.threshold, self.iter_num, self.order))

    def adjust_threshold(self, threshold, log = True):

        threshold = threshold if threshold < 1. else threshold / 255.
        self.threshold = threshold
        
        if log == True:
            print('Attacker adjusted, threshold = %1.2e' % (self.threshold,))


    def attack(self, model, data_batch, label_batch, criterion=nn.KLDivLoss(size_average=False)):

        model.eval()
        batch_size = len(data_batch)
        # generate adversarial example
        x_adv = data_batch.detach() + 0.001 * torch.randn(data_batch.shape).cuda().detach()
        if self.order == np.inf:
            for _ in range(self.iter_num):
                x_adv.requires_grad_()
                with torch.enable_grad():
                    loss_kl = criterion(F.log_softmax(model(x_adv), dim=1),
                                        F.softmax(model(data_batch), dim=1))
                grad = torch.autograd.grad(loss_kl, [x_adv])[0]
                x_adv = x_adv.detach() + self.step_size * torch.sign(grad.detach())
                x_adv = torch.min(torch.max(x_adv, data_batch - self.threshold), data_batch + self.threshold)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
        elif self.order == 2:
            delta = 0.001 * torch.randn(data_batch.shape).cuda().detach()
            delta = Variable(delta.data, requires_grad=True)

            # Setup optimizers
            optimizer_delta = optim.SGD([delta], lr=self.threshold / self.iter_num * 2)

            for _ in range(self.iter_num):
                adv = data_batch + delta

                # optimize
                optimizer_delta.zero_grad()
                with torch.enable_grad():
                    loss = (-1) * criterion(F.log_softmax(model(adv), dim=1),
                                            F.softmax(model(data_batch), dim=1))
                loss.backward()
                # renorming gradient
                grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
                delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
                # avoid nan or inf if gradient is 0
                if (grad_norms == 0).any():
                    delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
                optimizer_delta.step()

                # projection
                delta.data.add_(data_batch)
                delta.data.clamp_(0, 1).sub_(data_batch)
                delta.data.renorm_(p=2, dim=0, maxnorm=self.threshold)
            x_adv = Variable(data_batch + delta, requires_grad=False)
        else:
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        model.train()
        return x_adv, label_batch

class FGSM_RS(object):
    def __init__(self, step_size, threshold, order = np.inf, 
        log = 0, delta_reset=None, aug_policy=['crop', 'vflip'], n_class=10, warmup=0):
        self.step_size = step_size if step_size < 1. else step_size / 255.
        self.threshold = threshold if threshold < 1. else threshold / 255.
        self.iter_num = "Nan"
        self.order = order
        self.log = log
        # self.attacker = PGD(step_size, threshold, iter_num=1, order=order, log=log)
        self.delta_reset = delta_reset
        self.aug_policy = aug_policy
        self.idx2delta = {}
        self.idx2target = {}
        self.n_class = n_class
        self.warmup = warmup
        self.beta = 0.1
        self.rho = 0.9

        self.meta_threshold = self.threshold
        self.meta_step_size = self.step_size

        print('Create a FGSM-RS attacker')
        print('step_size = %1.2e, threshold = %1.2e' % (
            self.step_size, self.threshold))

    def get_name(self):
        return "FGSM-RS"

    def adjust_threshold(self, threshold, log = True):
        threshold = threshold if threshold < 1. else threshold / 255.

        self.step_size = self.meta_step_size * threshold / (self.meta_threshold + 1e-6)
        self.threshold = threshold

        if log == True:
            print('Attacker adjusted, threshold = %1.2e, step_size = %1.2e' % (self.threshold, self.step_size))

    def attack(self, model, epoch, data_batch, label_batch, idx_batch, criterion = None):

        channel_num, height, width = data_batch.size(1), data_batch.size(2), data_batch.size(3)
        compress_rate = 8 if height > 160 else 1

        # Obtain delta
        slices = []
        for idx_in_batch, instance_idx in enumerate(idx_batch):
            if instance_idx.__int__() not in self.idx2delta:
                # padding of 4 considered
                if 'crop' in self.aug_policy:
                    delta_this_instance = (torch.rand(channel_num, height + 8, height + 8).to(data_batch.device) * 2 - 1) * self.threshold
                else:
                    delta_this_instance = (torch.rand_like(data_batch[idx_in_batch]) * 2 - 1) * self.threshold
                slices.append(delta_this_instance.unsqueeze(0))
                self.idx2delta[instance_idx.__int__()] = zip_tensor(delta_this_instance, rate = compress_rate)
            else:
                if 'crop' in self.aug_policy:
                    slices.append(unzip_tensor(self.idx2delta[instance_idx.__int__()].unsqueeze(0), rate = compress_rate, post_height = height + 8, post_width = width + 8))
                else:
                    slices.append(unzip_tensor(self.idx2delta[instance_idx.__int__()].unsqueeze(0), rate = compress_rate, post_height = height, post_width = width))
        delta_batch = torch.cat(slices, dim = 0).detach().requires_grad_()

        # Apply augmentation
        ori_batch = data_batch.clone()
        aug_trans_log = None
        data_batch, soft_label, aug_trans_log = apply_aug(ori_batch, label_batch, self.n_class, self.aug_policy, aug_trans_log)
        
        perturb_batch = load_delta(data_batch, soft_label, delta_batch, self.aug_policy, aug_trans_log)
        perturb_batch = perturb_batch.detach().requires_grad_()

        logits = model(data_batch + perturb_batch)

        soft_prob = F.softmax(logits, dim = 1)
        for idx_in_batch, instance_idx in enumerate(idx_batch):
            if instance_idx.__int__() in self.idx2target:
                target_this_instance = self.idx2target[instance_idx.__int__()]
            else:
                target_this_instance = soft_label[idx_in_batch]
            self.idx2target[instance_idx.__int__()] = (target_this_instance * self.rho + soft_prob[idx_in_batch] * (1. - self.rho)).detach()
        target_this_batch = [self.idx2target[instance_idx.__int__()].unsqueeze(0) for instance_idx in idx_batch]
        target_this_batch = torch.cat(target_this_batch, dim = 0)
        target_this_batch = (soft_label * self.beta + target_this_batch * (1. - self.beta)).detach()

        ce_loss = - (F.log_softmax(logits, dim = -1) * soft_label).sum(dim = 1).mean()
        grad = torch.autograd.grad(ce_loss, perturb_batch)[0]
        perturb_batch = torch.clamp(perturb_batch + torch.sign(grad) * self.step_size, min = - self.threshold, max = self.threshold)
        perturb_batch = torch.clamp(data_batch + perturb_batch, min = 0., max = 1.) - data_batch

        # Update delta
        delta_batch = update_delta(data_batch, soft_label, delta_batch, perturb_batch, self.aug_policy, aug_trans_log)
        for idx_in_batch, instance_idx in enumerate(idx_batch):
            self.idx2delta[instance_idx.__int__()] = zip_tensor(delta_batch[idx_in_batch].detach(), rate = compress_rate)

        return data_batch + perturb_batch, soft_label
        # return self.attacker.attack(model, data_batch, label_batch, criterion)