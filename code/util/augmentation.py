import sys
sys.path.insert(0, './')

import numpy as np

import torch
import torch.nn.functional as F


def apply_aug(data_batch, label_batch, class_num, aug_policy=['crop', 'vflip'], aug_trans_log = None):

    data_batch = data_batch.clone()
    batch_size = data_batch.size(0)
    height, width = data_batch.size(2), data_batch.size(3)
    # one-hot label. If aug_policy does not include cutmix or acutmix, the return value should still be one-hot
    soft_label = torch.zeros([batch_size, class_num]).to(data_batch.device)
    soft_label.scatter_(1, label_batch.view(-1, 1), 1)

    is_random = False
    if aug_trans_log is None:
        aug_trans_log = {}
        is_random = True

    for policy in aug_policy:

        if policy.lower() in ['crop',]:
            if is_random == True:
                h_indices, w_indices = np.random.randint(0, 8, (batch_size,)), np.random.randint(0, 8, (batch_size,))
                aug_trans_log['crop_h'] = h_indices
                aug_trans_log['crop_w'] = w_indices
            else:
                h_indices = aug_trans_log['crop_h']
                w_indices = aug_trans_log['crop_w']                
            data_batch = F.pad(data_batch, (4, 4, 4, 4), 'constant', 0.5)
            slices = [data_batch[idx_in_batch, :, h_index: h_index + height, w_index: w_index + width].unsqueeze(0) \
                for idx_in_batch, (h_index, w_index) in enumerate(zip(h_indices, w_indices))]
            data_batch = torch.cat(slices, dim = 0)
        elif policy.lower() in ['vflip',]:
            if is_random == True:
                v_vflip = np.random.randint(0, 2, (batch_size,))
                aug_trans_log['vflip'] = v_vflip
            else:
                v_vflip = aug_trans_log['vflip']
            slices = [data_batch[idx_in_batch].flip(2).unsqueeze(0) if v == 1 else data_batch[idx_in_batch].unsqueeze(0) \
                for idx_in_batch, v in enumerate(v_vflip)]
            data_batch = torch.cat(slices, dim = 0)
        elif policy.lower() in ['cutmix',]:
            if is_random == True:
                mix_perm = np.random.permutation(batch_size)
                mix_prop = np.random.uniform(0, 1, [batch_size,])
                h_lens = [int(np.sqrt(prop) * height) for prop in mix_prop]
                w_lens = [int(np.sqrt(prop) * width) for prop in mix_prop]
                h_indices = [np.random.randint(height - h_len) for h_len in h_lens]
                w_indices = [np.random.randint(width - w_len) for w_len in w_lens]
                aug_trans_log['cutmix_perm'] = mix_perm
                aug_trans_log['cutmix_prop'] = mix_prop
                aug_trans_log['cutmix_h'] = h_indices
                aug_trans_log['cutmix_w'] = w_indices
                aug_trans_log['cutmix_h_len'] = h_lens
                aug_trans_log['cutmix_w_len'] = w_lens
            else:
                mix_perm = aug_trans_log['cutmix_perm']
                mix_prop = aug_trans_log['cutmix_prop']
                h_indices = aug_trans_log['cutmix_h']
                w_indices = aug_trans_log['cutmix_w']
                h_lens = aug_trans_log['cutmix_h_len']
                w_lens = aug_trans_log['cutmix_w_len']
            data_batch_2 = data_batch.clone()
            soft_label_2 = soft_label.clone()
            data_slices = [data_batch_2[idx_in_batch].unsqueeze(0) for idx_in_batch in mix_perm]
            label_slices = [soft_label_2[idx_in_batch].unsqueeze(0) for idx_in_batch in mix_perm]
            data_batch_2 = torch.cat(data_slices, dim = 0)
            soft_label_2 = torch.cat(label_slices, dim = 0)
            for idx_in_batch, (prop, h_index, w_index, h_len, w_len) in enumerate(zip(mix_prop, h_indices, w_indices, h_lens, w_lens)):
                data_batch[idx_in_batch, :, h_index: h_index + h_len, w_index: w_index + w_len].data.copy_(
                    data_batch_2[idx_in_batch, :, h_index: h_index + h_len, w_index: w_index + w_len])
                soft_label[idx_in_batch].data.copy_(soft_label[idx_in_batch] * (1. - prop) + soft_label_2[idx_in_batch] * prop)
        elif policy.lower() in ['acutmix',]:
            if is_random == True:
                mix_perm = np.random.permutation(batch_size)
                mix_prop = np.random.uniform(0, 1)
                h_len = int(np.sqrt(mix_prop) * height)
                w_len = int(np.sqrt(mix_prop) * width)
                h_index = np.random.randint(height - h_len)
                w_index = np.random.randint(width - w_len)
                aug_trans_log['acutmix_perm'] = mix_perm
                aug_trans_log['acutmix_prop'] = mix_prop
                aug_trans_log['acutmix_h'] = h_index
                aug_trans_log['acutmix_w'] = w_index
                aug_trans_log['acutmix_h_len'] = h_len
                aug_trans_log['acutmix_w_len'] = w_len
            else:
                mix_perm = aug_trans_log['acutmix_perm']
                mix_prop = aug_trans_log['acutmix_prop']
                h_index = aug_trans_log['acutmix_h']
                w_index = aug_trans_log['acutmix_w']
                h_len = aug_trans_log['acutmix_h_len']
                w_len = aug_trans_log['acutmix_w_len']
            data_batch_2 = data_batch.clone()
            soft_label_2 = soft_label.clone()
            data_slices = [data_batch_2[idx_in_batch].unsqueeze(0) for idx_in_batch in mix_perm]
            label_slices = [soft_label_2[idx_in_batch].unsqueeze(0) for idx_in_batch in mix_perm]
            data_batch_2 = torch.cat(data_slices, dim = 0)
            soft_label_2 = torch.cat(label_slices, dim = 0)
            data_batch[:, :, h_index: h_index + h_len, w_index: w_index + w_len].data.copy_(data_batch_2[:, :, h_index: h_index + h_len, w_index: w_index + w_len])
            soft_label.data.copy_(soft_label * (1. - mix_prop) + soft_label_2 * mix_prop)
        else:
            raise ValueError('Unrecognized augmentation policy: %s' % policy)

    return data_batch, soft_label, aug_trans_log

def load_delta(data_batch, soft_label, delta_batch, aug_policy, aug_trans_log):

    height, width = data_batch.size(2), data_batch.size(3)
    perturb_batch = delta_batch.clone()

    for policy in aug_policy:

        if policy.lower() in ['crop',]:
            h_indices = aug_trans_log['crop_h']
            w_indices = aug_trans_log['crop_w']
            slices = [perturb_batch[idx_in_batch, :, h_index: h_index + height, w_index: w_index + width].unsqueeze(0) \
                for idx_in_batch, (h_index, w_index) in enumerate(zip(h_indices, w_indices))]
            perturb_batch = torch.cat(slices, dim = 0)
        elif policy.lower() in ['vflip']:
            v_vflip = aug_trans_log['vflip']
            slices = [perturb_batch[idx_in_batch].flip(2).unsqueeze(0) if v == 1 else perturb_batch[idx_in_batch].unsqueeze(0) \
                for idx_in_batch, v in enumerate(v_vflip)]
            perturb_batch = torch.cat(slices, dim = 0)
        elif policy.lower() in ['cutmix',]:
            mix_perm = aug_trans_log['cutmix_perm']
            mix_prop = aug_trans_log['cutmix_prop']
            h_indices = aug_trans_log['cutmix_h']
            w_indices = aug_trans_log['cutmix_w']
            h_lens = aug_trans_log['cutmix_h_len']
            w_lens = aug_trans_log['cutmix_w_len']
            perturb_batch_2 = perturb_batch.clone()
            slices = [perturb_batch_2[idx_in_batch].unsqueeze(0) for idx_in_batch in mix_perm]
            perturb_batch_2 = torch.cat(slices, dim = 0)
            for idx_in_batch, (h_index, w_index, h_len, w_len) in enumerate(zip(h_indices, w_indices, h_lens, w_lens)):
                perturb_batch[idx_in_batch, :, h_index: h_index + h_len, w_index: w_index + w_len].data.copy_(
                    perturb_batch_2[idx_in_batch, :, h_index: h_index + h_len, w_index: w_index + w_len])
        elif policy.lower() in ['acutmix',]:
            mix_perm = aug_trans_log['acutmix_perm']
            mix_prop = aug_trans_log['acutmix_prop']
            h_index = aug_trans_log['acutmix_h']
            w_index = aug_trans_log['acutmix_w']
            h_len = aug_trans_log['acutmix_h_len']
            w_len = aug_trans_log['acutmix_w_len']
            perturb_batch_2 = perturb_batch.clone()
            slices = [perturb_batch_2[idx_in_batch].unsqueeze(0) for idx_in_batch in mix_perm]
            perturb_batch_2 = torch.cat(slices, dim = 0)
            perturb_batch[:, :, h_index: h_index + h_len, w_index: w_index + w_len].data.copy_(perturb_batch_2[:, :, h_index: h_index + h_len, w_index: w_index + w_len])
        else:
            raise ValueError('Unrecognized augmentation policy: %s' % policy)

    return perturb_batch

def update_delta(data_batch, soft_label, delta_batch, perturb_batch, aug_policy, aug_trans_log):

    height, width = data_batch.size(2), data_batch.size(3)
    mask = torch.ones_like(perturb_batch)

    w_pad, h_pad = None, None

    for policy in aug_policy[::-1]:

        if policy.lower() in ['crop',]:
            h_indices = aug_trans_log['crop_h']
            w_indices = aug_trans_log['crop_w']
            delta_slices = [F.pad(perturb_batch[idx_in_batch], (w_index, 8 - w_index, h_index, 8 - h_index), 'constant', 0).unsqueeze(0) \
                for idx_in_batch, (w_index, h_index) in enumerate(zip(w_indices, h_indices))]
            mask_slices = [F.pad(mask[idx_in_batch], (w_index, 8 - w_index, h_index, 8 - h_index), 'constant', 0).unsqueeze(0) \
                for idx_in_batch, (w_index, h_index) in enumerate(zip(w_indices, h_indices))]
            perturb_batch = torch.cat(delta_slices, dim = 0)
            mask = torch.cat(mask_slices, dim = 0)
            h_pad = h_indices
            w_pad = w_indices
        elif policy.lower() in ['vflip']:
            v_vflip = aug_trans_log['vflip']
            delta_slices = [perturb_batch[idx_in_batch].flip(2).unsqueeze(0) if v == 1 else perturb_batch[idx_in_batch].unsqueeze(0) \
                for idx_in_batch, v in enumerate(v_vflip)]
            mask_slices = [mask[idx_in_batch].flip(2).unsqueeze(0) if v == 1 else mask[idx_in_batch].unsqueeze(0) \
                for idx_in_batch, v in enumerate(v_vflip)]
            perturb_batch = torch.cat(delta_slices, dim = 0)
            mask = torch.cat(mask_slices, dim = 0)
            if w_pad is not None:
                w_pad = [8 - pad if v == 1 else pad for idx_in_batch, (pad, v) in enumerate(zip(w_pad, v_vflip))]
        elif policy.lower() in ['cutmix',]:
            mix_perm = aug_trans_log['cutmix_perm']
            mix_prop = aug_trans_log['cutmix_prop']
            h_indices = aug_trans_log['cutmix_h']
            w_indices = aug_trans_log['cutmix_w']
            h_lens = aug_trans_log['cutmix_h_len']
            w_lens = aug_trans_log['cutmix_w_len']
            accumulate_perturb_batch = perturb_batch.clone()
            accumulate_mask = mask.clone()
            for idx_in_batch, (idx2, h_index, w_index, h_len, w_len) in enumerate(zip(mix_perm, h_indices, w_indices, h_lens, w_lens)):
                h_pad1 = h_index if h_pad is None else h_index + h_pad[idx_in_batch]
                h_pad2 = h_index if h_pad is None else h_index + h_pad[idx2]
                w_pad1 = w_index if w_pad is None else w_index + w_pad[idx_in_batch]
                w_pad2 = w_index if w_pad is None else w_index + w_pad[idx2]
                accumulate_perturb_batch[idx_in_batch, :, h_pad1: h_pad1 + h_len, w_pad1: w_pad1 + w_len].sub_(perturb_batch[idx_in_batch, :, h_pad1: h_pad1 + h_len, w_pad1: w_pad1 + w_len])
                accumulate_perturb_batch[idx2, :, h_pad2: h_pad2 + h_len, w_pad2: w_pad2 + w_len].add_(perturb_batch[idx_in_batch, :, h_pad1: h_pad1 + h_len, w_pad1: w_pad1 + w_len])
                accumulate_mask[idx_in_batch, :, h_pad1: h_pad1 + h_len, w_pad1: w_pad1 + w_len].sub_(mask[idx_in_batch, :, h_pad1: h_pad1 + h_len, w_pad1: w_pad1 + w_len])
                accumulate_mask[idx2, :, h_pad2: h_pad2 + h_len, w_pad2: w_pad2 + w_len].add_(mask[idx_in_batch, :, h_pad1: h_pad1 + h_len, w_pad1: w_pad1 + w_len])
            perturb_batch = accumulate_perturb_batch / torch.clamp(accumulate_mask, min = 1e-6)
            mask = torch.clamp(accumulate_mask, max = 1.)
        elif policy.lower() in ['acutmix',]:
            mix_perm = aug_trans_log['acutmix_perm']
            mix_prop = aug_trans_log['acutmix_prop']
            h_index = aug_trans_log['acutmix_h']
            w_index = aug_trans_log['acutmix_w']
            h_len = aug_trans_log['acutmix_h_len']
            w_len = aug_trans_log['acutmix_w_len']
            accumulate_perturb_batch = perturb_batch.clone()
            accumulate_mask = mask.clone()
            for idx_in_batch, idx2 in enumerate(mix_perm):
                h_pad1 = h_index if h_pad is None else h_index + h_pad[idx_in_batch]
                h_pad2 = h_index if h_pad is None else h_index + h_pad[idx2]
                w_pad1 = w_index if w_pad is None else w_index + w_pad[idx_in_batch]
                w_pad2 = w_index if w_pad is None else w_index + w_pad[idx2]
                accumulate_perturb_batch[idx_in_batch, :, h_pad1: h_pad1 + h_len, w_pad1: w_pad1 + w_len].sub_(perturb_batch[idx_in_batch, :, h_pad1: h_pad1 + h_len, w_pad1: w_pad1 + w_len])
                accumulate_perturb_batch[idx2, :, h_pad2: h_pad2 + h_len, w_pad2: w_pad2 + w_len].add_(perturb_batch[idx_in_batch, :, h_pad1: h_pad1 + h_len, w_pad1: w_pad1 + w_len])
                accumulate_mask[idx_in_batch, :, h_pad1: h_pad1 + h_len, w_pad1: w_pad1 + w_len].sub_(mask[idx_in_batch, :, h_pad1: h_pad1 + h_len, w_pad1: w_pad1 + w_len])
                accumulate_mask[idx2, :, h_pad2: h_pad2 + h_len, w_pad2: w_pad2 + w_len].add_(mask[idx_in_batch, :, h_pad1: h_pad1 + h_len, w_pad1: w_pad1 + w_len])
            perturb_batch = accumulate_perturb_batch / torch.clamp(accumulate_mask, min = 1e-6)
            mask = torch.clamp(accumulate_mask, max = 1.)
        else:
            raise ValueError('Unrecognized augmentation policy: %s' % policy)

    delta_batch = delta_batch * (1. - mask) + perturb_batch * mask
    delta_batch = delta_batch.detach().requires_grad_()

    return delta_batch