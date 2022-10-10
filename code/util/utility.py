import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from collections import OrderedDict

## Project to an adversarial budget
def project_(ori_pt, threshold, order = np.inf):

    if order in [np.inf,]:
        prj_pt = torch.clamp(ori_pt, min = - threshold, max = threshold) 
    elif order in [2,]:
        ori_shape = ori_pt.size()
        pt_norm = torch.norm(ori_pt.view(ori_shape[0], -1), dim = 1, p = 2)
        pt_norm_clip = torch.clamp(pt_norm, max = threshold)
        prj_pt = ori_pt.view(ori_shape[0], -1) / (pt_norm.view(-1, 1) + 1e-8) * (pt_norm_clip.view(-1, 1) + 1e-8)
        prj_pt = prj_pt.view(ori_shape)
    else:
        raise ValueError('Invalid norms: %s' % order)

    return prj_pt


def project(ori_pt, threshold, order = np.inf):
    '''
    Project the data into a norm ball

    >>> ori_pt: the original point
    >>> threshold: maximum norms allowed, can be float or list of float
    >>> order: norm used
    '''
    if isinstance(threshold, float):
        prj_pt = project_(ori_pt, threshold, order)
    elif isinstance(threshold, (list, tuple)):
        if list(set(threshold)).__len__() == 1:
            prj_pt = project_(ori_pt, threshold[0], order)
        else:
            assert len(threshold) == ori_pt.size(0)
            prj_pt = torch.zeros_like(ori_pt)
            for idx, threshold_ in enumerate(threshold):
                prj_pt[idx: idx+1] = project_(ori_pt[idx: idx+1], threshold_, order)
    else:
        raise ValueError('Invalid value of threshold: %s' % threshold)

    return prj_pt
