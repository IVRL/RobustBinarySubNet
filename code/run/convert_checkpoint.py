import os
import sys
sys.path.insert(0, './')

import json
import random
import pathlib
import numpy as np

import torch
import torch.nn as nn
from datetime import datetime
from copy import deepcopy

from autoattack.autoattack import AutoAttack
from util.data_parser import parse_data
from util.model_parser import parse_model
from util.device_parser import set_gpu

from args import args
from arch.net_utils import LabelSmoothing
from arch.conv_type import GetSubnet
from arch.conv_type import FixedSubnetConv

# required argument fields: pretrained, config, multigpu, model_name, prune_rate

def _run_dir_exists(run_base_dir):
    log_base_dir = run_base_dir / "logs"
    ckpt_base_dir = run_base_dir / "checkpoints"

    return log_base_dir.exists() or ckpt_base_dir.exists()


def get_directories(args):
    if args.config is None or args.model_name is None:
        raise ValueError("Must have name and config")

    config = pathlib.Path(args.config).stem
    if args.log_dir is None:
        run_base_dir = pathlib.Path(
            f"runs/convert-{config}/{args.model_name}/prune_rate={args.prune_rate}"
        )
    else:
        run_base_dir = pathlib.Path(
            f"{args.log_dir}/convert-{config}/{args.model_name}/prune_rate={args.prune_rate}"
        )

    rep_count = 0
    while _run_dir_exists(run_base_dir / str(rep_count)):
        rep_count += 1

    run_base_dir = run_base_dir / str(rep_count)

    log_base_dir = run_base_dir / "logs"
    ckpt_base_dir = run_base_dir / "checkpoints"

    if not run_base_dir.exists():
        os.makedirs(run_base_dir)

    (run_base_dir / "settings.txt").write_text(
        str(args) + datetime.now().strftime('%Y/%m/%d, %H:%M:%S') + str(sys.argv))

    return run_base_dir, ckpt_base_dir, log_base_dir


def pretrained(args, model):
    if os.path.isfile(args.pretrained):
        print("=> loading pretrained weights from '{}'".format(args.pretrained))
        pretrained = torch.load(
            args.pretrained,
            map_location=torch.device("cuda:{}".format(args.multigpu[0])),
        )["state_dict"]

        load_ckpt(pretrained, model)

    else:
        raise ValueError("=> no pretrained weights found at '{}'".format(args.pretrained))

    for n, m in model.named_modules():
        if isinstance(m, FixedSubnetConv):
            m.set_subnet()

def load_ckpt(pretrained, model_tmp):
    model_state_dict = model_tmp.state_dict()
    for k, v in pretrained.items():
        if k not in model_state_dict or v.size() != model_state_dict[k].size():
            print("IGNORE:", k, "size:")
    pretrained = {
        k: v
        for k, v in pretrained.items()
        if (k in model_state_dict and v.size() == model_state_dict[k].size())
    }
    model_state_dict.update(pretrained)
    model_tmp.load_state_dict(model_state_dict)

def subnet_to_dense(model):
    """
        Convert a subnet state dict (with subnet layers) to dense i.e., which can be directly 
        loaded in network with dense layers.
    """

    state_dict = deepcopy(model.state_dict())
    print(list(state_dict.keys()))

    n_remain = 0
    n_total = 0

    # load dense variables
    for n, m in model.named_modules():
        if hasattr(m, "set_prune_rate"):
            print(n)
            assert f"{n}.weight" in state_dict
            state_dict[f"{n}.weight"] = GetSubnet.apply(m.clamped_scores, m.prune_rate) * m.weight
            n_remain += torch.count_nonzero(state_dict[f"{n}.weight"])
            n_total += state_dict[f"{n}.weight"].numel()
            del state_dict[f"{n}.scores"]

    print(f"Actual prune rate: {n_remain / n_total}")
    print(list(state_dict.keys()))
        
    return state_dict

def check_model(model, model_dense):
    def compare(n, tensor_a, tensor_b):
        print(f"checking {n}...{np.allclose(tensor_a.detach().cpu().numpy(), tensor_b.detach().cpu().numpy())}")

    n_remain = 0
    n_total = 0
    for n, m in model.named_modules():
        if hasattr(m, "set_prune_rate"):
            # import pdb
            # pdb.set_trace()
            compare(n, GetSubnet.apply(m.clamped_scores, m.prune_rate) * m.weight, model_dense.state_dict()[f"{n}.weight"])
            n_remain += torch.count_nonzero(model_dense.state_dict()[f"{n}.weight"])
            n_total += model_dense.state_dict()[f"{n}.weight"].numel()
    print(f"Actual prune rate for dense model: {n_remain / n_total}")


if __name__ == "__main__":
    # configure random seed
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # configure io 
    run_base_dir, ckpt_base_dir, log_base_dir = get_directories(args)
    args.ckpt_base_dir = ckpt_base_dir

    # configure dataset and model, should work for cifar10 dataset
    train_loader, valid_loader, test_loader, classes = parse_data(name = args.dataset.lower(), batch_size = args.batch_size, valid_ratio = args.valid_ratio)
    model = parse_model(dataset = args.dataset, model_type = args.model_type, normalize = args.normalize, args=args)

    loader = {'train': train_loader, 'test': test_loader}[args.subset]

    # configure gpu
    model, use_gpu, device = set_gpu(args, model)

    # load pretrained model if possible
    if not args.pretrained:
        raise ValueError("no pretrained model to be loaded.")
    pretrained(args, model)

    model_dense_ckpt = subnet_to_dense(model)

    args.conv_type = "DenseConv"
    # args.bn_type = "LearnedBatchNorm"
    model_dense = parse_model(dataset = args.dataset, model_type = args.model_type, normalize = args.normalize, args=args)
    model_dense, use_gpu, device = set_gpu(args, model_dense)
    print("model dense:")
    print(list(model_dense.state_dict().keys()))
    load_ckpt(model_dense_ckpt, model_dense)

    check_model(model, model_dense)

    # configure criterion
    if args.label_smoothing is None:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = LabelSmoothing(smoothing=args.label_smoothing)
    criterion = criterion.cuda() if use_gpu else criterion

    # configure attacker and adversary budget
    threshold = args.attack["threshold"]
    threshold = threshold if threshold < 1.0 else threshold / 255.0

    print(args)

    configs = {kwargs: str(value) for kwargs, value in args._get_kwargs()}

    json.dump(configs, open(run_base_dir / "configs.json", "w"))

    l = [x for (x, y, idx) in loader]
    images = torch.cat(l, 0)
    l = [y for (x, y, idx) in loader]
    labels = torch.cat(l, 0)

    # attacker = AutoAttack(model, norm=args.attack.get("norm", "Linf"), device=device, eps=threshold, log_path=run_base_dir / "sparse-log.txt")
    # model.eval()
    # x_adv = attacker.run_standard_evaluation(images, labels, bs=args.batch_size)
    attacker_dense = AutoAttack(model_dense, norm=args.attack.get("norm", "Linf"), device=device, eps=threshold, log_path=run_base_dir / "dense-log.txt")
    model_dense.eval()
    x_adv = attacker_dense.run_standard_evaluation(images, labels, bs=args.batch_size)

    # torch.save({'adv_complete': x_adv}, '{}/{}_{}_1_{}_eps_{:.5f}.pth'.format(
    #             run_base_dir, 'aa', "standard", x_adv.shape[0], args.attack.get("threshold",8)))