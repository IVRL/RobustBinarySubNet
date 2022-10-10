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

from autoattack.autoattack import AutoAttack
from util.data_parser import parse_data
from util.model_parser import parse_model
from util.device_parser import set_gpu

from args import args
from arch.net_utils import LabelSmoothing
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
            f"runs/{config}/{args.model_name}/prune_rate={args.prune_rate}"
        )
    else:
        run_base_dir = pathlib.Path(
            f"{args.log_dir}/{config}/{args.model_name}/prune_rate={args.prune_rate}"
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

        model_state_dict = model.state_dict()
        for k, v in pretrained.items():
            if k not in model_state_dict or v.size() != model_state_dict[k].size():
                print("IGNORE:", k)
        pretrained = {
            k: v
            for k, v in pretrained.items()
            if (k in model_state_dict and v.size() == model_state_dict[k].size())
        }
        model_state_dict.update(pretrained)
        model.load_state_dict(model_state_dict)

    else:
        raise FileNotFoundError("=> no pretrained weights found at '{}'".format(args.pretrained))

    for n, m in model.named_modules():
        if isinstance(m, FixedSubnetConv):
            m.set_subnet()


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
    
    # configure criterion
    if args.label_smoothing is None:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = LabelSmoothing(smoothing=args.label_smoothing)
    criterion = criterion.cuda() if use_gpu else criterion

    # configure attacker and adversary budget
    threshold = args.attack["threshold"]
    threshold = threshold if threshold < 1.0 else threshold / 255.0
    attacker = AutoAttack(model, norm=args.attack.get("norm", "Linf"), device=device, eps=threshold, log_path=run_base_dir / "log.txt")

    print(args)

    configs = {kwargs: str(value) for kwargs, value in args._get_kwargs()}

    json.dump(configs, open(run_base_dir / "configs.json", "w"))

    l = [x for (x, y, idx) in loader]
    images = torch.cat(l, 0)
    l = [y for (x, y, idx) in loader]
    labels = torch.cat(l, 0)

    model.eval()
    x_adv = attacker.run_standard_evaluation(images, labels, bs=args.batch_size)

    torch.save({'adv_complete': x_adv}, '{}/{}_{}_1_{}_eps_{:.5f}.pth'.format(
                run_base_dir, 'aa', "standard", x_adv.shape[0], args.attack.get("threshold",8)))