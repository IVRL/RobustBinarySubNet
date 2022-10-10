import os
import sys
import json
import time
import random
import pathlib
import pickle
import importlib
import traceback
import numpy as np

sys.path.insert(0, './')
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter

from util.attack import FGSM_RS, parse_attacker, PGD, TradesKL
from util.seq_parser import continuous_seq
from util.model_parser import parse_model
from util.optim_parser import get_optimizer
from util.device_parser import set_gpu
from util.data_parser import parse_data
from util.mask_saver import get_mask_info, save_mask_info
from util.bn_saver import get_bn_info
from visualize.lr_curve import plot_lr_curve_single

from args import args
from arch.net_utils import (
    save_checkpoint, 
    LabelSmoothing,
    get_lr,
)
from util.schedulers import get_policy
from util.logging import AverageMeter, ProgressMeter
from util.pr_scheduler import set_pr_scheduler_params
from util.change_monitor import monitor_mask_changes, monitor_bn_changes
from arch.conv_type import FixedSubnetConv



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


def is_exploded(valid_acc, prev_acc_list):
    prev_sub = prev_acc_list[-1] - prev_acc_list[-2]
    curr_sub = valid_acc - prev_acc_list[-1]
    if prev_sub >= 0 or curr_sub <= 0:
        return False

    return min(abs(prev_sub), abs(curr_sub)) > args.explode_th


def resume(args, model, optimizer):
    if os.path.isfile(args.resume):
        print(f"=> Resuming from checkpoint '{args.resume}'")

        checkpoint = torch.load(args.resume, map_location=f"cuda:{args.multigpu[0]}")
        if args.start_epoch is None:
            print(f"=> Setting new start epoch at {checkpoint['epoch']}")
            args.start_epoch = checkpoint["epoch"]

        # TODO: learning rate, attack eps, 

        best_train_acc1 = checkpoint["best_train_acc1"]
        best_train_acc5 = checkpoint["best_train_acc5"]
        best_valid_acc1 = checkpoint["best_valid_acc1"]
        best_valid_acc5 = checkpoint["best_valid_acc5"]
        best_test_acc1 = checkpoint["best_test_acc1"]
        best_test_acc5 = checkpoint["best_test_acc5"]

        model.load_state_dict(checkpoint["state_dict"])

        optimizer.load_state_dict(checkpoint["optimizer"])

        tosave_file = os.path.join(os.path.dirname(os.path.dirname(args.resume)), "logs", "logs.json")
        tosave = json.load(open(tosave_file, "r"))

        print(f"=> Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")

        return best_train_acc1, best_train_acc5, best_valid_acc1, best_valid_acc5, best_test_acc1, best_test_acc5, tosave
    else:
        print(f"=> No checkpoint found at '{args.resume}'")


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


def write_result_to_csv(**kwargs):
    results = pathlib.Path("runs") / "results-new.csv"

    if not results.exists():
        results.write_text(
            "Date Finished, "
            "Base Config, "
            "Name, "
            "Prune Rate, "
            "Current Test Top 1, "
            "Current Test Top 5, "
            "Best Test Top 1, "
            "Best Test Top 5, "
            "Current Val Top 1, "
            "Current Val Top 5, "
            "Best Val Top 1, "
            "Best Val Top 5, "
            "Best Train Top 1, "
            "Best Train Top 5\n"
        )

    now = time.strftime("%m-%d-%y_%H:%M:%S")

    with open(results, "a+") as f:
        f.write(
            (
                "{now}, "
                "{base_config}, "
                "{name}, "
                "{prune_rate}, "
                "{curr_test_acc1:.02f}, "
                "{curr_test_acc5:.02f}, "
                "{best_test_acc1:.02f}, "
                "{best_test_acc5:.02f}, "
                "{curr_valid_acc1:.02f}, "
                "{curr_valid_acc5:.02f}, "
                "{best_valid_acc1:.02f}, "
                "{best_valid_acc5:.02f}, "
                "{best_train_acc1:.02f}, "
                "{best_train_acc5:.02f}\n"
            ).format(now=now, **kwargs)
        )


if __name__ == "__main__":
    # configure random seed
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            # some other things
            # torch.backends.cudnn.enabled = False 
            # torch.backends.cudnn.benchmark = False
            # torch.backends.cudnn.deterministic = True
            # def _init_fn():
            #     np.random.seed(manualSeed)
                

            # DataLoding = data.DataLoader(..., batch_size = ..., 
            #                             collate_fn = ..., 
            #                             num_workers = 0, 
            #                             shuffle = ..., 
            #                             pin_memory = ...,
            #                             worker_init_fn=_init_fn)
            # torch.use_deterministic_algorithms(True)
            # refer to: https://pytorch.org/docs/stable/notes/randomness.html
            # and https://discuss.pytorch.org/t/random-seed-initialization/7854/30

    # do not augment data right inside the dataset when using fgsm-rs training
    if args.attack != None and args.attack["name"].lower() in ["fgsm-rs"]:
        assert (args.trainer == "adv_step_fast"), "Please use adv_step_fast as the trainer for fgsm-rs training"
        augmentation = False
    else:
        augmentation = True

    # configure dataset and model, should work for cifar10 dataset
    train_loader, valid_loader, test_loader, classes = parse_data(name = args.dataset.lower(), batch_size = args.batch_size, valid_ratio = args.valid_ratio, augmentation=augmentation)
    model = parse_model(dataset = args.dataset, model_type = args.model_type, normalize = args.normalize, args=args)

    # configure prune rate scheduler
    set_pr_scheduler_params(args)

    # configure gpu
    model, use_gpu, device = set_gpu(args, model)

    # load pretrained model if possible
    if args.pretrained:
        pretrained(args, model)

    # configure optimizer
    optimizer = get_optimizer(args, model)
    
    # configure criterion
    if args.label_smoothing is None:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = LabelSmoothing(smoothing=args.label_smoothing)
    criterion = criterion.cuda() if use_gpu else criterion

    # configure learning rate policy
    lr_policy = get_policy(args.lr_policy)(optimizer, args)

    # configure io 
    run_base_dir, ckpt_base_dir, log_base_dir = get_directories(args)
    args.ckpt_base_dir = ckpt_base_dir
    args.log_base_dir = log_base_dir

    # configure attacker and adversary budget
    args.train_attacker = None
    args.test_attacker = None
    if args.attack != None:
        args.train_attacker = parse_attacker(**args.attack)
        if args.eval_attack != None:
            args.test_attacker = parse_attacker(**args.eval_attack)
        else:
            args.test_attacker = parse_attacker(**args.attack)
    if args.trades_attack != None:
        args.trades_attacker = parse_attacker(**args.trades_attack)
    if args.attack["name"].lower() in ["simple-aa"]:
        args.train_attacker.init_setup(model, log_path=run_base_dir / "simple-aa-train-log.txt")
        if args.eval_attack is None or args.eval_attack["name"].lower() in ["simple-aa"]:
            args.test_attacker.init_setup(model, log_path=run_base_dir / "simple-aa-test-log.txt")
    eps_func = continuous_seq(**args.eps_schedule) if args.eps_schedule != None else None  

    print(args)


    # logging
    writer = SummaryWriter(log_dir=log_base_dir)
    epoch_time = AverageMeter("epoch_time", ":.4f", write_avg=False)
    validation_time = AverageMeter("validation_time", ":.4f", write_avg=False)
    test_time = AverageMeter("test_time", ":.4f", write_avg=False)
    train_time = AverageMeter("train_time", ":.4f", write_avg=False)
    progress_overall = ProgressMeter(
        1, [
            epoch_time, validation_time, test_time, train_time], prefix="Overall Timing and loss"
    )


    # configure saving 
    tosave = {
        'config': {},
        'train_loss': {}, 
        'train_acc': {}, 
        'valid_loss': {}, 
        'valid_acc': {}, 
        'test_loss': {}, 
        'test_acc': {}, 
        'lr': {}, 
    }

    for k, v in vars(args).items():
        if isinstance(v, pathlib.PosixPath):
            tosave['config'][k] = str(v)
        elif isinstance(v, PGD) or isinstance(v, TradesKL) or isinstance(v, FGSM_RS):
            tosave['config'][k] = {
                "name": type(v).__name__,
                "step size": v.step_size,
                "threshold": v.threshold,
                "iter number": v.iter_num,
                "order": str(v.order)
            }
        else:
            tosave['config'][k] = v

    # load trainer
    print(f"=> Using trainer from trainers.{args.trainer}")
    trainer = importlib.import_module(f"trainers.{args.trainer}")

    train = trainer.train
    validate = trainer.validate
    modifier = trainer.modifier

    # optionally resume from a checkpoint
    best_valid_acc1 = 0.0
    best_valid_acc5 = 0.0
    best_test_acc1 = 0.0
    best_test_acc5 = 0.0
    best_train_acc1 = 0.0
    best_train_acc5 = 0.0
    valid_acc1 = None
    test_acc1 = None

    # save the initial state
    checkpoint = {
        "epoch": 0,
        "arch": args.model_type,
        "state_dict": model.state_dict(),
        "best_valid_acc1": best_valid_acc1 if valid_loader else "Not evaluated",
        "best_valid_acc5": best_valid_acc5 if valid_loader else "Not evaluated",
        "best_test_acc1": best_test_acc1,
        "best_test_acc5": best_test_acc5,
        "best_train_acc1": best_train_acc1,
        "best_train_acc5": best_train_acc5,
        "optimizer": optimizer.state_dict(),
        "curr_valid_acc1": valid_acc1 if valid_acc1 else "Not evaluated",
        "curr_test_acc1": test_acc1 if test_acc1 else "Not evaluated",
    }
    save_checkpoint(
        checkpoint,
        False,
        filename=ckpt_base_dir / f"initial.state",
        save=False,
    )

    # initialize the mask buffer
    prev_masks_list = []
    prev_acc_list = []
    prev_bn_list = []

    # resume
    if args.resume:
        best_train_acc1, best_train_acc5, best_valid_acc1, best_valid_acc5, best_test_acc1, best_test_acc5, tosave = resume(args, model, optimizer)

    # evaluate
    if args.evaluate:
        test_acc1, test_acc5 = validate(
            test_loader, model, criterion, args, writer=None, epoch=args.start_epoch
        )
        valid_acc1, valid_acc5 = validate(
            valid_loader, model, criterion, args, writer=None, epoch=args.start_epoch
        )
        sys.exit(0)

    end_epoch = time.time()
    args.start_epoch = args.start_epoch or 0

    with open(log_base_dir/"net structure.txt", "w") as f_out:
        f_out.write(repr(model))

    try:
        # start training
        for epoch in range(args.start_epoch, args.epochs):
            lr_policy(epoch, iteration=None)
            epoch_stats = modifier(args, epoch, model)
            # reset best_test_acc, best_train_acc, best_valid_acc when the prune rate has changed
            if epoch_stats is not None and epoch_stats.get("pr-change", False):
                best_valid_acc1 = 0.0
                best_valid_acc5 = 0.0
                best_test_acc1 = 0.0
                best_test_acc5 = 0.0
                best_train_acc1 = 0.0
                best_train_acc5 = 0.0

            cur_lr = get_lr(optimizer)
            tosave["lr"][epoch] = cur_lr

            if eps_func is not None:
                threshold_this_epoch = eps_func(epoch)
                print(f"set threshold for train_attacker to {threshold_this_epoch}")
                args.train_attacker.adjust_threshold(threshold = threshold_this_epoch)

            # train for one epoch
            start_train = time.time()
            train_acc1, train_acc5, train_loss = train(
                train_loader, model, criterion, optimizer, epoch, args, writer=writer
            )
            train_time.update((time.time() - start_train) / 60)
            tosave["train_loss"][epoch] = train_loss
            tosave["train_acc"][epoch] = train_acc1

            # evaluate on validation set
            start_validation = time.time()
            if valid_loader:
                valid_acc1, valid_acc5, valid_loss = validate(valid_loader, model, criterion, args, writer, epoch)
            else:
                valid_acc1 = "Not evaluated"
                valid_acc5 = "Not evaluated"
                valid_loss = "Not evaluated"
            validation_time.update((time.time() - start_validation) / 60)
            tosave["valid_loss"][epoch] = valid_loss
            tosave["valid_acc"][epoch] = valid_acc1

            # evaluate on test set
            start_test = time.time()
            test_acc1, test_acc5, test_loss = validate(test_loader, model, criterion, args, writer, epoch)
            test_time.update((time.time() - start_test) / 60)
            tosave["test_loss"][epoch] = test_loss
            tosave["test_acc"][epoch] = test_acc1

            # remember best acc@1 and save checkpoint
            if valid_loader:
                is_best = valid_acc1 > best_valid_acc1
                best_valid_acc1 = max(valid_acc1, best_valid_acc1)
                best_valid_acc5 = max(valid_acc5, best_valid_acc5)
                best_test_acc1 = test_acc1 if is_best else best_test_acc1
                best_test_acc5 = test_acc5 if is_best else best_test_acc5
            else:
                is_best = test_acc1 > best_test_acc1
                best_valid_acc1 = "Not evaluated"
                best_valid_acc5 = "Not evaluated"
                best_test_acc1 = max(test_acc1, best_test_acc1)
                best_test_acc5 = max(test_acc5, best_test_acc5)

            best_train_acc1 = max(train_acc1, best_train_acc1)
            best_train_acc5 = max(train_acc5, best_train_acc5)

            # TODO: here we have an assumption: the exploded acc only appears in one epoch, i.e. high -> low -> high
            if args.debug:
                if len(prev_acc_list) < 2:
                    prev_masks_list.append(get_mask_info(args, model))
                    prev_acc_list.append(valid_acc1 if valid_loader else test_acc1)
                    prev_bn_list.append(get_bn_info(model))
                    continue

                explode_flag = is_exploded(valid_acc1, prev_acc_list) if valid_loader else is_exploded(test_acc1, prev_acc_list)
                # TODO: here we save the last 3 epochs for comparison
                if explode_flag or epoch == args.epochs - 1:
                    change_save_dir = log_base_dir / "huge_changes"
                    os.makedirs(change_save_dir, exist_ok=True)

                    # ==== save and analyze mask

                    temp_masks = {}
                    temp_masks[epoch-2] = prev_masks_list[-2]
                    temp_masks[epoch-1] = prev_masks_list[-1]
                    temp_masks[epoch] = get_mask_info(args, model)
                    
                    mask_filename = f"mask{epoch-2}-{epoch}.pkl"
                    path_save_mask = change_save_dir / mask_filename

                    with open(path_save_mask, "wb") as f_out:
                        pickle.dump(temp_masks, f_out)
                        temp_masks = {}
                    
                    monitor_mask_changes(change_save_dir, mask_filename)

                    # ==== save and analyze bn

                    temp_bn = {}
                    temp_bn[epoch-2] = prev_bn_list[-2]
                    temp_bn[epoch-1] = prev_bn_list[-1]
                    temp_bn[epoch] = get_bn_info(model)
                    
                    bn_filename = f"bn{epoch-2}-{epoch}.pkl"
                    path_save_bn = change_save_dir / bn_filename

                    with open(path_save_bn, "wb") as f_out:
                        pickle.dump(temp_bn, f_out)
                        temp_bn = {}
                    
                    monitor_bn_changes(change_save_dir, bn_filename)

                prev_masks_list = [prev_masks_list[-1], get_mask_info(args, model)]
                prev_acc_list = [prev_acc_list[-1], valid_acc1] if valid_loader else [prev_acc_list[-1], test_acc1]
                prev_bn_list = [prev_bn_list[-1], get_bn_info(model)]
                    

            save = ((epoch % args.save_every) == 0) and args.save_every > 0
            if is_best or save or epoch == args.epochs - 1:
                if is_best:
                    print(f"==> New best, saving at {ckpt_base_dir / 'model_best.pth'}")

                
                checkpoint = {
                    "epoch": epoch+1,
                    "arch": args.model_type,
                    "state_dict": model.state_dict(),
                    "best_valid_acc1": best_valid_acc1 if valid_loader else "Not evaluated",
                    "best_valid_acc5": best_valid_acc5 if valid_loader else "Not evaluated",
                    "best_test_acc1": best_test_acc1,
                    "best_test_acc5": best_test_acc5,
                    "best_train_acc1": best_train_acc1,
                    "best_train_acc5": best_train_acc5,
                    "optimizer": optimizer.state_dict(),
                    "curr_valid_acc1": valid_acc1 if valid_acc1 else "Not evaluated",
                    "curr_test_acc1": test_acc1 if test_acc1 else "Not evaluated",
                }
                
                save_checkpoint(
                    checkpoint,
                    is_best,
                    filename=ckpt_base_dir / f"epoch_{epoch}.state",
                    save=save,
                )

            epoch_time.update((time.time() - end_epoch) / 60)
            progress_overall.display(epoch)
            progress_overall.write_to_tensorboard(
                writer, prefix="diagnostics", global_step=epoch
            )

            if args.conv_type == "SampleSubnetConv":
                count = 0
                sum_pr = 0.0
                for n, m in model.named_modules():
                    if isinstance(m, SampleSubnetConv):
                        # avg pr across 10 samples
                        pr = 0.0
                        for _ in range(10):
                            pr += (
                                (torch.rand_like(m.clamped_scores) >= m.clamped_scores)
                                .float()
                                .mean()
                                .item()
                            )
                        pr /= 10.0
                        writer.add_scalar("pr/{}".format(n), pr, epoch)
                        sum_pr += pr
                        count += 1

                args.prune_rate = sum_pr / count
                writer.add_scalar("pr/average", args.prune_rate, epoch)

            writer.add_scalar("test/lr", cur_lr, epoch)
            end_epoch = time.time()

        write_result_to_csv(
            best_test_acc1=best_test_acc1,
            best_test_acc5=best_test_acc5,
            best_valid_acc1=best_valid_acc1,
            best_valid_acc5=best_valid_acc5,
            best_train_acc1=best_train_acc1,
            best_train_acc5=best_train_acc5,
            prune_rate=args.prune_rate,
            curr_test_acc1=test_acc1,
            curr_test_acc5=test_acc5,
            curr_valid_acc1=valid_acc1,
            curr_valid_acc5=valid_acc5,
            base_config=args.config,
            name=args.model_name,
        )

        mask_dir = args.log_base_dir / f"mask-{args.epochs}"
        os.makedirs(mask_dir, exist_ok=True)
        save_mask_info(args, model, mask_dir, no_fig=True)
    
    except Exception as ex:
        print("encounter problems.")
        traceback.print_exception(type(ex), ex, ex.__traceback__)
    
    except KeyboardInterrupt as e:
        print("stopped by user.")

    finally:
        if epoch != args.start_epoch:
            log_file = log_base_dir / "logs.json"
            json.dump(tosave, open(log_file, 'w'))

            fig_file = log_base_dir / "lr_curve.png"
            plot_lr_curve_single(tosave, fig_file)

