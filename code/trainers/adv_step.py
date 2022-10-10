import time
import torch
import tqdm
import os
import sys
import pdb
import numpy as np
sys.path.insert(0, './')

from util.evaluation import accuracy2
from util.logging import AverageMeter, ProgressMeter
from util.pr_scheduler import prune_rate_scheduler
from util.mask_saver import save_mask_info


__all__ = ["train", "validate", "modifier"]

DEBUG_GRADIENT = False


def train(train_loader, model, criterion, optimizer, epoch, args, writer):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.3f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix=f"Epoch: [{epoch}]",
    )

    # switch to train mode
    model.train()

    batch_size = train_loader.batch_size
    num_batches = len(train_loader)
    end = time.time()

    if DEBUG_GRADIENT:
        print(model)

        for name, layer in model.named_modules():
            if name == "module":
                continue
            if hasattr(layer, "scores"):
                print(f"{name}-> unique ratio:{torch.numel(torch.unique(layer.scores)) / torch.numel(layer.scores)}," \
                    f"mean:{torch.mean(layer.scores)}, std:{torch.std(layer.scores)}, theoretical std: {torch.max(layer.scores) / np.sqrt(3)}"
                )

    accum_zero_masks = {}

    for i, (images, target, idx_batch) in tqdm.tqdm(
        enumerate(train_loader), ascii=True, total=len(train_loader)
    ):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)

        target = target.cuda(args.gpu, non_blocking=True)

        if args.train_attacker != None:
            images, target = args.train_attacker.attack(model, images, target, criterion)

        assert not (images>1).any(), f"images should be within [0,1] for pixels"

        # compute output
        output = model(images)

        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy2(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            t = (num_batches * epoch + i) * batch_size
            progress.display(i)
            progress.write_to_tensorboard(writer, prefix="train", global_step=t)

    
    if DEBUG_GRADIENT:
        i=0
        for name, layer in model.named_modules():
            if name == "module":
                continue
            skip_flag = True
            for item in ["conv", "shortcut.0", "fc"]:
                if item in name:
                    skip_flag = False
                    break
            if skip_flag:
                continue
            if hasattr(layer, "scores"):
                if i == 0:
                    accum_zero_masks[name] = layer.scores.grad.detach()
                else:
                    accum_zero_masks[name] *= layer.scores.grad.detach()
                print(f"{name}-> #elements:{torch.numel(layer.scores.grad)} "\
                f"nonzero ratio:{torch.count_nonzero(layer.scores.grad) / torch.numel(layer.scores.grad)} "\
                f"accum:{torch.count_nonzero(accum_zero_masks[name]) / torch.numel(accum_zero_masks[name])} "\
                f"mean:{torch.mean(layer.scores.grad)} "\
                f"std{torch.std(layer.scores.grad)}")
            else:
                print(f"{name}-> no scores for this layer. only fixed weights")
        pdb.set_trace()
    
    return top1.avg, top5.avg, losses.avg


def validate(val_loader, model, criterion, args, writer, epoch):
    batch_time = AverageMeter("Time", ":6.3f", write_val=False)
    losses = AverageMeter("Loss", ":.3f", write_val=False)
    top1 = AverageMeter("Acc@1", ":6.2f", write_val=False)
    top5 = AverageMeter("Acc@5", ":6.2f", write_val=False)
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], prefix="Test: "
    )

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (images, target, idx_batch) in tqdm.tqdm(
        enumerate(val_loader), ascii=True, total=len(val_loader)
    ):

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)

        target = target.cuda(args.gpu, non_blocking=True)

        if args.test_attacker != None:
            images, target = args.test_attacker.attack(model, images, target, criterion)

        assert not (images>1).any(), f"images should be within [0,1] for pixels"

        with torch.no_grad():
            # compute output
            output = model(images)

            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy2(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

    progress.display(len(val_loader))

    if writer is not None:
        progress.write_to_tensorboard(writer, prefix="test", global_step=epoch)

    return top1.avg, top5.avg, losses.avg


def modifier(args, epoch, model):
    print("enter modifier...")
    change_flag = prune_rate_scheduler(epoch, args, model)
    # save the masks
    if change_flag:
        output_dir = args.log_base_dir / f"mask-{epoch}"
        os.makedirs(output_dir, exist_ok=True)
        save_mask_info(args, model, output_dir, no_fig=True)
    return {"pr-change":change_flag}

