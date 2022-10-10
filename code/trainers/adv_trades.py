import time
import torch
import tqdm
import sys
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, './')

from util.evaluation import accuracy2
from util.logging import AverageMeter, ProgressMeter


__all__ = ["train", "validate", "modifier"]

# adapted from TRADES
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

    # batch_size = train_loader.batch_size
    num_batches = len(train_loader)
    end = time.time()
    for i, (images, target, idx_batch) in tqdm.tqdm(
        enumerate(train_loader), ascii=True, total=len(train_loader)
    ):
        # measure data loading time
        data_time.update(time.time() - end)

        batch_size = len(images)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)

        target = target.cuda(args.gpu, non_blocking=True)

        criterion_kl = nn.KLDivLoss(size_average=False)

        if (images>1).any():
            print("images should be within [0,1] for pixels. Clamping images...")
            images = torch.clamp(images, 0, 1)

        # use TRADES attacker here, with kl-divergence as objective function
        if args.trades_attacker != None:
            adv_images, _ = args.trades_attacker.attack(model, images, target, criterion_kl)

        # compute output
        output = model(images)

        loss_natural = criterion(output, target)
        
        loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(adv_images), dim=1),
                                                        F.softmax(model(images), dim=1))
        loss = loss_natural + args.beta * loss_robust

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

        if (images>1).any():
            print("images should be within [0,1] for pixels. Clamping images...")
            images = torch.clamp(images, 0, 1)

        # use normal PGD attaker here, which is different from TRADES attacker.
        if args.test_attacker != None:
            images, target = args.test_attacker.attack(model, images, target, criterion)

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
    return
