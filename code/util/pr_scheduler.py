import sys
sys.path.insert(0, './')

from arch.net_utils import set_model_prune_rate

def set_pr_scheduler_params(args):
    assert(args.pr_steps >= 1)
    args.pr_epoch = (args.epochs - args.warmup_length) / args.pr_steps
    args.pr_ratio = args.prune_rate ** (1/args.pr_steps)
    args.current_pr = 1.0

def prune_rate_scheduler(epoch, args, model):
    if (epoch - args.warmup_length) % args.pr_epoch == 0:
        args.current_pr *= args.pr_ratio
        print(f"epoch {epoch}: start changing prune rate to {args.current_pr}...")
        set_model_prune_rate(model, args.current_pr, p=args.pr_scale)
        return True
    return False