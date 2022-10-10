import numpy as np

__all__ = [
    "multistep_lr", 
    "cosine_lr", 
    "constant_lr", 
    "linear_lr",
    "exponential_lr",
    "jump_lr",
    "manual_multistep_lr",
    "get_policy"
]


def get_policy(name):
    if name is None:
        return constant_lr

    out_dict = {
        "constant_lr": constant_lr,
        "cosine_lr": cosine_lr,
        "multistep_lr": multistep_lr,
        "manual_multistep_lr": manual_multistep_lr,
        "linear_lr": linear_lr,
        "exponential_lr": exponential_lr,
        "jump_lr": jump_lr,
    }

    return out_dict[name]


def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def gen_lr_scheduler(optimizer, args, func, **kwargs):
    def _lr_adjuster(epoch, iteration):
        if epoch < args.warmup_length:
            lr = _warmup_lr(args.lr, args.warmup_length, epoch)
        else:
            net_epoch = (epoch - args.warmup_length) % args.pr_epoch
            lr = func(net_epoch)

        # reset the lr at every stage of pruning
        if args.pr_epoch != 1 and (epoch - args.warmup_length) % args.epochs == 0:
            lr = args.lr

        max_v = args.max_v if args.max_v else np.inf
        min_v = args.min_v if args.min_v else -np.inf
        lr = np.clip(lr, a_min=min_v, a_max=max_v)
        
        assign_learning_rate(optimizer, lr)

        return lr

    return _lr_adjuster


def constant_lr(optimizer, args, **kwargs):
    def _lr_func(net_epoch):
        return args.lr

    return gen_lr_scheduler(optimizer, args, _lr_func, **kwargs)


def linear_lr(optimizer, args, **kwargs):
    def _lr_func(net_epoch):
        return args.lr + net_epoch * args.slope

    return gen_lr_scheduler(optimizer, args, _lr_func, **kwargs)


def exponential_lr(optimizer, args, **kwargs):
    def _lr_func(net_epoch):
        return args.lr * args.power ** (net_epoch / args.interval)

    return gen_lr_scheduler(optimizer, args, _lr_func, **kwargs)


def cosine_lr(optimizer, args, **kwargs):
    def _lr_func(net_epoch):
        return (args.alpha + (1 - args.alpha) / 2 * (
            1 + np.cos(np.pi * net_epoch / args.pr_epoch)
            )) * args.lr

    return gen_lr_scheduler(optimizer, args, _lr_func, **kwargs)


def jump_lr(optimizer, args, **kwargs): 
    def _lr_func(net_epoch):
        return args.lr * args.power ** (max(
                net_epoch - args.min_jump_pt + args.jump_freq, 0
                ) // args.jump_freq)

    return gen_lr_scheduler(optimizer, args, _lr_func, **kwargs)


def multistep_lr(optimizer, args, **kwargs):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    def _lr_func(net_epoch):
        return args.lr * (args.lr_gamma ** (net_epoch // args.lr_adjust))

    return gen_lr_scheduler(optimizer, args, _lr_func, **kwargs)


def manual_multistep_lr(optimizer, args, **kwargs):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    def _lr_func(net_epoch):
        max_decay = 1.0
        for ite in args.lr_epochs: 
            if net_epoch >= ite:
                max_decay *= args.lr_gamma
        lr = args.lr * max_decay

        return lr

    return gen_lr_scheduler(optimizer, args, _lr_func, **kwargs)


def _warmup_lr(base_lr, warmup_length, epoch):
    return base_lr * (epoch + 1) / warmup_length

