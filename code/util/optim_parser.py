# parse the optimizer configuration

import torch
import numpy as np

instructions = '''
instructions for setting an optimizer
>>> SGD
name=sgd,lr=$LR$,momentum=$0.9$,dampening=$0$,weight_decay=$0$

>>> Adagrad
name=adagrad,lr=$LR$,lr_decay=$0$,weight_decay=$0$

>>> Adadelta
name=adadelta,lr=$LR$,rho=$0.9$,eps=$1e-6$,weight_decay=$0$

>>> Adam
name=adam,lr=$LR$,beta1=$0.9$,beta2=$0.999$,eps=$1e-8$,weight_decay=$0$,amsgrad=$0$

>>> RMSprop
name=rmsprop,lr=$LR$,alpha=$0.99$,eps=$1e-8$,weight_decay=$0$,momentum=$0$
'''

def parse_optim(policy, params):

    kwargs = {}

    if policy['name'].lower() in ['sgd']:

        kwargs['lr'] = policy['lr']
        kwargs['momentum'] = policy['momentum'] if 'momentum' in policy else 0.9
        kwargs['dampening'] = policy['dampening'] if 'dampening' in policy else 0
        kwargs['weight_decay'] = policy['weight_decay'] if 'weight_decay' in policy else 0
        optimizer = torch.optim.SGD(params, **kwargs)

    elif policy['name'].lower() in ['adagrad']:

        kwargs['lr'] = policy['lr']
        kwargs['lr_decay'] = policy['lr_decay'] if 'lr_decay' in policy else 0.
        kwargs['weight_decay'] = policy['weight_decay'] if 'weight_decay' in policy else 0.
        optimizer = torch.optim.Adagrad(params, **kwargs)

    elif policy['name'].lower() in ['adadelta']:

        kwargs['lr'] = policy['lr']
        kwargs['rho'] = policy['rho'] if 'rho' in policy else 0.9
        kwargs['eps'] = policy['eps'] if 'eps' in policy else 1e-6
        kwargs['weight_decay'] = policy['weight_decay'] if 'weight_decay' in policy else 0.
        optimizer = torch.optim.Adadelta(params, **kwargs)

    elif policy['name'].lower() in ['adam']:

        kwargs['lr'] = policy['lr']
        kwargs['betas'] = (policy['beta1'] if 'beta1' in policy else 0.9, policy['beta2'] if 'beta2' in policy else 0.999)
        kwargs['eps'] = policy['eps'] if 'eps' in policy else 1e-8
        kwargs['weight_decay'] = policy['weight_decay'] if 'weight_decay' in policy else 0.
        kwargs['amsgrad'] = True if 'amsgrad' in policy and np.abs(policy['amsgrad']) > 1e-6 else False
        optimizer = torch.optim.Adam(params, **kwargs)

    elif policy['name'].lower() in ['rmsprop']:

        kwargs['lr'] = policy['lr']
        kwargs['alpha'] = policy['alpha'] if 'alpha' in policy else 0.99
        kwargs['eps'] = policy['eps'] if 'eps' in policy else 1e-8
        kwargs['weight_decay'] = policy['weight_decay'] if 'weight_decay' in policy else 0.
        kwargs['momentum'] = policy['momentum'] if 'momentum' in policy else 0.
        optimizer = torch.optim.RMSprop(params, **kwargs)

    elif policy['name'].lower() in ['h', 'help']:

        print(instructions)
        exit(0)

    else:
        raise ValueError('Unrecognized policy: %s'%policy)

    print('Optimizer : %s --'%policy['name'])
    for key in kwargs:
        print('%s: %s'%(key, kwargs[key]))
    print('-----------------')

    return optimizer

def set_optim_params(optim, params):

    if isinstance(params, torch.Tensor):
        raise TypeError("params argument given to the optimizer should be an iterable of Tensors or dicts, but got " + torch.typename(params))

    param_groups = list(params)
    if len(param_groups) == 0:
        raise ValueError('optimizer got an empty parameter list')
    if not isinstance(param_groups[0], dict):
        param_groups = [{'params': param_groups}]

    optim.param_groups = []
    for param_group in param_groups:
        optim.add_param_group(param_group)

    return optim

def get_optimizer(args, model, verbose=False):
    for n, v in model.named_parameters():
        if v.requires_grad and verbose:
            print("<DEBUG> gradient to", n)

        if not v.requires_grad and verbose:
            print("<DEBUG> no gradient to", n)

    policy_name = args.optimizer

    if policy_name.lower() in ['sgd']:
        parameters = list(model.named_parameters())
        bn_params = [v for n, v in parameters if ("bn" in n) and v.requires_grad]
        rest_params = [v for n, v in parameters if ("bn" not in n) and v.requires_grad]
        optimizer = torch.optim.SGD(
            [
                {
                    "params": bn_params,
                    "weight_decay": 0 if args.no_bn_decay else args.weight_decay,
                },
                {"params": rest_params, "weight_decay": args.weight_decay},
            ],
            args.lr,
            momentum=args.momentum if args.momentum else 0.9,
            dampening=args.dampening if args.dampening else 0,
            weight_decay=args.weight_decay if args.weight_decay else 0,
            nesterov=args.nesterov,
        )

    elif policy_name.lower() in ['adam']:
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=args.lr,
            betas=(
                args.beta1 if args.beta1 else 0.9,
                args.beta2 if args.beta2 else 0.999,
            ),
            eps=args.eps if args.eps else 1e-8,
            weight_decay=args.weight_decay if args.weight_decay else 0,
            amsgrad=args.amsgrad
        )

    else:
        raise ValueError('Unrecognized policy: %s'%policy_name)

    print('Optimizer : %s --'%policy_name)
    for key in optimizer.state_dict()["param_groups"][0]:
        if key != "params":
            print(f"{key}: {optimizer.state_dict()['param_groups'][0][key]}")
    print('-----------------')

    return optimizer