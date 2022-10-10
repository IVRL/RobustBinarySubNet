import os
import torch
import torch.backends.cudnn as cudnn

def config_visible_gpu(device_config):

    if device_config != None and device_config != 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = device_config
        print('Use GPU %s'%device_config)
    else:
        print('Use all GPUs' if device_config == None else 'Use CPUs')


def set_gpu(args, model):
    assert torch.cuda.is_available(), "CPU-only experiments currently unsupported"

    use_gpu = True
    device = None
    if args.multigpu is None:
        device = torch.device("cpu")
        use_gpu = False
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        print(f"=> Parallelizing on {args.multigpu} gpus")
        torch.cuda.set_device(f"cuda:{args.multigpu[0]}")
        device = torch.device(f"cuda:{args.multigpu[0]}")
        args.gpu = args.multigpu[0]
        model = torch.nn.DataParallel(model, device_ids=args.multigpu).to(device)

    cudnn.benchmark = True

    return model, use_gpu, device
