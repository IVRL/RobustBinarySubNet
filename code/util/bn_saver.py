import sys
import torch
sys.path.insert(0, './')


def get_bn_info(model):
    if(isinstance(model, torch.nn.DataParallel)):
        model = model.module

    value_dict = {}
    for name, module in model.named_modules():
        if "bn" not in name:
            continue
        mean_values = module.running_mean.cpu().detach().numpy()
        var_values = module.running_var.cpu().detach().numpy()
        
        value_dict[name] = {"mean": mean_values, "var": var_values}
    
    return value_dict

