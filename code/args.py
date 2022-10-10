import argparse

from util.param_parser import DictParser, IntListParser, FloatListParser, BooleanParser, ArgListParser
from util.config_parser import parse_configs_file

#TODO: check possible duplicate names
def parse_args():
    parser = argparse.ArgumentParser(description="AdvPrune project")

    # General Config
    parser.add_argument("--seed", default=None, type=int, 
        help="seed for initializing training. ")
    parser.add_argument("-j","--workers", dest="workers", default=20,type=int,metavar="N",
        help="number of data loading workers (default: 20)",)    
    parser.add_argument("-p","--print-freq",default=10,type=int,metavar="N",
        help="print frequency (default: 10)",)

    # configs
    parser.add_argument("--config", default=None, 
        help="Config file to use (see configs dir)")
    parser.add_argument("--log-dir", dest="log_dir", default=None, 
        help="Where to save the runs. If None use ./runs")
    parser.add_argument('--model_name', dest="model_name", default=None, type=str, 
        help="Experiment name to append to filepath")

    # dataset
    parser.add_argument('--dataset', dest="dataset", type = str, default = None,
        help = 'The dataset used, default = "cifar10".')
    parser.add_argument('--normalize', type = str, default = None,
        help = 'The nomralization mode, default is None.')
    parser.add_argument('--valid_ratio', type = float, default = None,
        help = 'The proportion of the validation set, default is None.')
    parser.add_argument("--num-classes", default=10, type=int,
        help="number of classes")

    # train parameters
    parser.add_argument("--multigpu", default=None, type=lambda x: [int(a) for a in x.split(",")], 
        help="Which GPUs to use for multigpu training",)
    parser.add_argument('--model_type', "-a", dest="model_type", type = str, default = 'resnet',
        help = 'The type of the model, default is "resnet".')
    parser.add_argument("--prune-rate",default=0.0,type=float,
        help="Amount of pruning to do during sparse training")
    parser.add_argument("--pr-steps", default=1.0, type=float,
        help="number of steps to update the prune rate.")
    parser.add_argument("--pr-scale", default=1.0, type=float,
        help="scaling factor when setting the prune rate for different layers.")
    parser.add_argument("--epochs", dest="epochs", type = int, default = 200,
        help = 'The number of epochs, default is 200.')
    parser.add_argument("--start-epoch",default=None,type=int,metavar="N",
        help="manual epoch number (useful on restarts)",)
    parser.add_argument("-b","--batch-size",default=256,type=int,metavar="N",
        help="mini-batch size (default: 256), this is the total batch size of all GPUs on the current node when using "
        "Data Parallel or Distributed Data Parallel")
    parser.add_argument("-e","--evaluate",dest="evaluate",action="store_true",
        help="evaluate model on validation set",)
    parser.add_argument("--debug", action="store_true",
        help="whether or not to save additional information such as huge changes")

    # trainers
    parser.add_argument("--trainer", type=str, default="default", 
        help="cs, ss, or standard training")
    parser.add_argument("--beta", default=6.0, type=float,
        help='regularization, i.e., 1/lambda in TRADES')
    parser.add_argument("--reweight", type=str, default=None,
        help="The reweighting mode for adv_step_fast training, default is None")
    parser.add_argument("--warmup_rw", type=int, default=0,
        help="The warmup period for reweighting for adv_step_fast training, default is 0")

    # --> save
    parser.add_argument("--save_every", default=-1, type=int, 
        help="Save every ___ epochs")

    # --> resume
    parser.add_argument("--pretrained", dest="pretrained", type = str, default = None,
        help = 'The model to be loaded, default is None.')
    parser.add_argument("--resume",default="",type=str,metavar="PATH",
        help="path to latest checkpoint (default: none)",)

    # --> adversary
    parser.add_argument('--eps_schedule', action = DictParser, default = None,
        help = 'The scheduler of the adversarial budget, default is None')
    parser.add_argument('--attack', action = DictParser, default = None,
        help = 'Play adversarial attack or not, default = None, use name=h to obtain help messages.')
    parser.add_argument('--eval-attack', action = DictParser, default = None,
        help = 'Play adversarial attack for testing or not, default = None, use name=h to obtain help messages.')
    parser.add_argument("--trades_attack", default=None, action=DictParser,
        help='settings for generating adv examples in TRADES')
    parser.add_argument('--subset', type = str, default = 'test',
        help = 'Specify which set is used for the attacks, default = "test".')

    # learning rate scheduler
    parser.add_argument("--lr","--learning-rate",default=0.1,type=float,metavar="LR",
        help="initial learning rate",dest="lr",)
    parser.add_argument("--lr-policy", default="constant_lr", 
        help="Policy for the learning rate.")
    parser.add_argument("--min-v", type=float,
        help="min value for learning rate")
    parser.add_argument("--max-v", type=float,
        help="max value for learning rate")
    parser.add_argument("--warmup_length", default=0, type=int, 
        help="Number of warmup iterations")

    # --> linear
    parser.add_argument("--slope", type=float, 
        help="slope for linear lr scheduler")

    # --> exponential
    parser.add_argument("--power", type=float, 
        help="power for exponential lr scheduler")
    parser.add_argument("--interval", type=int,
        help="interval for exponential lr scheduler")

    # --> cosine
    parser.add_argument("--alpha", type=float, default=0,
        help="alpha for cosine lr scheduler")

    # --> jump
    parser.add_argument("--min-jump-pt", type=int,
        help="min jump pt for jump lr scheduler")
    parser.add_argument("--jump-freq", type=int,
        help="jump frequency for jump lr scheduler")

    # --> multistep
    parser.add_argument("--lr-adjust", default=30, type=int, 
        help="Interval to drop lr")
    parser.add_argument("--lr-gamma", default=0.1, type=float, 
        help="Multistep multiplier")

    # --> manual_multistep
    parser.add_argument("--lr-epochs", default=None, action=IntListParser, 
        help="Interval to drop lr")

    # optimizer: general
    parser.add_argument("--optimizer", default="sgd", 
        help="Which optimizer to use")
    parser.add_argument("--wd","--weight-decay",default=1e-4,type=float,metavar="W",dest="weight_decay",
        help="weight decay (default: 1e-4)",)

    # --> SGD
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", 
        help="momentum")
    parser.add_argument("--nesterov",default=False,action="store_true",
        help="Whether or not to use nesterov for SGD",)
    parser.add_argument("--dampening", default=0, type=float,
        help="dampening in SGD")

    # --> Adam
    parser.add_argument("--beta1", default=0.9, type=float,
        help="beta1 for Adam")
    parser.add_argument("--beta2", default=0.999, type=float,
        help="beta2 for Adam")
    parser.add_argument("--eps", default=1e-8, type=float,
        help="eps for Adam")
    parser.add_argument("--amsgrad", action="store_true",
        help="whether to use the AMSGrad variant of this algorithm")

    # activation related
    parser.add_argument("--nonlinearity", default="relu", 
        help="Nonlinearity used by initialization")
    parser.add_argument("--leaky-relu-slope", default=0, type=float, 
        help="slope for leaky relu.")

    # weight init related
    parser.add_argument("--init", default="kaiming_normal", 
        help="Weight initialization modifications")
    parser.add_argument("--scale-fan", action="store_true", default=False, 
        help="scale fan")

    # conv related
    parser.add_argument("--score-init-scale", default=1.0, type=float, 
        help="Score initialization scale in SubnetConv")
    parser.add_argument("--fan-scaled-score-mode", type=str, choices=["except_last", "only_last", "all", "none"], default="all",
        help="whether to use fan-in to scale scores.")
    parser.add_argument("--binconv-freeze-rand", action="store_true",
        help="whether to freeze binarized weights or not in BinConv.")
    parser.add_argument("--binconv-stochastic", action="store_true",
        help="whether to use stochastic binconv or not.")

    # conv type related
    parser.add_argument("--conv-type", type=str, default=None, 
        help="What kind of sparsity to use")
    parser.add_argument("--first-layer-dense", action="store_true", 
        help="First layer dense or sparse")
    parser.add_argument("--first-layer-type", type=str, default=None, 
        help="Conv type of first layer")
    
    # bn type related
    parser.add_argument("--bn-type", default=None, 
        help="BatchNorm type")
    parser.add_argument("--end-with-bn", action="store_true", 
        help="whether or not to add a bn layer after fc")
    parser.add_argument("--last-layer-dense", action="store_true", 
        help="Last layer dense or sparse")
    parser.add_argument("--last-layer-type", type=str, default=None, 
        help="Conv type of last layer")
    parser.add_argument("--no-bn-decay", action="store_true", default=False, 
        help="No batchnorm decay")

    # network structure
    parser.add_argument("--mode", default="fan_in", 
        help="Weight initialization mode")
    parser.add_argument("--label-smoothing",type=float,default=None,
        help="Label smoothing to use, default 0.0",)    
    parser.add_argument("--freeze-weights",action="store_true",
        help="Whether or not to train only subnet (this freezes weights)",)

    ## others
    parser.add_argument("--file", type=str, help="checkpoints that you want to visualize")
    parser.add_argument("--explode-th", type=float, default=5.0, help="threshold (in %) for identifying huge changes in learning curve")

    return parser.parse_args()

args = parse_args()
parse_configs_file(args)