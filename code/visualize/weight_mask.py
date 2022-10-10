import os
import sys
import numpy as np

import torch
import torch.backends.cudnn as cudnn

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

mpl.use('Agg')
sys.path.insert(0, './')

from args import args
import arch as models
from arch.conv_type import GetSubnet, SubnetConv
from util.model_parser import parse_model
from util.mask_saver import heatmap_weights

SAVE_EXTENSION = None


def set_gpu(args, model):
    assert torch.cuda.is_available(), "CPU-only experiments currently not supported"

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    elif args.multigpu is None:
        device = torch.device("cpu")
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        print(f"=> Parallelizing on {args.multigpu} gpus")
        torch.cuda.set_device(args.multigpu[0])
        args.gpu = args.multigpu[0]
        model = torch.nn.DataParallel(model, device_ids=args.multigpu).cuda(
            args.multigpu[0]
        )

    cudnn.benchmark = True

    return model


def distribution_weights(exp_name, name, weight, masks, pr):
    output_dir = f"./figs/{exp_name}/"
    if not os.path.isdir(output_dir):
        os.makedirs(f"./figs/{exp_name}/")

    plots = []
    num_trials = 500

    # simulation - irregular
    row_trials = []
    col_trials = []
    for _ in range(num_trials):
        num_rows, num_cols, _, _ = masks.shape
        random_matrix = np.zeros_like(masks).reshape(-1)
        indices = np.random.choice(range(len(random_matrix)), int(np.sum(masks)))
        random_matrix[indices] = 1.0
        random_matrix = random_matrix.reshape(masks.shape)

        merged_masks = np.sum(random_matrix, axis=(2,3))

        rows_masks = np.sum(merged_masks, axis=1)
        cols_masks = np.sum(merged_masks, axis=0)

        sorted_rows = np.sort(rows_masks)[::-1].reshape(1,-1)
        sorted_cols = np.sort(cols_masks)[::-1].reshape(1,-1)

        row_trials.append(sorted_rows)
        col_trials.append(sorted_cols)

    merged_row_trials = np.concatenate(row_trials, axis=0)
    merged_col_trials = np.concatenate(col_trials, axis=0)

    mean_row_trials = np.mean(merged_row_trials, axis=0)
    # std_row_trials = np.std(merged_row_trials, axis=0)
    mean_col_trials = np.mean(merged_col_trials, axis=0)
    # std_col_trials = np.std(merged_col_trials, axis=0)

    plots.append([mean_row_trials, mean_col_trials])

    # pruned
    merged_masks = np.sum(masks, axis=(2,3))

    rows_masks = np.sum(merged_masks, axis=1)
    cols_masks = np.sum(merged_masks, axis=0)

    plots.append([rows_masks, cols_masks])

    fig1, ax1 = plt.subplots(1,1,figsize=(5,5))
    fig2, ax2 = plt.subplots(1,1,figsize=(5,5))
    ax1.set_xlabel("Channel index", fontsize=16)
    ax1.set_ylabel("Number of remaining weights", fontsize=16)
    ax2.set_xlabel("Channel index", fontsize=16)
    ax2.set_ylabel("Number of remaining weights", fontsize=16)

    x1_title_str = []
    x2_title_str = []
    x1_y_lim = 0
    x2_y_lim = 0

    for [rows, cols],c,l in zip(plots, ["blue", "red"], ["random", "pruned"]):

        empty_rows = np.sum(rows == 0)
        empty_cols = np.sum(cols == 0)

        print(f"===> Input {num_cols} channels, {empty_cols} empty ones. Output {num_rows} channels, {empty_rows} empty ones.")

        sorted_row_idxs = np.argsort(rows)[::-1]
        sorted_col_idxs = np.argsort(cols)[::-1]

        sorted_rows = rows[sorted_row_idxs]
        sorted_cols = cols[sorted_col_idxs]

        ax1.plot(range(num_rows),sorted_rows, c=c, label=l)
        ax1.fill_between(range(num_rows), 0, sorted_rows, facecolor=c, alpha=0.1, interpolate=True)

        x1_title_str.append(f"{empty_rows}/{num_rows}")        
        x1_y_lim = max(x1_y_lim, np.max(sorted_rows))

        # axs[0].set_xticklabels([])

        ax2.plot(range(num_cols),sorted_cols, c=c, label=l)
        ax2.fill_between(range(num_cols), 0, sorted_cols, facecolor=c, alpha=0.1, interpolate=True)

        x2_title_str.append(f"{empty_cols}/{num_cols}")
        x2_y_lim = max(x2_y_lim, np.max(sorted_cols))
        
        # axs[1].set_xticklabels([])

    # axs[0].fill_between(range(num_rows), mean_row_trials - 3 * std_row_trials, mean_row_trials + 3 * std_row_trials, facecolor='red', alpha=0.2, interpolate=True)
    # axs[1].fill_between(range(num_cols), mean_col_trials - 3 * std_col_trials, mean_col_trials + 3 * std_col_trials, facecolor='red', alpha=0.2, interpolate=True)

    # axs[0].set_title(f"output {' '.join(x1_title_str)} empty")
    ax1.set_xlim((0, num_rows - 1))
    ax1.set_ylim((0, 1.01 * x1_y_lim))
    ax1.legend(fontsize=16)

    # axs[1].set_title(f"input {' '.join(x2_title_str)} empty")
    ax2.set_xlim((0, num_cols - 1))
    ax2.set_ylim((0, 1.01 * x2_y_lim))
    ax2.legend(fontsize=16)

    # fig.suptitle(f"prune rate {pr}")

    fig1.savefig(f"./figs/{exp_name}/{name}-distribution-out.{SAVE_EXTENSION}", dpi=300, bbox_inches='tight')
    fig2.savefig(f"./figs/{exp_name}/{name}-distribution-in.{SAVE_EXTENSION}", dpi=300, bbox_inches='tight')
    # fig.savefig(f"./figs/{exp_name}/{name}-distribution.{SAVE_EXTENSION}", dpi=300, bbox_inches='tight')
    plt.close("all")


def distribution_weights_2(exp_name, name, weight, masks, pr):
    output_dir = f"./figs/{exp_name}/"
    if not os.path.isdir(output_dir):
        os.makedirs(f"./figs/{exp_name}/")

    kernel_size = masks.shape[3]

    prune_rate = np.count_nonzero(masks) / np.size(masks)
    print(f"prune rate: {prune_rate}")
    random_left = kernel_size ** 2 * prune_rate

    num_trials = 100
    trials = []

    for _ in range(num_trials):
        num_rows, num_cols, _, _ = masks.shape
        random_matrix = np.zeros_like(masks).reshape(-1)
        indices = np.random.choice(range(len(random_matrix)), int(np.sum(masks)))
        random_matrix[indices] = 1.0
        random_matrix = random_matrix.reshape(masks.shape)

        merged_masks = np.sum(random_matrix, axis=(2,3)).reshape(-1).astype(int)

        weight_counts = []
        for i in range(kernel_size**2+1):
            weight_counts.append(np.count_nonzero(merged_masks == i))

        trials.append(weight_counts)

    mean_trials = np.mean(trials, axis=0)
    mean_masks = []
    for idx, count in enumerate(mean_trials):
        for _ in range(int(count)):
            mean_masks.append(idx)
    mean_masks = np.array(mean_masks)

    print(f"trial: {np.size(mean_masks)}, actual: {np.size(merged_masks)}")

    plt.hist(mean_masks, bins=kernel_size**2+1, range=(0, kernel_size**2), log=True, alpha=0.2, align='mid', color='red', label='random')

    # pruned
    merged_masks = np.sum(masks, axis=(2,3)).reshape(-1).astype(int)

    # merged_masks = merged_masks[merged_masks != 0]

    max_y = np.max(np.bincount(merged_masks))
    plt.hist(merged_masks, bins=kernel_size**2+1, range=(0, kernel_size**2), log=True, alpha=0.2, align='mid', color='blue', label='pruned')

    # kernel-wise
    # kernel_masks = np.zeros_like(merged_masks)
    # kernel_count = math.ceil(np.count_nonzero(masks) / kernel_size**2)
    # kernel_masks[:kernel_count] = kernel_size ** 2
    # plt.hist(kernel_masks, bins=kernel_size**2+1, range=(0, kernel_size**2), log=True, alpha=0.2, align='mid', color='green', label='kernel')

    # plt.axvline(random_left, label="random", color='red')
    plt.xlim(0, kernel_size**2)
    plt.xlabel("Number of kept weights in a kernel")
    plt.ylabel("Number of kernels")
    plt.legend()

    plt.savefig(f"./figs/{exp_name}/{name}-distribution2.{SAVE_EXTENSION}", dpi=300, bbox_inches='tight')
    # fig.savefig(f"./figs/{exp_name}/{name}-distribution.{SAVE_EXTENSION}", dpi=300, bbox_inches='tight')
    plt.close("all")

def plot_two_layers(exp_name, name, mask1, mask2):
    kernel_size = mask1.shape[3]

    merged_mask1 = np.vstack((np.hstack(mask1[i]) for i in list(range(mask1.shape[0]))))
    merged_mask2 = np.vstack((np.hstack(mask2[i]) for i in list(range(mask2.shape[0])))).transpose()

    height = merged_mask1.shape[0]
    width = merged_mask1.shape[1]

    plot_width = max(10, int(25.0/384.0*width))*2 + 2
    plot_height = max(10, int(25.0/384.0*height))

    fig = plt.figure(figsize=(plot_width, plot_height))
    grid_size_width = width
    grid_size_height = height
    bar_size = max(1,min(15, int(0.05*height), int(0.05*width)))
    gs = gridspec.GridSpec(grid_size_height, grid_size_width * 2 + bar_size*2, wspace=10, hspace=10)
    ax_mask1 = plt.subplot(gs[:, :grid_size_width])
    ax_point1 = plt.subplot(gs[:, grid_size_width:grid_size_width+bar_size], sharey=ax_mask1)
    ax_mask2 = plt.subplot(gs[:, grid_size_width + bar_size*2:])
    ax_point2 = plt.subplot(gs[:, grid_size_width+bar_size: grid_size_width+bar_size*2], sharey=ax_mask2)

    ax_mask1.spy(merged_mask1, markersize=4, markeredgewidth=0)
    ax_mask1.set_xticks(np.arange(-0.5,width,3))
    ax_mask1.set_yticks(np.arange(-0.5,height,3))
    ax_mask1.set_xticklabels([])
    ax_mask1.set_yticklabels([])
    ax_mask1.set_xlim((-0.5,width-0.5))
    ax_mask1.set_ylim((height-0.5, -0.5))
    # remove small lines of the ticks
    ax_mask1.xaxis.set_ticks_position('none')
    ax_mask1.yaxis.set_ticks_position('none')

    ax_mask1.grid()

    ax_mask2.spy(merged_mask2, markersize=4, markeredgewidth=0)
    ax_mask2.set_xticks(np.arange(-0.5,width,3))
    ax_mask2.set_yticks(np.arange(-0.5,height,3))
    ax_mask2.set_xticklabels([])
    ax_mask2.set_yticklabels([])
    ax_mask2.set_xlim((-0.5,width-0.5))
    ax_mask2.set_ylim((height-0.5, -0.5))
    # remove small lines of the ticks
    ax_mask2.xaxis.set_ticks_position('none')
    ax_mask2.yaxis.set_ticks_position('none')

    ax_mask2.grid()

    verts2 = list(zip([-60.,-60.,60.,60],[-12.5,12.5,12.5,-12.5]))

    rows_masks1 = np.sign(np.sum(merged_mask1, axis=1))
    rows_masks2 = np.sign(np.sum(merged_mask2, axis=1))

    rows_indices1 = np.nonzero(rows_masks1)[0] * kernel_size + int((kernel_size-1)/2)
    rows_indices2 = np.nonzero(rows_masks2)[0] * kernel_size + int((kernel_size-1)/2)

    ax_point1.scatter(np.zeros_like(rows_indices1), rows_indices1, c='blue', marker=verts2, s=54**2, edgecolors='none')
    ax_point1.set_xticks([])
    ax_point1.set_xticklabels([])
    # ax_point1.grid()
    ax_point1.xaxis.set_ticks_position('none')
    ax_point1.yaxis.set_ticks_position('none')

    ax_point2.scatter(np.zeros_like(rows_indices2), rows_indices2, c='blue', marker=verts2, s=54**2, edgecolors='none')
    ax_point2.set_xticks([])
    ax_point2.set_xticklabels([])
    # ax_point1.grid()
    ax_point2.xaxis.set_ticks_position('none')
    ax_point2.yaxis.set_ticks_position('none')

    ax_mask1.set_aspect('auto')
    ax_mask2.set_aspect('auto')
    ax_point1.set_aspect('auto')
    ax_point2.set_aspect('auto')

    fig.savefig(f"./figs/{exp_name}/2layers-{name}.{SAVE_EXTENSION}", dpi=300, bbox_inches='tight')
    plt.close("all")

def calculate_flop(flop, masks, layer_type="conv"):
    pr = np.mean(masks.astype(float))
    r_out, r_in, c, _ = masks.shape
    print(pr, r_in, r_out, c)
    r_out = float(r_out)
    r_in = float(r_in)
    pr = float(pr)
    c = float(c)
    if layer_type == "conv":

        s = 2048.0 / r_out

        binary_add = 0.0
        binary_add += pr * s*s * c*c * r_in * r_out + 11 * r_out * s*s
        binary_add += 3 * pr * c*c * s*s * r_in * r_out + 4 * r_out * s*s
        flop["binary forward"] += pr * s*s * c*c * r_in * r_out + 11 * r_out * s*s
        flop["binary backward"] += 3 * pr * c*c * s*s * r_in * r_out + 4 * r_out * s*s
        flop["binary"] += binary_add

        fp_add = 0.0
        fp_add += 2 * pr * s*s * c*c * r_in * r_out + 11 * r_out * s*s
        fp_add += 4 * pr * c*c * s*s * r_in * r_out + 4 * r_out * s*s + c*c * r_in * r_out
        flop["fp forward"] += 2 * pr * s*s * c*c * r_in * r_out + 11 * r_out * s*s
        flop["fp backward"] += 4 * pr * c*c * s*s * r_in * r_out + 4 * r_out * s*s + c*c * r_in * r_out
        flop["fp"] += fp_add

        print(f"binary: {binary_add} fp: {fp_add}")

    elif layer_type == "linear":
        s = 1.0

        binary_add = 0.0
        binary_add += pr * s*s * c*c * r_in * r_out + 10 * r_out * s*s
        binary_add += 3 * pr * c*c * s*s * r_in * r_out + 3 * r_out * s*s
        flop["binary forward"] += pr * s*s * c*c * r_in * r_out + 10 * r_out * s*s
        flop["binary backward"] += 3 * pr * c*c * s*s * r_in * r_out + 3 * r_out * s*s
        flop["binary"] += binary_add

        fp_add = 0.0
        fp_add += 2 * pr * s*s * c*c * r_in * r_out + 10 * r_out * s*s
        fp_add += 4 * pr * c*c * s*s * r_in * r_out + 3 * r_out * s*s + c*c * r_in * r_out
        flop["fp forward"] += 2 * pr * s*s * c*c * r_in * r_out + 10 * r_out * s*s
        flop["fp backward"] += 4 * pr * c*c * s*s * r_in * r_out + 3 * r_out * s*s + c*c * r_in * r_out
        flop["fp"] += fp_add

        print(f"binary: {binary_add} fp: {fp_add}")
    else:
        raise ValueError(f"Unrecognized layer {layer_type}")

def get_masks(conv):
    if(isinstance(conv, SubnetConv)):
        if conv.prune_rate != 1.0:
            subnet =  GetSubnet.apply(conv.clamped_scores, conv.prune_rate).cpu().data.numpy()
            return subnet
        else:
            return np.ones_like(conv.weight.data.cpu().data.numpy())
    else:
        raise ValueError(f"{type(conv)} not supported for subnets.")

def plot_mask(func_names):
    print(args)
    args.gpu = None
    device = torch.device("cuda:0")
    checkpoint = torch.load(args.file, map_location=device)

    print(checkpoint['arch'])

    model = parse_model(dataset = args.dataset, model_type = args.model_type, normalize = args.normalize, args=args)
    model = set_gpu(args, model)

    model.load_state_dict(checkpoint["state_dict"])

    if(isinstance(model, torch.nn.DataParallel)):
        model = model.module

    model = model[1]

    if(isinstance(model, models.resnet_cifar.ResNet) or isinstance(model, models.resnet.ResNet)):
        weights = {"conv1": model.conv1.weight.cpu().data.numpy(), "fc":model.fc.weight.cpu().data.numpy()}
        masks = {"conv1": get_masks(model.conv1), "fc":get_masks(model.fc)}
        layers = [model.layer1, model.layer2, model.layer3, model.layer4]
        prs = {"conv1": model.conv1.prune_rate, "fc": model.fc.prune_rate}
        for i, layer in enumerate(layers):
            for j, block in enumerate(layer):
                if(isinstance(block, models.resnet_cifar.BasicBlock) or isinstance(block, models.resnet.BasicBlock)):
                    weights[f"layer{i}_block{j}_conv1"] = block.conv1.weight.cpu().data.numpy()
                    weights[f"layer{i}_block{j}_conv2"] = block.conv2.weight.cpu().data.numpy()
                    masks[f"layer{i}_block{j}_conv1"] = get_masks(block.conv1)
                    masks[f"layer{i}_block{j}_conv2"] = get_masks(block.conv2)
                    prs[f"layer{i}_block{j}_conv1"] = block.conv1.prune_rate
                    prs[f"layer{i}_block{j}_conv2"] = block.conv2.prune_rate
                elif(isinstance(block, models.resnet_cifar.Bottleneck) or isinstance(block, models.resnet.Bottleneck)):
                    weights[f"layer{i}_block{j}_conv1"] = block.conv1.weight.cpu().data.numpy()
                    weights[f"layer{i}_block{j}_conv2"] = block.conv2.weight.cpu().data.numpy()
                    weights[f"layer{i}_block{j}_conv3"] = block.conv3.weight.cpu().data.numpy()
                    masks[f"layer{i}_block{j}_conv1"] = get_masks(block.conv1)
                    masks[f"layer{i}_block{j}_conv2"] = get_masks(block.conv2)
                    masks[f"layer{i}_block{j}_conv3"] = get_masks(block.conv3)
                    prs[f"layer{i}_block{j}_conv1"] = block.conv1.prune_rate
                    prs[f"layer{i}_block{j}_conv2"] = block.conv2.prune_rate
                    prs[f"layer{i}_block{j}_conv3"] = block.conv3.prune_rate
                else:
                    raise ValueError(f"no such block: {block}")

        # for name in ["layer2, block5, conv2"]:
        for name in weights:
            # if name != "layer0, block0, conv2":
            #     continue
            if masks[name] is None:
                print(f"{name} has pruning rate 1.0, skip plotting.")
                continue
            print(f"now plot heatmap on {name}, shape: {weights[name].shape}") 
            for func_name in func_names:
                if func_name == "heatmap":
                    heatmap_weights(name, weights[name],masks[name], f"./figs/{args.model_name.replace('plot_mask_', '')}/")
                elif func_name == "distribution":
                    distribution_weights(args.model_name.replace("plot_mask_", ""), name, weights[name], masks[name], prs[name])
                elif func_name == "distribution2":
                    distribution_weights_2(args.model_name.replace("plot_mask_", ""), name, weights[name], masks[name], prs[name])
            # print(f"now plot heatmap on fc, shape: {weights['fc'].shape}") 
            # heatmap_weights(args.name, "fc", weights["fc"],masks["fc"])

        if "twolayers" in func_names:
            plot_two_layers(args.model_name.replace("plot_mask_", ""), "251-252", masks["layer2, block5, conv1"], masks["layer2, block5, conv2"])

        flop = {"fp":0.0, "binary":0.0, "fp forward": 0, "fp backward": 0, "binary forward": 0, "binary backward":0}
        for name in weights:
            for func_name in func_names:
                if func_name == "flop":
                    if "conv" in name:
                        calculate_flop(flop, masks[name], layer_type="conv")
                    else:
                        calculate_flop(flop, masks[name], layer_type="linear")
        if flop["fp"] != 0:
            print(flop)
            print(flop["binary"] / flop["fp"])
            print(flop["binary forward"] / flop["fp forward"])
            print(flop["binary backward"] / flop["fp backward"])


if __name__ == "__main__":
    # for debug

    SAVE_EXTENSION = "pdf"
    # SAVE_EXTENSION = "png"

    plot_mask(["distribution"])

    # plot_mask(["flop"])

    plot_mask(["distribution2"])

    plot_mask(["heatmap"])

    plot_mask(["twolayers"])