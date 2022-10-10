import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys
sys.path.insert(0, './')


def get_mask_info(args, model):
    if(isinstance(model, torch.nn.DataParallel)):
        model = model.module

    value_dict = {}
    for name, params in model.named_parameters():
        values = params.squeeze().cpu().detach().numpy()
        layer_name, value_type = name.rsplit(".", 1)
        if layer_name not in value_dict:
            value_dict[layer_name] = {}
        value_dict[layer_name][value_type] = values
        if name.endswith("scores"):
            th = np.sort(values, axis=None)[int((1-args.current_pr)*values.size)]
            value_dict[layer_name]["masks"] = np.where(values > th, 1, 0)
    return value_dict


def save_mask_info(args, model, output_dir, no_fig=True):
    value_dict = get_mask_info(args, model)

    for name in value_dict:
        if "masks" not in value_dict[name]:
            continue

        for value_type in value_dict[name]:
            save_file = output_dir / f"{name}.{value_type}"
            np.save(save_file, value_dict[name][value_type])

    if no_fig:
        print("no figure plotted. To enable plotting, please set 'no_fig=False' in function 'save_mask_info()'.")
        return

    for name in value_dict:
        if "masks" not in value_dict[name]:
            continue

        weights = value_dict[name]["weight"]
        masks = value_dict[name]["masks"]
        annotations = masks.astype(str)
        annotations[masks == 0] = ""
        annotations[masks == 1] = "O"
        print(f"now plot heatmap on {name}, shape: {weights.shape}") 
        heatmap_weights(name, weights, annotations, output_dir)
    # print("heatmap function is currently disabled, because it takes too long time.")


def heatmap_weights(name, weight, masks, output_dir, use_x=True, use_y=True, shade_empty=False):
    kernel_size = weight.shape[3]

    merged_weight = np.vstack((np.hstack(weight[i]) for i in list(range(weight.shape[0]))))
    merged_masks = np.vstack((np.hstack(masks[i]) for i in list(range(masks.shape[0]))))

    height = merged_masks.shape[0]
    width = merged_masks.shape[1]

    plot_width = max(10, int(25.0/384.0*width))
    plot_height = max(10, int(25.0/384.0*height))

    # print(plot_width, plot_height)

    fig = plt.figure(figsize=(plot_width, plot_height))
    grid_size_width = width
    grid_size_height = height
    bar_size = max(1,min(15, int(0.05*height), int(0.05*width)))
    gs = gridspec.GridSpec(grid_size_height, grid_size_width, wspace=10, hspace=10)

    y_split = bar_size if use_x else 0
    x_split = grid_size_width - bar_size if use_y else grid_size_width

    ax_main = plt.subplot(gs[y_split:, :x_split])
    # main plot
    ax_main.spy(merged_masks, markersize=4, markeredgewidth=0)

    if shade_empty:
        empty_mask = np.zeros_like(merged_masks)
        for col in range(0,merged_masks.shape[1],kernel_size):
            if np.sum(merged_masks[:,col:col+kernel_size]) == 0:
                empty_mask[:,col:col+kernel_size] = 1
        for row in range(0,merged_masks.shape[0],kernel_size):
            if np.sum(merged_masks[row:row+kernel_size,:]) == 0:
                empty_mask[row:row+kernel_size,:] = 1
        ax_main.spy(empty_mask, markersize=4, markeredgewidth=0, color="orange", alpha=0.05)

    ax_main.set_xticks(np.arange(-0.5,width,3))
    ax_main.set_yticks(np.arange(-0.5,height,3))
    ax_main.set_xticklabels([])
    ax_main.set_yticklabels([])
    ax_main.set_xlim((-0.5,width-0.5))
    ax_main.set_ylim((height-0.5, -0.5))
    # remove small lines of the ticks
    ax_main.xaxis.set_ticks_position('none')
    ax_main.yaxis.set_ticks_position('none')
    ax_main.set_aspect('auto')

    ax_main.grid()

    merged_masks_2 = np.sum(masks, axis=(2,3))

    rows_masks = np.sign(np.sum(merged_masks_2, axis=1))
    cols_masks = np.sign(np.sum(merged_masks_2, axis=0))

    rows_indices = np.nonzero(rows_masks)[0] * kernel_size + int((kernel_size-1)/2)
    cols_indices = np.nonzero(cols_masks)[0] * kernel_size + int((kernel_size-1)/2)

    verts = list(zip([-12.5,12.5,12.5,-12.5],[-60.,-60.,60.,60]))
    verts2 = list(zip([-60.,-60.,60.,60],[-12.5,12.5,12.5,-12.5]))

    if use_x:
        ax_xDist = plt.subplot(gs[:y_split, :x_split],sharex=ax_main)
        ax_xDist.scatter(cols_indices, np.zeros_like(cols_indices), c='blue', marker=verts, s=54**2, edgecolors='none')
        ax_xDist.set_yticks([])
        ax_xDist.set_yticklabels([])
        # ax_xDist.grid()
        ax_xDist.xaxis.set_ticks_position('none')
        ax_xDist.yaxis.set_ticks_position('none')
        ax_xDist.set_aspect('auto')
    if use_y:
        ax_yDist = plt.subplot(gs[y_split:, x_split:],sharey=ax_main)
        ax_yDist.scatter(np.zeros_like(rows_indices), rows_indices, c='blue', marker=verts2, s=54**2, edgecolors='none')
        ax_yDist.set_xticks([])
        ax_yDist.set_xticklabels([])
        # ax_yDist.grid()
        ax_yDist.xaxis.set_ticks_position('none')
        ax_yDist.yaxis.set_ticks_position('none')
        ax_yDist.set_aspect('auto')

    plt.style.use("classic")
    
    fig.savefig(f".{output_dir}/{name}{'-shade' if shade_empty else ''}.pdf", dpi=300, bbox_inches='tight')