import os
import sys
import json
import argparse
sys.path.insert(0, './')

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np


def extract_points(curve_dict, acc=False):
    if acc:
        return sorted([[int(k),float(v)/100.0] for k,v in curve_dict.items()], key=lambda x: (x[0]))
    else:
        return sorted([[int(k),float(v)] for k,v in curve_dict.items()], key=lambda x: (x[0]))


def plot_lr_curve_single(logs, fig_name):
    early_stop = find_early_stop(logs["valid_loss"])

    # handle interrupt cases where logs have different epochs
    keys = ["train_acc", "test_acc", "valid_acc", "train_loss", "test_loss", "valid_loss"]
    lengths = [len(logs[key]) for key in keys]
    for k, l in zip(keys, lengths):
        print(f"{k} has points: {l}")
    max_idx = np.min(lengths)
    print(f"lr curve plot maximum epoch: {max_idx-1}")

    curve_groups = [
        {
            "ylabel": "Accuracy",
            "curves": [
                [extract_points(logs["train_acc"], acc=True)[:max_idx],"training accuracy", early_stop],
                [extract_points(logs["test_acc"], acc=True)[:max_idx],"testing accuracy", early_stop],
                [extract_points(logs["valid_acc"], acc=True)[:max_idx],"validation accuracy", early_stop]
            ]
        },
        {
            "ylabel": "Loss",
            "loc": 1,
            "curves": [
                [extract_points(logs["train_loss"])[:max_idx],"training loss", early_stop],
                [extract_points(logs["test_loss"])[:max_idx],"testing loss", early_stop],
                [extract_points(logs["valid_loss"])[:max_idx],"validating loss", early_stop]
            ],
            "is_loss": True
        },

    ]

    plot_lr_curve(curve_groups, fig_name, align_axis=False)


def plot_lr_curve(lr_curve_group, fig_name, align_axis=True):
    num_subfigures = len(lr_curve_group)

    fig, axs = plt.subplots(1, num_subfigures, figsize=(5*num_subfigures,5))
    figs = []
    axes = []
    for _ in range(num_subfigures):
        fig_tmp, ax_tmp = plt.subplots(1, 1, figsize=(5,5))
        figs.append(fig_tmp)
        axes.append(ax_tmp)

    if num_subfigures == 1:
        axs = [axs]

    colors = ['b', 'g', 'c', 'm', 'k', 'r']
    lines = ["-", "--", "-.", ":"]

    min_bound = np.inf
    max_bound = -np.inf

    for curves in lr_curve_group:
        ys = []
        for curve_settings in curves["curves"]:
            ys += [p[1] for p in curve_settings[0]]
        min_bound = min(min_bound, np.min(ys))
        max_bound = max(max_bound, np.max(ys))

    for count, curves in enumerate(lr_curve_group):
        y_label = curves.get("ylabel", "")
        legend_loc = curves.get("loc", 2)
        is_loss = curves.get("is_loss", False)
        for count2, [curve, name, early_stop] in enumerate(curves["curves"]):
            axs[count].plot([p[0] for p in curve], [p[1] for p in curve], color=colors[count2], linestyle=lines[count2], label=name)
            axes[count].plot([p[0] for p in curve], [p[1] for p in curve], color=colors[count2], linestyle=lines[count2], label=name)
            if early_stop:
                axs[count].scatter(curve[early_stop][0], curve[early_stop][1], c="red", s=10)
                axes[count].scatter(curve[early_stop][0], curve[early_stop][1], c="red", s=10)
        if align_axis:
            axs[count].set_ylim(0.99*min_bound, 1.01*max_bound)
            axes[count].set_ylim(0.99*min_bound, 1.01*max_bound)
        else:
            if is_loss:
                axs[count].set_ylim(0.99 * np.min([[p[1] for p in curve] for curve,_,_ in curves["curves"]]), 2 * np.percentile([[p[1] for p in curve] for curve,_,_ in curves["curves"]], 95))
                axes[count].set_ylim(0.99 * np.min([[p[1] for p in curve] for curve,_,_ in curves["curves"]]), 2 * np.percentile([[p[1] for p in curve] for curve,_,_ in curves["curves"]], 95))
        axs[count].yaxis.set_minor_locator(AutoMinorLocator())
        axs[count].set_xlabel("Iterations", fontsize=14)
        axes[count].yaxis.set_minor_locator(AutoMinorLocator())
        axes[count].set_xlabel("Iterations", fontsize=14)
        # if "Accuracy" in y_label:
        #     axs[count].set_ylim(0, 0.6)
        #     axes[count].set_ylim(0, 0.6)
        if y_label != "":
            axs[count].set_ylabel(y_label, fontsize=14)
            axes[count].set_ylabel(y_label, fontsize=14)
        axs[count].legend(loc=legend_loc, prop={'size': 14})
        axes[count].legend(loc=legend_loc, prop={'size': 14})

    fig.savefig(fig_name, dpi=600)
    fig_name_prefix, fig_ext = os.path.splitext(fig_name)
    for idx, figure in enumerate(figs):
        figure.savefig(f"{fig_name_prefix}-idx{idx}{fig_ext}", dpi=600, bbox_inches='tight')

def plot_lr_curve_with_std(lr_curve_group, fig_name, align_axis=True):
    num_subfigures = len(lr_curve_group)

    fig, axs = plt.subplots(1, num_subfigures, figsize=(5*num_subfigures,5))

    if num_subfigures == 1:
        axs = [axs]

    colors = ['b', 'g', 'c', 'm', 'k', 'r']
    lines = ["-", "--", "-.", ":"]

    min_bound = np.inf
    max_bound = -np.inf

    for curves in lr_curve_group:
        for curve_settings in curves["curves"]:
            x = [item[0] for item in curve_settings[0][0]]
            y = []
            for sub_sequence in curve_settings[0]:
                temp_y = [item[1] for item in sub_sequence]
                y.append(temp_y)

            curve_matrix = np.array(y)
            assert len(curve_matrix.shape) > 1
            mean = np.mean(curve_matrix, axis=0)
            std = np.std(curve_matrix, axis=0)
            curve_settings.append(x)
            curve_settings.append(mean.tolist())
            curve_settings.append((mean+std).tolist())
            curve_settings.append((mean-std).tolist())
            min_bound = min(min_bound, np.min(curve_settings[-1]))
            max_bound = max(max_bound, np.max(curve_settings[-2]))

    for count, curves in enumerate(lr_curve_group):
        y_label = curves.get("ylabel", "")
        legend_loc = curves.get("loc", 4)
        for count2, [curve, name, early_stop, x, mean, upper, lower] in enumerate(curves["curves"]):
            axs[count].plot(x, mean, color=colors[count2], linestyle=lines[count2], label=name)
            axs[count].fill_between(range(len(x)), lower, upper,alpha=.1)
            if early_stop:
                axs[count].scatter(mean[early_stop][0], mean[early_stop][1], c="red", s=10)
        if align_axis:
            axs[count].set_ylim(0.99*min_bound, 1.01*max_bound)
        axs[count].yaxis.set_minor_locator(AutoMinorLocator())
        axs[count].set_xlabel("Iterations", fontsize=14)
        if y_label != "":
            axs[count].set_ylabel(y_label, fontsize=14)
        axs[count].legend(loc=legend_loc, prop={'size': 14})

    plt.savefig(fig_name, dpi=600)


def load_json(json_file):
    with open(json_file, "r") as f_in:
        logs = json.load(f_in)
        return logs 


def find_early_stop(curve):
    min_loss = np.inf
    min_idx = -1
    for ite, loss in extract_points(curve):
        if min_loss > loss:
            min_loss = loss
            min_idx = ite
    return min_idx


def smooth_best(points, validation, descend=False):
    best_valid_y = np.inf
    smooth_y = 0
    smoothed_points = []
    for [valid_x, valid_y], [x, y] in zip(validation, points):
        if not descend and valid_y < best_valid_y:
            best_valid_y = valid_y
            smooth_y = y
        elif descend and valid_y < best_valid_y:
            best_valid_y = valid_y
            smooth_y = y
        smoothed_points.append([x, smooth_y])

    return smoothed_points

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot function")

    parser.add_argument("--logs", type=str, help="location of logs.json for plotting")

    args = parser.parse_args()

    with open(args.logs, "rb") as f_in:
        logs = json.load(f_in)

        lr_curve_dir = os.path.dirname(args.logs)
        fig_file = os.path.join(lr_curve_dir, "lr_curve.pdf")
        plot_lr_curve_single(logs, fig_file)
