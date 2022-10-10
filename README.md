# Robust-binary-random-networks
Official implementation of the NeurIPS 2022 accepted paper "Robust Binary Models by Pruning Randomly-initialized Networks"

Authors: Chen Liu*, Ziqi Zhao*, Sabine Süsstrunk, Mathieu Salzmann

*: equal contributions

[Paper](https://openreview.net/pdf?id=5g-h_DILemH), [OpenReview](https://openreview.net/forum?id=5g-h_DILemH)

## Overview
Robustness to adversarial attacks was shown to require a larger model capacity, and thus a larger memory footprint. In this paper, we introduce an approach to obtain robust yet compact models by pruning randomly-initialized binary networks.
Unlike adversarial training, which learns the model parameters, we initialize the model parameters as either $+1$ or $-1$, keep them fixed, and find a subnetwork structure that is robust to attacks.
Our method confirms the *Strong Lottery Ticket Hypothesis* in the presence of adversarial attacks, and extends this to binary networks.
Furthermore, it yields more compact networks with competitive performance than existing works by 1) adaptively pruning different network layers; 2) exploiting an effective binary initialization scheme; 3) incorporating a last batch normalization layer to improve training stability.
Our experiments demonstrate that our approach not only always outperforms the state-of-the-art robust binary networks, but also can achieve accuracy better than full-precision ones on some datasets.
Finally, we show the structured patterns of our pruned binary networks.

## Getting Started
To run the code in this repository, be sure to install the following packages:
```
pytorch
torchvision
numpy
pandas
matplotlib
tensorboard
tqdm
pyyaml
```

### A quick start

To prune a ResNet34 model using CIFAR10 with pruning rate $r=0.99$, $p=0.1$ and $a=0.01$, you can simply run the following command:

```bash
cd code;
python run/train.py \
    --config configs/advprune/cifar10/resnet34/resnet34-binary-weights.yaml \
    --model_name rn34-cifar10-quick-start \
    --log-dir /path/to/store/the/output \
    --multigpu 0,1 \
    --prune-rate 0.01 \
    --pr-scale 0.1 \
    --epochs 400 \
    --end-with-bn \
    --score-init-scale 0.01 \
    --fan-scaled-score-mode none;
```

Please note that ```--prune-rate``` in the code has the opposite meaning as in the paper, i.e. here it means the ratio of retained parameters. ```--pr-scale``` refers to $p$ in the paper. ```--score-init-scale``` refers to $a$ in the paper. If you want to track the significant loss changes during training, please add ```--debug``` to your command.

The experiment output will be stored under ```--log-dir/config_filename/model_name/prune_rate=pr/```, for example ```/path/to/store/the/output/rn34-cifar10-quick-start/prune_rate=0.01/```. If you run the code multiple times with the same config and same model name, each trial will be stored in a subfolder with the trail id (starting from 0).

To use AutoAttack for evaluation, please run the following command (this example is consistent with the one above):

```bash
cd code;
python run/aa_standard.py \
    --config configs/advprune/cifar10/resnet34/aa-standard.yaml \
    --model_name rn34-cifar10-quick-start \
    --model_type cResNet34 \
    --multigpu 0,1 \
    --pretrained /path/to/store/the/output/resnet34-binary-weights/rn34-cifar10-quick-start/prune_rate=0.01/0/checkpoints/model_best.pth \
    --prune-rate 0.01 \
    --pr-scale 0.1;
```

This command will generate a separate folder ```code/runs/cifar10-resnet34/aa/aa-standard/rn34-cifar10-quick-start/prune_rate=0.01/0/``` to store the AA results, and the result file is ```aa-log.txt```.

We also provide a set of example scripts in ```bash_scripts``` so that you don't need to type in the command every time you run the code. You can also use them to create your own scripts. Scripts starting with "baseline" are simply adversarial training for unpruned models, while scripts starting with "best_settings" are the ones we use to produce pruned subnetworks. The script ```bash_scripts/plot_mask.sh``` is used to plot the figures we use in the paper. Enjoy!

### Config files

Some predefined configs are stored in code/configs in the yaml format. You can load them using ```--config <path/to/config>```. If you specify an argument in the command but this argument is already in the config file, the argument value in the config file will be supressed and the code will run with the value in the command.

### Setting up ImageNet100

We use CIFAR10, CIFAR100 and ImageNet100 in our experiments. CIFAR10 and CIFAR100 can be directly downloaded to ```code/data``` by torchvision when you try to load them. However, to use ImageNet100, you have to manually download, extract and move it to ```code/data```. We give the instructions below to help you set up ImageNet100.

ImageNet100 is a subset of ImageNet, and to use ImageNet100, you have to first download ImageNet from [Kaggle](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data) or [Official website](https://www.image-net.org/). 

After downloading and extracting the ImageNet dataset, you should find two subfolders in its data folder, ```train``` and ```val``` containing all the images. Images in both of them should be organized in subfolders with its class id. If you find that images in the ```val``` folder are not organized in this way, please use [this bash script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh) to make them organized.

The 100 classes we use for ImageNet100 follows the list from a python package called [continuum](https://github.com/Continvvm/continuum/blob/838ad2ba3571f1563627301c30152c0f07d3cffa/continuum/datasets/imagenet.py#L44). We provide ```code/run/format_imagenet100.py``` to help you create the ImageNet100 dataset. Depending on the speed of your server, this operation may takes a few hours.

## Model Zoo

Since the checkpoints are too large to store, we currently only release the ones from our method in Table4 and Table5. Here are the list of checkpoints:

| Model | Dataset | Fast train? | Clean Accuracy (\%)| Robust Accuracy (\%) | Download link | 
|-------|---------|-------------|--|----------------------|---------------|
|ResNet34|CIFAR10|N|77.30|45.06|[download](https://onedrive.live.com/download?cid=BDAB630C72C069ED&resid=BDAB630C72C069ED%2148431&authkey=AMvdxFxlNZ38r8A)|
|ResNet34|CIFAR10|Y|81.63|40.77|[download](https://onedrive.live.com/download?cid=BDAB630C72C069ED&resid=BDAB630C72C069ED%2148430&authkey=ABS9t5egBVCZg-k)|
|ResNet34|CIFAR100|N|60.16|34.83|[download](https://onedrive.live.com/download?cid=BDAB630C72C069ED&resid=BDAB630C72C069ED%2148436&authkey=AAXpscq7w3Y9hm8)|
|ResNet34|CIFAR100|Y|63.73|34.45|[download](https://onedrive.live.com/download?cid=BDAB630C72C069ED&resid=BDAB630C72C069ED%2148437&authkey=APqIiI253plVeyA)|
|ResNet34|ImageNet100|Y|58.94|33.04|[download](https://onedrive.live.com/download?cid=BDAB630C72C069ED&resid=BDAB630C72C069ED%2148435&authkey=ANTglfOeNw9Vw6w)|
|ResNet18|CIFAR10|N|72.35|39.65|[download](https://onedrive.live.com/download?cid=BDAB630C72C069ED&resid=BDAB630C72C069ED%2148429&authkey=AJjcx2hotq58YIk)|
|ResNet18|CIFAR10|Y|66.12|30.86|[download](https://onedrive.live.com/download?cid=BDAB630C72C069ED&resid=BDAB630C72C069ED%2148434&authkey=AA7CkflOXFeQpkI)|
|ResNet50|CIFAR10|N|76.66|42.72|[download](https://onedrive.live.com/download?cid=BDAB630C72C069ED&resid=BDAB630C72C069ED%2148433&authkey=AKQvhbgBuZ-mgcE)|
|ResNet50|CIFAR10|Y|78.24|37.93|[download](https://onedrive.live.com/download?cid=BDAB630C72C069ED&resid=BDAB630C72C069ED%2148432&authkey=AG5bIYv9NA-qIeo)|

You can either click the link to download it or directly use wget to download. If you choose to use wget, don't forget to specify the download filename using ```-O filename```. For example, the first checkpoint in the table can be downloaded by:

```bash
wget "https://onedrive.live.com/download?cid=BDAB630C72C069ED&resid=BDAB630C72C069ED%2148431&authkey=AMvdxFxlNZ38r8A" -O rn34_cifar10_n.pth
```

To evaluate these checkpoints using AutoAttack, please refer to the script ```evaluate_pretrained_models_examples.sh``` in ```bash_scripts```.

Note that you should get exactly the same clean accuracy as shown in the table above if your script is correct. However, the robust accuracy might be slighly different from what we report.

## Acknowledgement
Some of the codes are adapted from [`What's hidden in a randomly weighted neural network?`](https://github.com/allenai/hidden-networks) and [`AutoAttack`](https://github.com/fra31/auto-attack).

## Bibliography

If you find this work useful, please consider citing it.
```
@inproceedings{
liu2022robust,
title={Robust Binary Models by Pruning Randomly-initialized Networks},
author={Chen Liu and Ziqi Zhao and Sabine Süsstrunk and Mathieu Salzmann},
booktitle={Advances in Neural Information Processing Systems},
year={2022},
url={https://openreview.net/forum?id=5g-h_DILemH}
}
```

## License
This repository is released under the [MIT license](LICENSE).