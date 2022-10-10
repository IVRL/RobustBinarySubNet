import torch
import torch.nn as nn
import torch.nn.functional as F

from arch.builder import get_builder
import args as Args

class BasicBlockCustom(nn.Module):
    expansion = 1

    def __init__(self, builder, in_planes, planes, stride=1):
        super(BasicBlockCustom, self).__init__()
        self.conv1 = builder.conv3x3(in_planes, planes, stride=stride)
        self.bn1 = builder.batchnorm(planes)
        self.conv2 = builder.conv3x3(planes, planes, stride=1)
        self.bn2 = builder.batchnorm(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                builder.conv1x1(in_planes, self.expansion * planes, stride=stride),
                builder.batchnorm(self.expansion * planes),
            )

        self.builder = builder

    def forward(self, x):
        # out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.builder.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        # out = F.relu(out, inplace=True)
        out = self.builder.activation(out)
        return out

class BottleneckCustom(nn.Module):
    expansion = 4

    def __init__(self, builder, in_planes, planes, stride=1):
        super(BottleneckCustom, self).__init__()
        self.conv1 = builder.conv1x1(in_planes, planes)
        self.bn1 = builder.batchnorm(planes)
        self.conv2 = builder.conv3x3(planes, planes, stride=stride)
        self.bn2 = builder.batchnorm(planes)
        self.conv3 = builder.conv1x1(planes, self.expansion * planes)
        self.bn3 = builder.batchnorm(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                builder.conv1x1(in_planes, self.expansion * planes, stride=stride),
                builder.batchnorm(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out, inplace=True)

        return out

class ResNetCustom(nn.Module):
    def __init__(self, builder, block, planes, num_blocks, num_classes=10):
        super(ResNetCustom, self).__init__()
        self.in_planes = planes[0]
        self.builder = builder


        self.conv1 = builder.conv3x3(3, self.in_planes, stride=1, first_layer=True)
        self.bn1 = builder.batchnorm(self.in_planes)
        self.layer1 = self._make_layer(block, planes[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, planes[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, planes[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, planes[3], num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc = builder.conv1x1(planes[3] * block.expansion, num_classes, last_layer=True)
        if Args.args.end_with_bn:
            self.last = builder.batchnorm(num_classes)

    def _make_layer(self, block, plane, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.builder, self.in_planes, plane, stride))
            self.in_planes = plane * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        # out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.builder.activation(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = self.fc(out)
        if Args.args.end_with_bn:
            out = self.last(out)
        return out.flatten(1)

def ResNet34_001_p1():
    return ResNetCustom(get_builder(), BasicBlockCustom, [6, 13, 26, 51], [3, 4, 6, 3])

def ResNet34_001_p05():
    return ResNetCustom(get_builder(), BasicBlockCustom, [14, 20, 29, 40], [3, 4, 6, 3])

def ResNet34_001_p01():
    return ResNetCustom(get_builder(), BasicBlockCustom, [23, 25, 27, 29], [3, 4, 6, 3])

def ResNet34_001_p0():
    return ResNetCustom(get_builder(), BasicBlockCustom, [26, 26, 26, 26], [3, 4, 6, 3])

def ResNet34_001_p1_cifar100():
    return ResNetCustom(get_builder(), BasicBlockCustom, [6, 13, 26, 51], [3, 4, 6, 3], num_classes=100)

def ResNet34_001_p05_cifar100():
    return ResNetCustom(get_builder(), BasicBlockCustom, [14, 20, 29, 40], [3, 4, 6, 3], num_classes=100)

def ResNet34_001_p01_cifar100():
    return ResNetCustom(get_builder(), BasicBlockCustom, [23, 25, 27, 29], [3, 4, 6, 3], num_classes=100)

def ResNet34_001_p0_cifar100():
    return ResNetCustom(get_builder(), BasicBlockCustom, [26, 26, 26, 26], [3, 4, 6, 3], num_classes=100)

def ResNet18_001_p01():
    return ResNetCustom(get_builder(), BasicBlockCustom, [23, 25, 27, 29], [2, 2, 2, 2])

def ResNet18_001_p1():
    return ResNetCustom(get_builder(), BasicBlockCustom, [6, 13, 26, 51], [2, 2, 2, 2])

def ResNet18_001_p01_cifar100():
    return ResNetCustom(get_builder(), BasicBlockCustom, [23, 25, 27, 29], [2, 2, 2, 2], num_classes=100)

def ResNet18_001_p1_cifar100():
    return ResNetCustom(get_builder(), BasicBlockCustom, [6, 13, 26, 51], [2, 2, 2, 2], num_classes=100)

def ResNet50_001_p01():
    return ResNetCustom(get_builder(), BottleneckCustom, [23, 25, 27, 31], [3, 4, 6, 3])

def ResNet50_001_p1():
    return ResNetCustom(get_builder(), BottleneckCustom, [7, 13, 26, 51], [3, 4, 6, 3])

def ResNet50_001_p01_cifar100():
    return ResNetCustom(get_builder(), BottleneckCustom, [23, 25, 27, 31], [3, 4, 6, 3], num_classes=100)

def ResNet50_001_p1_cifar100():
    return ResNetCustom(get_builder(), BottleneckCustom, [7, 13, 26, 51], [3, 4, 6, 3], num_classes=100)