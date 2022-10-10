import torch.nn as nn

from arch.builder import get_builder
import args as Args

# BasicBlock {{{
class BasicBlock(nn.Module):
    M = 2
    expansion = 1

    def __init__(self, builder, inplanes, planes, stride=1, downsample=None, base_width=64):
        super(BasicBlock, self).__init__()
        if base_width / 64 > 1:
            raise ValueError("Base width >64 does not work for BasicBlock")

        self.conv1 = builder.conv3x3(inplanes, planes, stride)
        self.bn1 = builder.batchnorm(planes)
        self.conv2 = builder.conv3x3(planes, planes)
        self.bn2 = builder.batchnorm(planes, last_bn=True)
        self.downsample = downsample
        self.stride = stride

        self.builder = builder

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        if self.bn1 is not None:
            out = self.bn1(out)

        out = self.builder.activation(out)

        out = self.conv2(out)

        if self.bn2 is not None:
            out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.builder.activation(out)

        return out


# BasicBlock }}}

# Bottleneck {{{
class Bottleneck(nn.Module):
    M = 3
    expansion = 4

    def __init__(self, builder, inplanes, planes, stride=1, downsample=None, base_width=64):
        super(Bottleneck, self).__init__()
        width = int(planes * base_width / 64)
        self.conv1 = builder.conv1x1(inplanes, width)
        self.bn1 = builder.batchnorm(width)
        self.conv2 = builder.conv3x3(width, width, stride=stride)
        self.bn2 = builder.batchnorm(width)
        self.conv3 = builder.conv1x1(width, planes * self.expansion)
        self.bn3 = builder.batchnorm(planes * self.expansion, last_bn=True)
        self.downsample = downsample
        self.stride = stride

        self.builder = builder

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.builder.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.builder.activation(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        out = self.builder.activation(out)

        return out


# Bottleneck }}}

# ResNet {{{
class ResNet(nn.Module):
    def __init__(self, builder, block, planes, layers, num_classes=1000, base_width=64):
        self.inplanes = planes[0]
        super(ResNet, self).__init__()

        self.base_width = base_width
        if self.base_width // planes[0] > 1:
            print(f"==> Using {self.base_width // planes[0]}x wide model")

        if Args.args.first_layer_dense:
            self.conv1 = nn.Conv2d(
                3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
            )
        else:
            self.conv1 = builder.conv7x7(3, self.inplanes, stride=2, first_layer=True)

        self.bn1 = builder.batchnorm(self.inplanes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(builder, block, planes[0], layers[0])
        self.layer2 = self._make_layer(builder, block, planes[1], layers[1], stride=2)
        self.layer3 = self._make_layer(builder, block, planes[2], layers[2], stride=2)
        self.layer4 = self._make_layer(builder, block, planes[3], layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.builder = builder

        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        if Args.args.last_layer_dense:
            self.fc = nn.Conv2d(planes[3] * block.expansion, Args.args.num_classes, 1)
        else:
            self.fc = builder.conv1x1(planes[3] * block.expansion, num_classes)

        if Args.args.end_with_bn:
            self.last = builder.batchnorm(num_classes)

    def _make_layer(self, builder, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            dconv = builder.conv1x1(
                self.inplanes, planes * block.expansion, stride=stride
            )
            dbn = builder.batchnorm(planes * block.expansion)
            if dbn is not None:
                downsample = nn.Sequential(dconv, dbn)
            else:
                downsample = dconv

        layers = []
        layers.append(block(builder, self.inplanes, planes, stride, downsample, base_width=self.base_width))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(builder, self.inplanes, planes, base_width=self.base_width))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        if self.bn1 is not None:
            x = self.bn1(x)
        x = self.builder.activation(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.fc(x)

        if Args.args.end_with_bn:
            x = self.last(x)

        x = x.view(x.size(0), -1)

        return x


# ResNet }}}
def ResNet18_001_p01_imagenet100(pretrained=False):
    return ResNet(get_builder(), BasicBlock, [23, 25, 27, 29], [2, 2, 2, 2], 100)

def ResNet34_001_p01_imagenet100():
    return ResNet(get_builder(), BasicBlock, [23, 25, 27, 29], [3, 4, 6, 3], 100)

def ResNet50_001_p01_imagenet100(pretrained=False):
    return ResNet(get_builder(), Bottleneck, [23, 25, 27, 31], [3, 4, 6, 3], 100)

