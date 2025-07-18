from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class ShuffleNetV2(nn.Module):
    """ShuffleNetV2 implementation.
    """

    def __init__(self, scale=1.0, in_channels=3, c_tag=0.5, num_classes=1000, activation=nn.ReLU,
                 SE=False, residual=False, groups=2):
        """
        ShuffleNetV2 constructor
        :param scale:
        :param in_channels:
        :param c_tag:
        :param num_classes:
        :param activation:
        :param SE:
        :param residual:
        :param groups:
        """

        super(ShuffleNetV2, self).__init__()

        self.scale = scale
        self.c_tag = c_tag
        self.residual = residual
        self.SE = SE
        self.groups = groups

        self.activation_type = activation
        self.activation = activation(inplace=True)
        self.num_classes = num_classes

        self.num_of_channels = {0.5: [24, 48, 96, 192, 1024], 1: [24, 116, 232, 464, 1024],
                                1.5: [24, 176, 352, 704, 1024], 2: [24, 244, 488, 976, 2048]}
        self.c = [_make_divisible(chan, groups) for chan in self.num_of_channels[scale]]
        self.n = [3, 8, 3]  # TODO: should be [3,7,3]
        self.conv1 = nn.Conv2d(in_channels, self.c[0], kernel_size=3, bias=False, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(self.c[0])
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.shuffles = self._make_shuffles()

        self.conv_last = nn.Conv2d(self.c[-2], self.c[-1], kernel_size=1, bias=False)
        self.bn_last = nn.BatchNorm2d(self.c[-1])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.c[-1], self.num_classes)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def _make_stage(self, inplanes, outplanes, n, stage):
        modules = OrderedDict()
        stage_name = "ShuffleUnit{}".format(stage)

        # First module is the only one utilizing stride
        first_module = DownsampleUnit(inplanes=inplanes, activation=self.activation_type, c_tag=self.c_tag,
                                      groups=self.groups)
        modules["DownsampleUnit"] = first_module
        second_module = BasicUnit(inplanes=inplanes * 2, outplanes=outplanes, activation=self.activation_type,
                                  c_tag=self.c_tag, SE=self.SE, residual=self.residual, groups=self.groups)
        modules[stage_name + "_{}".format(0)] = second_module
        # add more LinearBottleneck depending on number of repeats
        for i in range(n - 1):
            name = stage_name + "_{}".format(i + 1)
            module = BasicUnit(inplanes=outplanes, outplanes=outplanes, activation=self.activation_type,
                               c_tag=self.c_tag, SE=self.SE, residual=self.residual, groups=self.groups)
            modules[name] = module

        return nn.Sequential(modules)

    def _make_shuffles(self):
        modules = OrderedDict()
        stage_name = "ShuffleConvs"

        for i in range(len(self.c) - 2):
            name = stage_name + "_{}".format(i)
            module = self._make_stage(inplanes=self.c[i], outplanes=self.c[i + 1], n=self.n[i], stage=i)
            modules[name] = module

        return nn.Sequential(modules)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.maxpool(x)

        x = self.shuffles(x)
        x = self.conv_last(x)
        x = self.bn_last(x)
        x = self.activation(x)

        # average pooling layer
        x = self.avgpool(x)

        # flatten for input to fully-connected layer
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

