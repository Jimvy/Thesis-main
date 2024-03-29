'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:

name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.

@misc{Idelbayev18a,
    author       = "Yerlan Idelbayev",
    title        = "Proper {ResNet} Implementation for {CIFAR10/CIFAR100} in {PyTorch}",
    howpublished = "\\url{https://github.com/akamaster/pytorch_resnet_cifar10}",
    note         = "Accessed: 20xx-xx-xx"
}
'''


import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


__all__ = [
    'ResNet',
    'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202',
    'resnet18', 'resnet34', 'resnet50', 'resnet101',
]


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, nonlinearity='relu')


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(
                        x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4),
                        "constant", 0)
                )
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1,
                               stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

    def depth():
        return 2


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, option='B'):
        super(BottleNeck, self).__init__()
        self._in_planes = in_planes
        self._planes = planes
        self._stride = stride
        self._option = option

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1,
                               stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            if option == 'A':
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(x[:, :, ::2, ::2],
                                    (0, 0, 0, 0, planes//4, planes//4),
                                    "constant", 0)
                )
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1,
                              stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion*planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

    def depth():
        return 3

    """
    def __str__(self):
        stride_s = f"/{self._stride}" if self._stride != 1 else ""
        if self._in_planes == self._planes:
            return f"BottleNeck-{self._option}-{self._in_planes}{stride_s}"
        else:
            return f"BottleNeck-{self._option}-{self._in_planes}/{self._planes}{stride_s}"
    """


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, base_width=16):
        super(ResNet, self).__init__()
        self._base_width = base_width
        self._block = block
        self._num_classes = num_classes
        self._num_blocks = num_blocks

        self.in_planes = base_width
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, base_width*1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, base_width*2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, base_width*4, num_blocks[2], stride=2)
        if len(num_blocks) > 3:
            self.layer4 = self._make_layer(block, base_width*8, num_blocks[3], stride=2)
            out_planes = base_width*8*block.expansion
        else:
            self.layer4 = nn.Sequential()
            out_planes = base_width*4*block.expansion
        self.linear = nn.Linear(out_planes, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        if self.layer4:
            out = self.layer4(out)
        #out = F.avg_pool2d(out, out.size()[3])
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def get_model_name(self):
        num_layers = sum(self._num_blocks) * self._block.depth() + 2
        return f"ResNet{num_layers}-{self._base_width}"


def resnet20(**kwargs):
    r"""Should reach 91.73% accuracy, 0.27M params"""
    return ResNet(BasicBlock, [3, 3, 3], **kwargs)


def resnet32(**kwargs):
    r"""Should reach 92.63% accuracy, 0.46M params"""
    return ResNet(BasicBlock, [5, 5, 5], **kwargs)


def resnet44(**kwargs):
    r"""Should reach 93.10% accuracy, 0.66M params"""
    return ResNet(BasicBlock, [7, 7, 7], **kwargs)


def resnet56(**kwargs):
    r"""Should reach 93.39% accuracy, 0.85M params"""
    return ResNet(BasicBlock, [9, 9, 9], **kwargs)


def resnet110(**kwargs):
    r"""Should reach 93.68% accuracy, 1.7M params"""
    return ResNet(BasicBlock, [18, 18, 18], **kwargs)


def resnet1202(**kwargs):
    r"""Should reach 93.82% accuracy, 19.4M params"""
    return ResNet(BasicBlock, [200, 200, 200], **kwargs)


# ResNets for ImageNet (but also "compatible" with CIFAR)


def resnet18(base_width=64, **kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], base_width=base_width, **kwargs)


def resnet34(base_width=64, **kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], base_width=base_width, **kwargs)


def resnet50(base_width=64, **kwargs):
    return ResNet(BottleNeck, [3, 4, 6, 3], base_width=base_width, **kwargs)


def resnet101(base_width=64, **kwargs):
    return ResNet(BottleNeck, [3, 4, 23, 3], base_width=base_width, **kwargs)


def resnet152(base_width=64, **kwargs):
    return ResNet(BottleNeck, [3, 8, 36, 3], base_width=base_width, **kwargs)


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()
