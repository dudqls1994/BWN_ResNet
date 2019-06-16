import torch.nn as nn
import torch
import torch.nn.functional as F
import math

# init Binary conv --> BN - Conv - Active
class BinConv2d(nn.Module):
    def __init__(self, input_channels, output_channels,
                 kernel_size=-1, stride=1, padding=0, bias= False):
        super(BinConv2d, self).__init__()
        self.layer_type = 'BinConv2d'
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bn = nn.BatchNorm2d(input_channels,  affine=False)
        self.conv = nn.Conv2d(input_channels, output_channels,kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.bn(x)
        x = self.conv(x)
        x = self.relu(x)
        return x

# VGG8 like  --> 8Conv -> 1x1conv -> GAP
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.vgg = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            BinConv2d(input_channels=64, output_channels=128, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            BinConv2d(input_channels=128, output_channels=256, kernel_size=3, padding=1),
            BinConv2d(input_channels=256, output_channels=256, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            BinConv2d(input_channels=256, output_channels=512, kernel_size=3, padding=1),
            BinConv2d(input_channels=512, output_channels=512, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            BinConv2d(input_channels=512, output_channels=512, kernel_size=3, padding=1),
            BinConv2d(input_channels=512, output_channels=512, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.bn = nn.BatchNorm2d(512, affine=False)
        self.conv_last = nn.Conv2d(512, 3, kernel_size=1)
        self.bn_last = nn.BatchNorm2d(3)

    def forward(self, x):
        x = self.vgg(x)
        # x = x.view(x.size(0), 10)
        x = self.bn(x)
        x = F.relu(self.conv_last(x))
        x = self.bn_last(x)
        return F.avg_pool2d(x, kernel_size=x.shape[-2:]).view(x.shape[0], -1)
