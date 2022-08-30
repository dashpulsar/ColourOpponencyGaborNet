import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.io import read_image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from torchsummary import summary
import numpy as np
import math
import os
import psutil
print(torch.__version__)
print(torch.cuda.get_device_name(0))

class BasicBlock(nn.Module):
    """
    For shallow networks, such as ResNet-18/34, etc., use the basic Block
    The base module is not compressed, so expansion=1
    """
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock,self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels,out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        # If the input and output dimensions are not equal, use a 1x1 convolutional layer to change the dimension
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels),
            )
    def forward(self, x):
        out = self.features(x)
#         print(out.shape)
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class Bottleneck(nn.Module):
    """
    For deep networks, we use BottleNeck, which is proposed in the paper to have approximate computational complexity but save a lot of resources
    zip_channels: The compressed dimension, the final output dimension is expansion * zip_channels
    For the network structure of ResNet50/101/152, the expansion=4 is mainly because the third layer is 4 times that of the second layer.
    """
    expansion = 4

    def __init__(self, in_channels, zip_channels, stride=1):
        super(Bottleneck, self).__init__()
        out_channels = self.expansion * zip_channels
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, zip_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(zip_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(zip_channels, zip_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(zip_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(zip_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.features(x)
        #         print(out.shape)
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class GaborConv2d(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
            padding_mode="zeros",
    ):
        super().__init__()

        self.is_calculated = False

        self.conv_layer = nn.modules.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        self.kernel_size = self.conv_layer.kernel_size

        # small addition to avoid division by zero
        self.delta = 1e-3

        # freq, theta, sigma are set up according to S. Meshgini,
        # A. Aghagolzadeh and H. Seyedarabi, "Face recognition using
        # Gabor filter bank, kernel principal component analysis
        # and support vector machine"
        self.freq = nn.Parameter(
            (math.pi / 2) * math.sqrt(2) ** (-torch.randint(0, 5, (out_channels, in_channels))).type(torch.Tensor),
            requires_grad=True, )
        self.theta = nn.Parameter((math.pi / 8) * torch.randint(0, 8, (out_channels, in_channels)).type(torch.Tensor),
                                  requires_grad=True, )
        self.sigma = nn.Parameter(math.pi / self.freq, requires_grad=True)
        self.psi = nn.Parameter(math.pi * torch.rand(out_channels, in_channels), requires_grad=True)
        self.x0 = nn.Parameter(torch.ceil(torch.Tensor([self.kernel_size[0] / 2]))[0], requires_grad=False)
        self.y0 = nn.Parameter(torch.ceil(torch.Tensor([self.kernel_size[1] / 2]))[0], requires_grad=False)
        self.y, self.x = torch.meshgrid(
            [torch.linspace(-self.x0 + 1, self.x0 + 0, self.kernel_size[0]),
             torch.linspace(-self.y0 + 1, self.y0 + 0, self.kernel_size[1]), ])
        self.y = nn.Parameter(self.y)
        self.x = nn.Parameter(self.x)
        self.weight = nn.Parameter(torch.empty(self.conv_layer.weight.shape, requires_grad=True), requires_grad=True, )

        self.register_parameter("freq", self.freq)
        self.register_parameter("theta", self.theta)
        self.register_parameter("sigma", self.sigma)
        self.register_parameter("psi", self.psi)
        self.register_parameter("x_shape", self.x0)
        self.register_parameter("y_shape", self.y0)
        self.register_parameter("y_grid", self.y)
        self.register_parameter("x_grid", self.x)
        self.register_parameter("weight", self.weight)

    def forward(self, input_tensor):
        if self.training:
            self.calculate_weights()
            # print(self.conv_layer.weight.data.shape)
            self.is_calculated = False
        if not self.training:
            if not self.is_calculated:
                self.calculate_weights()
                self.is_calculated = True
        return self.conv_layer(input_tensor)

    def calculate_weights(self):
        for i in range(self.conv_layer.out_channels):
            for j in range(self.conv_layer.in_channels):
                sigma = self.sigma[i, j].expand_as(self.y)
                freq = self.freq[i, j].expand_as(self.y)
                theta = self.theta[i, j].expand_as(self.y)
                psi = self.psi[i, j].expand_as(self.y)

                rotx = self.x * torch.cos(theta) + self.y * torch.sin(theta)
                roty = -self.x * torch.sin(theta) + self.y * torch.cos(theta)

                g = torch.exp(-0.5 * ((rotx ** 2 + roty ** 2) / (sigma + self.delta) ** 2))
                g = g * torch.cos(freq * rotx + psi)
                g = g / (2 * math.pi * sigma ** 2)
                self.conv_layer.weight.data[i, j] = g


class GaborConvFixed(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 input_dict,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=False,
                 padding_mode="zeros",
                 ):
        super().__init__()
        self.input_dict = input_dict
        if self.input_dict == None:
            self.input_dict = {  # 'ksize': (127, 127),
                'ksize': (31, 31),
                'gammas': [0.5],
                #           'bs': np.linspace(0.4, 2.6, num=3),  # 0.4, 1, 1.8, 2.6
                #           'bs': np.linspace(0.4, 2.6, num=5),
                'bs': np.linspace(1, 2.6, num=3).tolist(),
                #           'bs': np.linspace(1, 2.6, num=5),
                #           'sigmas': [4, 8, 16],  # , 32
                'sigmas': [8],
                'thetas': np.linspace(0, np.pi, 4, endpoint=False).tolist(),
                'psis': [np.pi / 2, 3 * np.pi / 2]}

        self.ksize = self.input_dict["ksize"]
        self.sigmas = self.input_dict["sigmas"]
        self.bs = self.input_dict["bs"]
        self.gammas = self.input_dict["gammas"]
        self.thetas = self.input_dict["thetas"]
        self.psis = self.input_dict["psis"]

        self.conv_layer = nn.modules.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )

        self.weight = nn.Parameter(torch.empty(self.conv_layer.weight.shape, requires_grad=False),
                                   requires_grad=False, )

    def calc_lambda(self, sigma, bandwidth):
        p = 2 ** bandwidth
        c = np.sqrt(np.log(2) / 2)
        return sigma * np.pi / c * (p - 1) / (p + 1)

    def forward(self, input_tensor):
        self.calculate_weights()
        return self.conv_layer(input_tensor)

    def calculate_weights(self):
        for i in range(self.conv_layer.out_channels):
            for j in range(self.conv_layer.in_channels):
                for sigma in self.sigmas:
                    for theta in self.thetas:
                        # for lambd in lambdas:
                        for b in self.bs:
                            lambd = self.calc_lambda(sigma, b)
                            for gamma in self.gammas:
                                for psi in self.psis:
                                    gf = cv2.getGaborKernel(self.ksize, sigma, theta, lambd, gamma, psi,
                                                            ktype=cv2.CV_64F)
                                    self.conv_layer.weight.data[i, j] = torch.tensor(gf)


class ResNet(nn.Module):
    """
    Different ResNet architectures are unified with one layer of feature extraction and four layers of residuals, and the difference lies in the depth of each layer of residuals.
    For cifar10, the change of feature map size is as follows:
    (32, 32, 3) -> [Conv2d] -> (32, 32, 64) -> [Res1] -> (32, 32, 64) -> [Res2]
 -> (16, 16, 128) -> [Res3] -> (8, 8, 256) ->[Res4] -> (4, 4, 512) -> [AvgPool]
 -> (1, 1, 512) -> [Reshape] -> (512) -> [Linear] -> (10)
    """

    def __init__(self, block, num_blocks, num_classes=10, verbose=False):
        super(ResNet, self).__init__()
        self.verbose = verbose

        self.in_channels = 64
        self.features = nn.Sequential(
            # nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            GaborConv2d(in_channels=5, out_channels=self.in_channels, kernel_size=(31, 31), padding=15),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # Use the _make_layer function to generate the structures of conv2_x, conv3_x, conv4_x, conv5_x corresponding to the above table
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # After cifar10 goes through the above structure, the feature map size here is 4 x 4 x 512 x expansion
        # So here is 4 x 4 average pooling
        self.avg_pool = nn.AvgPool2d(kernel_size=4)
        self.classifer = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        # The first block needs to be downsampled
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            # If it is Bottleneck Block, you need to compress the input dimension of each layer, and then increase the dimension after compression
            # Therefore, the input dimension of each layer should also change accordingly.
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def opponency_encoder(self, data_tensor):
        num, channel, row, col = data_tensor.shape
        inputshape = (num, int(channel / 3), row, col)
        r = data_tensor[:, 0, :, :]
        g = data_tensor[:, 1, :, :]
        b = data_tensor[:, 2, :, :]
        I = ((r + g + b) / 3).reshape(inputshape)
        R = torch.clamp(r - (g + b) / 2, min=0.0)
        G = torch.clamp(g - (r - b) / 2, min=0.0)
        B = torch.clamp(b - (r + g) / 2, min=0.0)
        Y = torch.clamp((r + g) / 2 - (r - g) / 2 - b, min=0.0)
        RG = torch.clamp(R - G, min=0.0).reshape(inputshape)
        GR = torch.clamp(G - R, min=0.0).reshape(inputshape)
        BY = torch.clamp(B - Y, min=0.0).reshape(inputshape)
        YB = torch.clamp(Y - B, min=0.0).reshape(inputshape)
        return torch.cat((I, RG, GR, BY, YB), 1)

    def forward(self, x):
        out = self.features(self.opponency_encoder(x))
        if self.verbose:
            print('block 1 output: {}'.format(out.shape))
        out = self.layer1(out)
        if self.verbose:
            print('block 2 output: {}'.format(out.shape))
        out = self.layer2(out)
        if self.verbose:
            print('block 3 output: {}'.format(out.shape))
        out = self.layer3(out)
        if self.verbose:
            print('block 4 output: {}'.format(out.shape))
        out = self.layer4(out)
        if self.verbose:
            print('block 5 output: {}'.format(out.shape))
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.classifer(out)
        return out