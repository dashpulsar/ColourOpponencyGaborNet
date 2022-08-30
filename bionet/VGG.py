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

Batch_Size = 256
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

'''cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}'''


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


class VGG(nn.Module):

    def __init__(self, vgg_name, param=None):
        super(VGG, self).__init__()
        self.param = param
        self.Gabor_out_channels = 64
        '''if param:
          self.g0=GaborConvFixed(in_channels=3, out_channels= 24, input_dict=self.param,kernel_size=(31, 31) ,padding=15)
        else:'''
        self.g0 = GaborConv2d(in_channels=5, out_channels=self.Gabor_out_channels, kernel_size=(31, 31), padding=15)
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(512, 10),
        )
        #         self.classifier = nn.Linear(512,10)

        self._initialize_weight()

    def forward(self, x):
        x = self.opponency_encoder(x)
        out = self.features(self.g0(x))
        # 在进入
        out = out.view(out.size(0), -1)
        # out = self.classifier(out)
        return out

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

    # make layers

    def _make_layers(self, cfg):
        layers = []
        in_channels = self.Gabor_out_channels  # RGB 初始通道为3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]  # kernel_size 为 2 x 2,然后步长为2
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),  # 都是(3.3)的卷积核
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]  # RelU
                in_channels = x  # 重定义通道
        #         layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    # 初始化参数
    def _initialize_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # xavier is used in VGG's paper
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


input_dict = {  # 'ksize': (127, 127),
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

