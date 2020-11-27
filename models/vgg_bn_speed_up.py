import torch.nn as nn
import torch
from utils.builder import get_builder
from args import args as parser_args
import time

class VGG19(nn.Module):

    def __init__(self):
        super(VGG19, self).__init__()
        builder = get_builder()
        self.conv1 = builder.conv3x3(3, 64)
        self.bn1 = nn.BatchNorm2d(64, track_running_stats=False, affine=False)
        self.bn1_w = nn.Parameter(torch.Tensor(64))
        self.bn1_b = nn.Parameter(torch.Tensor(64))
        # print("size b_w", self.bn1_w.size())
        # print("size b_b", self.bn1_b.size())
        self.conv2 = builder.conv3x3(64, 64)
        self.bn2 = nn.BatchNorm2d(64, track_running_stats=False, affine=False)
        self.conv3 = builder.conv3x3(64, 128)
        self.bn3 = nn.BatchNorm2d(128, track_running_stats=False, affine=False)
        self.conv4 = builder.conv3x3(128, 128)
        self.bn4 = nn.BatchNorm2d(128, track_running_stats=False, affine=False)
        self.conv5 = builder.conv3x3(128, 256)
        self.bn5 = nn.BatchNorm2d(256, track_running_stats=False, affine=False)
        self.conv6 = builder.conv3x3(256, 256)
        self.bn6 = nn.BatchNorm2d(256, track_running_stats=False, affine=False)
        self.conv7 = builder.conv3x3(256, 256)
        self.bn7 = nn.BatchNorm2d(256, track_running_stats=False, affine=False)
        self.conv8 = builder.conv3x3(256, 256)
        self.bn8 = nn.BatchNorm2d(256, track_running_stats=False, affine=False)
        self.conv9 = builder.conv3x3(256, 512)
        self.bn9 = nn.BatchNorm2d(512, track_running_stats=False, affine=False)
        self.conv10 = builder.conv3x3(512, 512)
        self.bn10 = nn.BatchNorm2d(512, track_running_stats=False, affine=False)
        self.conv11 = builder.conv3x3(512, 512)
        self.bn11 = nn.BatchNorm2d(512, track_running_stats=False, affine=False)
        self.conv12 = builder.conv3x3(512, 512)
        self.bn12 = nn.BatchNorm2d(512, track_running_stats=False, affine=False)
        self.conv13 = builder.conv3x3(512, 512)
        self.bn13 = nn.BatchNorm2d(512, track_running_stats=False, affine=False)
        self.conv14 = builder.conv3x3(512, 512)
        self.bn14 = nn.BatchNorm2d(512, track_running_stats=False, affine=False)
        self.conv15 = builder.conv3x3(512, 512)
        self.bn15 = nn.BatchNorm2d(512, track_running_stats=False, affine=False)
        self.conv16 = builder.conv3x3(512, 512)
        self.bn16 = nn.BatchNorm2d(512, track_running_stats=False, affine=False)
        num_classes = 10 if parser_args.set == "CIFAR10" else 100
        self.linear = builder.conv1x1(512, num_classes)

    def forward(self, x):
        # print("--1--")
        t1 = time.time()
        for i in range(100):
            x, mask = self.conv1(x)
        t2 = time.time()
        print(t2-t1)

        # print("--2--")
        # print("x, mask", x.size(), mask.size())
        # x = self.bn1(x)
        # # print("x, after bn1", x.size())
        # # print("before select: ", self.bn1_w.size(), mask.size())
        # # print("after select: ", torch.masked_select(self.bn1_w, mask.squeeze()).size())
        # masked_bn1_w = torch.masked_select(self.bn1_w, mask.squeeze()).view(1, mask.sum(), 1, 1)
        # masked_bn1_b = torch.masked_select(self.bn1_b, mask.squeeze()).view(1, mask.sum(), 1, 1)
        # # print("m_w, m_b, x", masked_bn1_w.size(), masked_bn1_b.size(), x.size())
        # x = x*masked_bn1_w+masked_bn1_b
        # # print("m_w, m_b, x", masked_bn1_w.size(), masked_bn1_b.size(), x.size())
        # x = nn.ReLU(inplace=True)(x)
        # x, mask = self.conv2(x, mask)
        # # print("--3--")
        #
        # x = nn.ReLU(inplace=True)(self.bn2(x))
        # x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        #
        # x, mask = self.conv3(x, mask)
        # # print("--4--")
        #
        # x = nn.ReLU(inplace=True)(self.bn3(x))
        # x, mask = self.conv4(x, mask)
        # x = nn.ReLU(inplace=True)(self.bn4(x))
        # x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        #
        # x, mask = self.conv5(x, mask)
        # x = nn.ReLU(inplace=True)(self.bn5(x))
        # x, mask = self.conv6(x, mask)
        # x = nn.ReLU(inplace=True)(self.bn6(x))
        # x, mask = self.conv7(x, mask)
        # x = nn.ReLU(inplace=True)(self.bn7(x))
        # x, mask = self.conv8(x, mask)
        # x = nn.ReLU(inplace=True)(self.bn8(x))
        # x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        #
        # x, mask = self.conv9(x, mask)
        # x = nn.ReLU(inplace=True)(self.bn9(x))
        # x, mask = self.conv10(x, mask)
        # x = nn.ReLU(inplace=True)(self.bn10(x))
        # x, mask = self.conv11(x, mask)
        # x = nn.ReLU(inplace=True)(self.bn11(x))
        # x, mask = self.conv12(x, mask)
        # x = nn.ReLU(inplace=True)(self.bn12(x))
        # x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        #
        # x, mask = self.conv13(x, mask)
        # x = nn.ReLU(inplace=True)(self.bn13(x))
        # x, mask = self.conv14(x, mask)
        # x = nn.ReLU(inplace=True)(self.bn14(x))
        # x, mask = self.conv15(x, mask)
        # x = nn.ReLU(inplace=True)(self.bn15(x))
        # x, mask = self.conv16(x, mask)
        # x = nn.ReLU(inplace=True)(self.bn16(x))
        # x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        # # print("before linear")
        # x, mask = self.linear(x, mask)
        # print("after linear")
        return x.squeeze()

def vgg19_bn_speed_up():
    return VGG19()
cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}