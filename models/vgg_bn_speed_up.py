import torch.nn as nn
from utils.builder import get_builder
from args import args as parser_args

class VGG19(nn.Module):

    def __init__(self):
        super(VGG19, self).__init__()
        builder = get_builder()
        self.conv1 = builder.conv3x3(3, 64)
        self.bn1 = nn.BatchNorm2d(64, track_running_stats=False)
        self.conv2 = builder.conv3x3(64, 64)
        self.bn2 = nn.BatchNorm2d(64, track_running_stats=False)
        self.conv3 = builder.conv3x3(64, 128)
        self.bn3 = nn.BatchNorm2d(128, track_running_stats=False)
        self.conv4 = builder.conv3x3(128, 128)
        self.bn4 = nn.BatchNorm2d(128, track_running_stats=False)
        self.conv5 = builder.conv3x3(128, 256)
        self.bn5 = nn.BatchNorm2d(256, track_running_stats=False)
        self.conv6 = builder.conv3x3(256, 256)
        self.bn6 = nn.BatchNorm2d(256, track_running_stats=False)
        self.conv7 = builder.conv3x3(256, 256)
        self.bn7 = nn.BatchNorm2d(256, track_running_stats=False)
        self.conv8 = builder.conv3x3(256, 256)
        self.bn8 = nn.BatchNorm2d(256, track_running_stats=False)
        self.conv9 = builder.conv3x3(256, 512)
        self.bn9 = nn.BatchNorm2d(512, track_running_stats=False)
        self.conv10 = builder.conv3x3(512, 512)
        self.bn10 = nn.BatchNorm2d(512, track_running_stats=False)
        self.conv11 = builder.conv3x3(512, 512)
        self.bn11 = nn.BatchNorm2d(512, track_running_stats=False)
        self.conv12 = builder.conv3x3(512, 512)
        self.bn12 = nn.BatchNorm2d(512, track_running_stats=False)
        self.conv13 = builder.conv3x3(512, 512)
        self.bn13 = nn.BatchNorm2d(512, track_running_stats=False)
        self.conv14 = builder.conv3x3(512, 512)
        self.bn14 = nn.BatchNorm2d(512, track_running_stats=False)
        self.conv15 = builder.conv3x3(512, 512)
        self.bn15 = nn.BatchNorm2d(512, track_running_stats=False)
        self.conv16 = builder.conv3x3(512, 512)
        self.bn16 = nn.BatchNorm2d(512, track_running_stats=False)
        num_classes = 10 if parser_args.set == "CIFAR10" else 100
        self.linear = builder.conv1x1(512, num_classes)

    def forward(self, x):
        print("--1--")
        x, mask = self.conv1(x)
        print("--2--")


        x = nn.ReLU(inplace=True)(self.bn1(x))
        x, mask = self.conv2(x, mask)
        print("--3--")

        x = nn.ReLU(inplace=True)(self.bn2(x))
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)

        x, mask = self.conv3(x, mask)
        print("--4--")

        x = nn.ReLU(inplace=True)(self.bn3(x))
        x, mask = self.conv4(x, mask)
        x = nn.ReLU(inplace=True)(self.bn4(x))
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)

        x, mask = self.conv5(x, mask)
        x = nn.ReLU(inplace=True)(self.bn5(x))
        x, mask = self.conv6(x, mask)
        x = nn.ReLU(inplace=True)(self.bn6(x))
        x, mask = self.conv7(x, mask)
        x = nn.ReLU(inplace=True)(self.bn7(x))
        x, mask = self.conv8(x, mask)
        x = nn.ReLU(inplace=True)(self.bn8(x))
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)

        x, mask = self.conv9(x, mask)
        x = nn.ReLU(inplace=True)(self.bn9(x))
        x, mask = self.conv10(x, mask)
        x = nn.ReLU(inplace=True)(self.bn10(x))
        x, mask = self.conv11(x, mask)
        x = nn.ReLU(inplace=True)(self.bn11(x))
        x, mask = self.conv12(x, mask)
        x = nn.ReLU(inplace=True)(self.bn12(x))
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)

        x, mask = self.conv13(x, mask)
        x = nn.ReLU(inplace=True)(self.bn13(x))
        x, mask = self.conv14(x, mask)
        x = nn.ReLU(inplace=True)(self.bn14(x))
        x, mask = self.conv15(x, mask)
        x = nn.ReLU(inplace=True)(self.bn15(x))
        x, mask = self.conv16(x, mask)
        x = nn.ReLU(inplace=True)(self.bn16(x))
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)

        x = self.linear(x)
        return x.squeeze()

def vgg19_bn_speed_up():
    return VGG19()
cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}