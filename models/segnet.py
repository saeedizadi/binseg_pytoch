import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from utils.tools import initialize_weights


class DecodeBlock(nn.Module):
    def __init__(self, in_channels, out_channles, kernel_size=3, stride=1, padding=1, num_layers=2):
        super(DecodeBlock, self).__init__()


        layers = [nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                  nn.BatchNorm2d(in_channels),
                  nn.ReLU(inplace=True)] * (num_layers -1)

        layers += [nn.Conv2d(in_channels, out_channles, kernel_size=kernel_size, stride=stride, padding=padding),
                  nn.BatchNorm2d(out_channles),
                  nn.ReLU(inplace=True)]


        self.layer = nn.Sequential(*layers)


    def forward(self, x):
        return  self.layer(x)

class SegNet(nn.Module):
    def __init__(self, num_classes):
        super(SegNet, self).__init__()

        modules= list(models.vgg16(pretrained=True).features.children())

        self.conv1 = nn.Sequential(*modules[0:4])
        self.conv2 = nn.Sequential(*modules[5:9])
        self.conv3 = nn.Sequential(*modules[10:16])
        self.conv4 = nn.Sequential(*modules[17:23])
        self.conv5 = nn.Sequential(*modules[24:30])

        self.dec512 = DecodeBlock(512,512,3,1,num_layers=3)
        self.dec256 = DecodeBlock(512, 256, 3, 1, num_layers=3)
        self.dec128 = DecodeBlock(256, 128, 3, 1, num_layers=3)
        self.dec64 = DecodeBlock(128, 64, 3, 1, num_layers=2)

        self.final = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                   nn.Conv2d(64, 1, kernel_size=3, padding=1))


        self.activation = nn.Sigmoid()

        initialize_weights(self.dec512,self.dec256,self.dec128,self.dec64, self.final)


    def forward(self,x):


        ## Assumion X of size 240x320x3
        pool1, indices1 = F.max_pool2d(self.conv1(x), 2, 2, return_indices=True) ## 120x160x64
        pool2, indices2 = F.max_pool2d(self.conv2(pool1), 2,2,return_indices=True) ## 60x80x128
        pool3, indices3 = F.max_pool2d(self.conv3(pool2), 2,2, return_indices=True) ## 30x40x256
        pool4, indices4 = F.max_pool2d(self.conv4(pool3), 2, 2, return_indices=True) ## 15x20x512
        pool5, indices5 = F.max_pool2d(self.conv5(pool4), 2, 2, return_indices=True) ## 7x10x512
        dec1 = self.dec512(F.max_unpool2d(pool5,indices5,2, 2, output_size=pool4.size())) #15x20x512
        dec2 = self.dec256(F.max_unpool2d(dec1, indices4,2,2)) ## 30x40x256
        dec3 = self.dec128(F.max_unpool2d(dec2, indices3, 2, 2)) ## 60x80x128
        dec4 = self.dec64(F.max_unpool2d(dec3, indices2, 2, 2))  ## 120x160x64
        dec5 = self.final(F.max_unpool2d(dec4, indices1, 2, 2))  ## 240x320x1

        return self.activation(dec5)
