import torch.nn as nn
import torchvision.models as models

from utils.tools import initialize_weights


class DUC(nn.Module):
    def __init__(self, in_channels, factor, num_classes=1):
        super(DUC, self).__init__()

        self.layer = nn.Sequential(nn.Conv2d(in_channels, (factor**2)*num_classes, kernel_size=3, padding=1),
                                   nn.PixelShuffle(factor))

    def forward(self, x):
        return self.layer(x)


class ResentDUC(nn.Module):
    def __init__(self, num_channels):
        super(ResentDUC, self).__init__()

        resent= models.resnet152(pretrained=True)

        self.layer0 = nn.Sequential(resent.conv1, resent.bn1, resent.relu, resent.maxpool)
        self.layer1 = resent.layer1
        self.layer2 = resent.layer2
        self.layer3 = resent.layer3
        self.layer4 = resent.layer4

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation = (2,2)
                m.padding = (2,2)
                m.stride = (1,1)
            if 'downsample.0' in n:
                m.stride = (1,1)

        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation = (4,4)
                m.padding = (4,4)
                m.stride = (1,1)
            if 'downsample.0' in n:
                m.stride = (1,1)

            self.duc = DUC(2048, 8, num_classes=num_channels)
            self.activation = nn.Sigmoid()

            initialize_weights(self.duc)

    def forward(self,x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.duc(x)
        return self.activation(x)


class ResentDUCHDC(nn.Module):
    def __init__(self, num_channels):
        super(ResentDUCHDC, self).__init__()

        resent = models.resnet152(pretrained=True)

        self.layer0 = nn.Sequential(resent.conv1, resent.bn1, resent.relu, resent.maxpool)
        self.layer1 = resent.layer1
        self.layer2 = resent.layer2
        self.layer3 = resent.layer3
        self.layer4 = resent.layer4

        for n, m in self.layer3.named_modules():
            if 'conv2' in n or 'downsample.0' in n:
                m.stride = (1, 1)

        for n, m in self.layer4.named_modules():
            if 'conv2' in n or 'downsample.0' in n:
                m.stride = (1, 1)

        layer3_group_dilation = [1, 2, 5, 9]
        for idx in range(0, 23):
            self.layer3[idx].conv2.dilation = (layer3_group_dilation[idx%4], layer3_group_dilation[idx%4])
            self.layer3[idx].conv2.padding = (layer3_group_dilation[idx % 4], layer3_group_dilation[idx % 4])

        layer4_group_dilation = [5, 9, 17]
        for idx in range(0, 3):
            self.layer4[idx].conv2.dilation = (layer4_group_dilation[idx], layer4_group_dilation[idx])
            self.layer4[idx].conv2.padding = (layer4_group_dilation[idx], layer4_group_dilation[idx])


        self.duc = DUC(2048, 8)
        self.activation = nn.Sigmoid()

        initialize_weights(self.duc)

    def forward(self,x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.duc(x)
        return self.activation(x)
