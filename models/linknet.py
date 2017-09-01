import torch.nn as nn

import torch.nn.functional as F
from utils.tools import initialize_weights


class ConvBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,  stride, padding=0, bias=True):
        super(ConvBatchNorm, self).__init__()
        self.cb_unit = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                               padding=padding, stride=stride, bias=bias),
                                     nn.BatchNorm2d(out_channels))
        
    def forward(self, inputs):        
        return self.cb_unit(inputs)


class ConvBatchNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,  stride, padding=0, bias=True):
        super(ConvBatchNormRelu, self).__init__()
        self.cbr_unit = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                                padding=padding, stride=stride, bias=bias),
                                      nn.BatchNorm2d(out_channels),
                                      nn.ReLU(inplace=True))

    def forward(self, inputs):
        return self.cbr_unit(inputs)
    
    
class DeConvBatchNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, bias=True):
        super(DeConvBatchNormRelu, self).__init__()
        self.dcbr_unit = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size,
                                                padding=padding, stride=stride, bias=bias),
                                       nn.BatchNorm2d(out_channels),
                                       nn.ReLU(inplace=True))

    def forward(self, inputs):
        return self.dcbr_unit(inputs)


class DeConvBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,  stride, padding=0, bias=True):
        super(DeConvBatchNorm, self).__init__()
        self.dcb_unit = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size,
                                               padding=padding, stride=stride, bias=bias),
                                 nn.BatchNorm2d(out_channels),)

    def forward(self, inputs):
        return self.dcb_unit(inputs)
    
    
class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()

        self.convbnrelu1 = ConvBatchNormRelu(in_channels, out_channels, 3,  stride, 1, bias=False)
        self.convbn2 = ConvBatchNorm(out_channels, out_channels, 3, 1, 1, bias=False)
        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.convbnrelu1(x)
        out = self.convbn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class ResidualBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBottleneck, self).__init__()
        self.convbn1 = nn.ConvBatchNorm(in_channels,  out_channels, kernel_size=1, bias=False)
        self.convbn2 = nn.ConvBatchNorm(out_channels,  out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.convbn3 = nn.ConvBatchNorm(out_channels,  out_channels * 4, kernel_size=1, bias=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.convbn1(x)
        out = self.convbn2(out)
        out = self.convbn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = F.relu(out)

        return out


class linknetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(linknetUp, self).__init__()

        # B, 2C, H, W -> B, C/2, H, W
        self.convbnrelu1 = ConvBatchNormRelu(in_channels, out_channels/2, kernel_size=1, stride=1)

        # B, C/2, H, W -> B, C/2, H, W
        self.deconvbnrelu2 = DeConvBatchNormRelu(out_channels/2, out_channels/2, kernel_size=2,  stride=2)

        # B, C/2, H, W -> B, C, H, W
        self.convbnrelu3 = ConvBatchNormRelu(out_channels/2, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.convbnrelu1(x)
        x = self.deconvbnrelu2(x)
        x = self.convbnrelu3(x)
        return x


class LinkNet(nn.Module):

    def __init__(self, n_classes=1):
        super(LinkNet, self).__init__()
        self.is_deconv = True
        self.in_channels = 3
        self.is_batchnorm = True
        self.feature_scale = 1
        self.layers = [2, 2, 2, 2] # Currently hardcoded for ResNet-18

        filters = [64, 128, 256, 512]
        filters = [x / self.feature_scale for x in filters]

        self.inplanes = filters[0]


        # Encoder
        self.convbnrelu1 = ConvBatchNormRelu(in_channels=3, kernel_size=7, out_channels=64,
                                               padding=3, stride=2, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        block = ResidualBlock
        self.encoder1 = self._make_layer(block, filters[0], self.layers[0])
        self.encoder2 = self._make_layer(block, filters[1], self.layers[1], stride=2)
        self.encoder3 = self._make_layer(block, filters[2], self.layers[2], stride=2)
        self.encoder4 = self._make_layer(block, filters[3], self.layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)


        # Decoder
        self.decoder4 = linknetUp(filters[3], filters[2])
        self.decoder3 = linknetUp(filters[2], filters[1])
        self.decoder2 = linknetUp(filters[1], filters[0])
        self.decoder1 = linknetUp(filters[0], filters[0])

        # Final Classifier
        self.finaldeconvbnrelu1 = nn.Sequential(nn.ConvTranspose2d(filters[0], 32/self.feature_scale, 2, 2, 0),
                                      nn.BatchNorm2d(32/self.feature_scale),
                                      nn.ReLU(inplace=True),)
        self.finalconvbnrelu2 = ConvBatchNormRelu(in_channels=32/self.feature_scale, kernel_size=3, out_channels=32/self.feature_scale, padding=1, stride=1)
        self.finalconv3 = nn.Conv2d(32/self.feature_scale, n_classes, 1, 1, 0)
        self.activation = nn.Sigmoid()

        initialize_weights(self)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion,
                                                 kernel_size=1, stride=stride, bias=False),
                                       nn.BatchNorm2d(planes * block.expansion),)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        # print layers
        return nn.Sequential(*layers)

    def forward(self, x):
        # print 'x', x.size()
        # Encoder
        x = self.convbnrelu1(x)
        # print 'x', x.size()
        x = self.maxpool(x)
        # print 'x', x.size()

        e1 = self.encoder1(x)
        # print 'e1', e1.size()
        e2 = self.encoder2(e1)
        # print 'e2', e2.size()
        e3 = self.encoder3(e2)
        # print 'e3', e3.size()
        e4 = self.encoder4(e3)
        # print 'e4', e4.size()

        # Decoder with Skip Connections
        d4 = F.upsample_bilinear(self.decoder4(e4), e3.size()[2:])
        # print 'd4', d4.size()
        d4 = d4 + e3
        d3 = self.decoder3(d4)
        # print 'd3', d3.size()
        d3 = d3 + e2
        d2 = self.decoder2(d3)
        # print 'd2', d2.size()
        d2 = d2 + e1
        d1 = self.decoder1(d2)
        # print 'd1', d1.size()

        # Final Classification
        f1 = self.finaldeconvbnrelu1(d1)
        # print 'f1', f1.size()
        f2 = self.finalconvbnrelu2(f1)
        # print 'f2', f2.size()
        f3 = self.finalconv3(f2)
        # print 'f3', f3.size()
        return self.activation(f3)
