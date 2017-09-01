import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class FCN32(nn.Module):
    def __init__(self, num_classes):
        super(FCN32,self).__init__()


        #Set TORCH_MODEL_ZOO in your enviroment fro pretrained setting
        self.feats = models.vgg16(pretrained=True).features
        self.fconn = nn.Sequential(nn.Conv2d(512, 4096, 7, padding=3),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(4096, 4096,1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(4096, num_classes, 1)
                                   )
        self.activation = nn.Sigmoid()

    def forward(self,x):

        feats = self.feats(x)
        fconn = self.fconn(feats)
        upsample = F.upsample_bilinear(fconn, x.size()[2:])
        out = self.activation(upsample)
        return out


class FCN16(nn.Module):
    def __init__(self, num_classes):
        super(FCN16, self).__init__()

        feats = list(models.vgg16(pretrained=True).features.children())
        self.feats = nn.Sequential(*feats[0:17])
        self.pool4 = nn.Sequential(*feats[17:24])
        self.pool5 = nn.Sequential(*feats[24:])

        self.fconn = nn.Sequential(nn.Conv2d(512, 4096, 7, padding=3),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(4096, 4096, 1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(4096, num_classes, 1)
                                   )
        self.score_pool4 = nn.Conv2d(512, num_classes, 1)
        self.activation = nn.Sigmoid()

    def forward(self,x):
        feats = self.feats(x)
        pool4 = self.pool4(feats)
        pool5 = self.pool5(pool4)

        score_fconn = self.fconn(pool5)
        score_pool4 = self.score_pool4(pool4)

        resize_score_fconn = F.upsample_bilinear(score_fconn, score_pool4.size()[2:])
        prediction = resize_score_fconn + score_pool4
        upsample = F.upsample_bilinear(prediction, x.size()[2:])

        return self.activation(upsample)


class FCN8(nn.Module):
    def __init__(self, num_classes):
        super(FCN8, self).__init__()

        feats = list(models.vgg16(pretrained=True).features.children())

        self.feats = nn.Sequential(*feats[0:10])
        self.pool3= nn.Sequential(*feats[10:17])
        self.pool4 = nn.Sequential(*feats[17:24])
        self.pool5 = nn.Sequential(*feats[24:])

        self.fconn = nn.Sequential(nn.Conv2d(512, 4096, 7, padding=3),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(4096, 4096, 1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(4096, num_classes, 1)
                                   )

        self.score_pool3 = nn.Conv2d(256, num_classes, 1)
        self.score_pool4 = nn.Conv2d(512, num_classes, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        feats = self.feats(x)
        pool3 = self.pool3(feats)
        pool4 = self.pool4(pool3)
        pool5 = self.pool5(pool4)
        fconn = self.fconn(pool5)

        score_pool3 = self.score_pool3(pool3)
        score_pool4 = self.score_pool4(pool4)

        resized_score_pool4 = F.upsample_bilinear(score_pool4, pool3.size()[2:])
        resized_score_fconn = F.upsample_bilinear(fconn, pool3.size()[2:])

        prediction = resized_score_pool4 + resized_score_fconn + score_pool3
        upsample = F.upsample_bilinear(prediction, x.size()[2:])

        return self.activation(upsample)




