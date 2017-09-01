import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weights=None):
        super(CrossEntropyLoss2d, self).__init__()

        self.loss = nn.NLLLoss2d(weight=weights)
        self.loss.cuda()

    def forward(self, outputs, targets):
        return self.loss(F.log_softmax(outputs), targets)
