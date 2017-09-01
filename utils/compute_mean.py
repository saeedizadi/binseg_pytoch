import torch
from dataset import GIANA
import numpy as np
import argparse



def compute_mean(args):

    dsetTrain = GIANA(args.imgdir, args.gtdir, train=True).train_data
    dsetTrain = dsetTrain.astype(np.float32)/255

    print dsetTrain.shape
    mean = []
    std = []

    for i in range(3):
        pixels = dsetTrain[:, :, :, i].ravel()
        mean.append(np.mean(pixels))
        std.append(np.std(pixels))
    print("means: {}".format(mean))
    print("stdevs: {}".format(std))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgdir', type=str, required=True)
    parser.add_argument('--gtdir', type=str)

    args = parser.parse_args()
    compute_mean(args)