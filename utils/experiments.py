import os
import glob
import cv2
import torch.utils.data as data
from numpy.f2py.auxfuncs import throw_error

from utils.dataset import GIANA
import utils.transforms as t
from torch.autograd import Variable
from utils.evaluation import Evaluation

from PIL import Image
from tqdm import tqdm
import numpy as np
from utils.visualize import Dashboad


class Experiments():
    def __init__(self):

        self.eval = Evaluation()
        self.board = Dashboad(8098)

    def error_hist(self, gtdir, resdir, imgprefix, plot=False):
        listGTFiles = [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join(gtdir, '*.bmp'))]

        filename_jacc = dict()
        filename_dice = dict()
        filename_sens = dict()
        filename_spec = dict()
        for currfile in tqdm(listGTFiles):
            if currfile.count('_') == 2:
                continue
            gt = np.array(Image.open(os.path.join(gtdir, currfile + '.bmp')))/255
            res = np.array(Image.open(os.path.join(resdir, currfile +'_'+ imgprefix + '.bmp')))
            res[res>10] = 255
            res /= 255

            jacc_index = self.eval.jaccard_similarity_coefficient(gt.squeeze(),res.squeeze())
            dice = self.eval.dice_coefficient(gt.squeeze(), res.squeeze())
            spec, sens, _ = self.eval.specificity_sensitivity(gt.squeeze(), res.squeeze())
            filename_jacc[currfile] = jacc_index
            filename_dice[currfile] = dice
            filename_sens[currfile] = sens
            filename_spec[currfile] = spec
        if plot:
            self.board.metric_bar(filename_jacc.values(), 'Jaccard_'+imgprefix, nbins=20)
            self.board.metric_bar(filename_dice.values(), 'Dice_'+imgprefix, nbins=20)
            self.board.metric_bar(filename_sens.values(), 'Sens_'+imgprefix, nbins=20)
            self.board.metric_bar(filename_spec.values(), 'Spec_'+imgprefix, nbins=20)


        return filename_jacc, filename_dice, filename_sens, filename_spec

    def get_failure_cases(self, args, threshold=0.5):
        list_dics = []

        for m in args.methods:
            result, _,  _, _ = self.error_hist(args.gtdir, args.resdir, m, plot=False)
            list_dics.append(result)

        # Remove the images with jaccard greater than 0.5
        for d in list_dics:
            for k, v in d.items():
                if v > threshold:
                    del d[k]
        
        # Find the failure cases common between all methods
        common_failures = set.intersection(*tuple(set(d.keys()) for d in list_dics))
        return common_failures

   
    # TODO Remove val argument in function 
    def make_grid(self, args, val=True, selected_filenames=None):        
        bordersize=2
        batch = np.empty((0,3,244,324)) 
        num2sample = 60
        if selected_filenames is not None:
            filenames = list(selected_filenames)
            num2sample = len(filenames)
        else:
            if val:
                filenames = [k.split('.')[-2].split('/')[-1] for k in glob.glob(os.path.join(args.imgdir, "val_*"))]
            if args.test:
                filenames = [k.split('.')[-2].split('/')[-1] for k in glob.glob(os.path.join(args.imgdir, "test_*.bmp"))]
            else:
                filenames = [k.split('.')[-2].split('/')[-1] for k in glob.glob(os.path.join(args.imgdir, "*.bmp"))]

        train_ind = np.random.choice(np.arange(0, len(filenames)), num2sample, replace=False)
        for i in range(train_ind.shape[0]):
            currfile = filenames[train_ind[i]]
            im = np.array(Image.open(os.path.join(args.imgdir, currfile + ".bmp")).convert('RGB'))
            # Applies a border on the top of the image
            im = cv2.copyMakeBorder(im, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize,
                                    borderType=cv2.BORDER_CONSTANT, value=[255,0,0])
            im = cv2.putText(im, currfile, (10, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75,  color=(255, 255, 255), thickness=2)
            im = im.transpose((2, 0, 1))
            batch = np.append(batch, im[np.newaxis, :, :, :], axis=0)

            if val:
                im = np.array(Image.open(os.path.join(args.gtdir, currfile + ".bmp")).convert('L'))
                im = np.repeat(im[:, :, np.newaxis], 3, axis=2)
                im = cv2.copyMakeBorder(im, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize,
                                                 borderType=cv2.BORDER_CONSTANT,value=[0,255,0])
                im = im.transpose((2, 0, 1))
                batch = np.append(batch, im[np.newaxis, :, :, :], axis=0)

            for m in args.methods:
                res = np.array(Image.open(os.path.join(args.resdir, currfile + "_" + m + ".bmp")).convert('L'))
                res = np.repeat(res[:, :, np.newaxis], 3, axis=2)
                res = cv2.copyMakeBorder(res, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize,
                                        borderType=cv2.BORDER_CONSTANT,value=[0, 255, 200])

                # Writes the name of the models.
                res = cv2.putText(res, m, (10, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75,  color=(255, 255, 255), thickness=2)
                res = res.transpose((2, 0, 1))
                batch = np.append(batch, res[np.newaxis, :, :, :], axis=0)

        return batch


