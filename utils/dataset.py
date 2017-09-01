import torch
import torchvision.datasets as dst
import torch.utils.data as data
from PIL import Image
import glob
import os
import  numpy as np
from random import shuffle
import shutil
from PIL import ImageOps
from PIL import Image
from tqdm import tqdm


class GIANA(data.Dataset):
    def __init__(self, img_root, gt_root, input_size=(320, 240), train=True, transform=None, target_transform=None, co_transform=None):
        self.img_root = img_root
        self.gt_root = gt_root

        self.transform = transform
        self.target_transform = target_transform
        self.co_transform = co_transform

        self.img_filenames = []
        self.gt_filenames = []
        self.input_width = input_size[0]
        self.input_height = input_size[1]

        # --- Check if the dataset is already partitoned and augmented ---
        # --- Otherwise. first partition the data, and them augment the training set ---
        temp = glob.glob(os.path.join(self.img_root, "train_*.bmp"))
        if not temp:
            self.partition_data(1.)
            self.augment_data()

        if train:
            self.img_filenames = glob.glob(os.path.join(self.img_root, "train_*.bmp"))
            self.gt_filenames = glob.glob(os.path.join(self.gt_root, "train_*.bmp"))

            self.train_data = []
            for fname in self.img_filenames:
                im = np.array(Image.open(fname).convert('RGB'))
                self.train_data.append(im)
            self.train_data = np.stack(self.train_data, axis=0)

        else:
            self.img_filenames = glob.glob(os.path.join(self.img_root, "val_*.bmp"))
            self.gt_filenames = glob.glob(os.path.join(self.gt_root, "val_*.bmp"))        

    def __getitem__(self, index):
        im = Image.open(self.img_filenames[index]).convert('RGB')
        target = Image.open(self.gt_filenames[index]).convert('L')

        # Note: co_transforms must came before ToTensor(), because some functions like flipr does not work on TorchTensor
        # So you should apply the transformations first, and then transform it to TorchTensor
        if self.co_transform is not None:
            im, target = self.co_transform(im, target)
        if self.transform is not None:
            im = self.transform(im)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return im, target

    def __len__(self):
        return len(self.img_filenames)

    def partition_data(self, fraction_value):
        filenameList= [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join(self.img_root, '*.bmp'))]
        shuffle(filenameList)

        nSamples = len(filenameList)
        nTraining = int(fraction_value*nSamples)
        nValidation = int((1-fraction_value)*nSamples)

        dirs = [self.img_root, self.gt_root]
        for index in range(nTraining):
            for d in dirs:
                srcFilename = os.path.join(d, filenameList[index] + '.bmp')
                dstFilename = os.path.join(d, 'train_' + filenameList[index]+ '.bmp')
                shutil.move(srcFilename, dstFilename)
        
        if nValidation != 0:
            for index in range(nTraining, nTraining+nValidation+1):
                for d in dirs:
                    srcFilename = os.path.join(d, filenameList[index] + '.bmp')
                    dstFilename = os.path.join(d, 'val_' + filenameList[index] + '.bmp')
                    shutil.move(srcFilename, dstFilename)

    def augment_data(self, rotation=True, HFlip=True, VFlip=True):
        
        listImgFiles = [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join(self.img_root, '*.bmp'))]
        listFilesTrain = [k for k in listImgFiles if 'train' in k]
        listFilesVal = [k for k in listImgFiles if 'train' not in k]

        for filenames in tqdm(listFilesVal):
            src_im = Image.open(os.path.join(self.img_root, filenames + '.bmp')).resize((self.input_width, self.input_height),Image.ANTIALIAS)
            gt_im = Image.open(os.path.join(self.gt_root, filenames + '.bmp')).resize((self.input_width, self.input_height),Image.ANTIALIAS)
            src_im.save(os.path.join(self.img_root, filenames + '.bmp'))
            gt_im.save(os.path.join(self.gt_root, filenames + '.bmp'))

        for filenames in tqdm(listFilesTrain):
            src_im = Image.open(os.path.join(self.img_root, filenames + '.bmp')).resize((self.input_width, self.input_height),Image.ANTIALIAS)
            gt_im = Image.open(os.path.join(self.gt_root, filenames + '.bmp')).resize((self.input_width, self.input_height),Image.ANTIALIAS)
            src_im.save(os.path.join(self.img_root, filenames + '.bmp'))
            gt_im.save(os.path.join(self.gt_root, filenames + '.bmp'))
            if rotation:
                for angle in [90, 180, 270]:
                    src_im = Image.open(os.path.join(self.img_root, filenames + '.bmp'))
                    gt_im = Image.open(os.path.join(self.gt_root, filenames + '.bmp'))
                    rot_im = src_im.rotate(angle, expand=True).resize((self.input_width, self.input_height),Image.ANTIALIAS)
                    rot_gt = gt_im.rotate(angle, expand=True).resize((self.input_width, self.input_height), Image.ANTIALIAS)
                    rot_im.save(os.path.join(self.img_root, filenames + '_' + str(angle) + '.bmp'))
                    rot_gt.save(os.path.join(self.gt_root, filenames + '_' + str(angle) + '.bmp'))
            if VFlip:
                vert_im = ImageOps.flip(src_im)
                vert_gt = ImageOps.flip(gt_im)
                vert_im.save(os.path.join(self.img_root, filenames + '_vert.bmp'))
                vert_gt.save(os.path.join(self.gt_root, filenames + '_vert.bmp'))
            if HFlip:
                horz_im = ImageOps.mirror(src_im)
                horz_gt = ImageOps.mirror(gt_im)
                horz_im.save(os.path.join(self.img_root, filenames + '_horz.bmp'))
                horz_gt.save(os.path.join(self.gt_root, filenames + '_horz.bmp'))

