import os
import numpy as np
import glob
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from PIL import Image


class Evaluation():
    def __init__(self):
        self.jaccard_score = 0.
        self.dice = 0.
        self.spec = 0.
        self.sens = 0.
        self.acc = 0.

    def __call__(self, gtdir, resdir, resprefix):

        listImgFiles = [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join(gtdir, 'val_*.bmp'))]
        for currFile in tqdm(listImgFiles):
	    res = np.array(Image.open(os.path.join(resdir, currFile + '_' + resprefix + '.bmp')).convert('L'))
	    res = np.float32(res)/255
	    
            gt = np.array(Image.open(os.path.join(gtdir, currFile + '.bmp')).convert('L'))
   	    gt = np.float32(gt)/255

            self.jaccard_score += self.jaccard_similarity_coefficient(gt, res)
            self.dice += self.dice_coefficient(gt, res)
            spec_tmp, sens_tmp, acc_tmp = self.specificity_sensitivity(gt, res)
            self.spec += spec_tmp
            self.sens += sens_tmp
            self.acc += acc_tmp

        self.dice /= len(listImgFiles)
        self.jaccard_score /= len(listImgFiles)
        self.spec /= len(listImgFiles)
        self.sens /= len(listImgFiles)
        self.acc /= len(listImgFiles)

    def print_vals(self):
        print('DiceCoefficient: {}\n'
              'JaccardIndex: {}\n'
              'Specificity: {}\n'
              'Sensitivity: {}\n'
              'Accuracy: {}'.format(self.dice, self.jaccard_score, self.spec, self.sens, self.acc))

    def dice_coefficient(self, res, gt):
        A = gt.flatten()
        B = res.flatten()

        A = np.array([1 if x > 0.5 else 0.0 for x in A])
        B = np.array([1 if x > 0.5 else 0.0 for x in B])
        dice = np.sum(B[A==1.0])*2.0 / (np.sum(B) + np.sum(A))
        return dice

    def specificity_sensitivity(self, gt, res):
        A = gt.flatten()
        B = res.flatten()

        A = np.array([1 if x > 0.5 else 0.0 for x in A])
        B = np.array([1 if x > 0.5 else 0.0 for x in B])

        tn, fp, fn, tp = np.float32(confusion_matrix(A, B).ravel())
        specificity = tn/(fp + tn)
        sensitivity = tp / (tp + fn)
        accuracy = (tp+tn)/(tp+fp+fn+tn)

        return specificity, sensitivity,accuracy

    def jaccard_similarity_coefficient(self, A, B, no_positives=1.0):
        """Returns the jaccard index/similarity coefficient between A and B.

        This should work for arrays of any dimensions.

        J = len(intersection(A,B)) / len(union(A,B))

        To extend to probabilistic input, to compute the intersection, use the min(A,B).
        To compute the union, use max(A,B).

        Assumes that a value of 1 indicates the positive values.
        A value of 0 indicates the negative values.

        If no positive values (1) in either A or B, then returns no_positives.
        """
        # Make sure the shapes are the same.
        A = A.squeeze()
        B = B.squeeze()
        if not A.shape == B.shape:
            raise ValueError("A and B must be the same shape")



        # Make sure values are between 0 and 1.
        if np.any( (A>1.) | (A<0) | (B>1.) | (B<0)):
            raise ValueError("A and B must be between 0 and 1")

        # Flatten to handle nd arrays.
        A = A.flatten()
        B = B.flatten()

        intersect = np.minimum(A,B)
        union = np.maximum(A, B)

        # Special case if neither A or B have a 1 value.
        if union.sum() == 0:
            return no_positives

        # Compute the Jaccard.
        J = float(intersect.sum()) / union.sum()
        return J
