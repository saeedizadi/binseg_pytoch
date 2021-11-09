
# AECNN: Adversarial and Enhanced Convolutional Neural Networks

# Abstract

Method for segmenting gastrointestinal polyps from colonscopy images uses an adversarial and enhanced convolutional neural networks (AECNN). As the number of training images is small, the core of AECNN relies on fine-tuning an existing deep CNN model (ResNet152). AECNN’s enhanced convolutions incorporate both dense upsampling, which learns to upsample the low-resolution feature maps into pixel-level segmentation masks, as well as hybrid dilation, which improves the dilated convolution by using different dilation rates for different layers. AECNN further boosts the performance of its segmenter by incorporating a discriminator competing with the segmenter, where both are trained through a generative adversarial network formulation.

# Keywords
Segmentation, Binary segmentation, Deep learning, polyp segmentation, Endoscopy analysis

# Cite
If you use our code, please cite our paper: 
[AECNN: Adversarial and Enhanced Convolutional Neural Networks](https://www2.cs.sfu.ca/~hamarneh/ecopy/caagv2021.pdf)

The corresponding bibtex entry is:

```
@incollection{izadi2021aecnn,
  title={AECNN: Adversarial and Enhanced Convolutional Neural Networks},
  author={Izadi, Saeed and Hamarneh, Ghassan},
  booktitle={Computer-Aided Analysis of Gastrointestinal Videos},
  pages={59--62},
  year={2021},
  publisher={Springer}
}
```

# The source code
The implementation for several state of the art binary segmentation method in PyTorch. This repo was developed for participating in "[Endoscopic Vision Challenge  - Gastointetinal Image Analysis 2018](https://giana.grand-challenge.org)". And we won the 2nd place for polyp segmentation using DUCHDC model. 

Here are the implemented models:
- [x] [FCN8]
- [x] [FCN16]
- [x] [FCN32]
- [x] [SegNet]
- [X] [PSPNet]
- [x] [UNet]
- [x] [Residual UNet]
- [x] [DUC]
- [x] [duchdc]
- [x] [LinkNet]
- [x] [FusionNet]
- [x] [GCN]


## Usage

For quick hints about commands:

```
python main.py -h
```

Modify the condiguration in the `settings.py` file

### Training

After customizing the `settings.py`, use the following command to start training
```
python main.py --cuda train
```
### Evaluation
For evaluation, put all your test images in a folder and set path in the `settings.py`. Then run the following command:
```
python main.py --cuda eval
```
The results will be place in the `results` directory


