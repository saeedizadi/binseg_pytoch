Implementation of several state of the art segmentation method for binary tasks in PyTorch

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


This doc will be improved
