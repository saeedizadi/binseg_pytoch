import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
import glob
from tqdm import tqdm
import os
from PIL import Image
from utils.dataset import GIANA
import utils.transforms as t
import utils.co_transforms as cot
from models.unet import (UNet, UNetRes)
from models.fcn import (FCN32, FCN16, FCN8)
from models.pspnet import PSPNet
from models.segnet import SegNet
from models.gcn import ResnetGCN
from models.fusionnet import FusionNet
from models.linknet import LinkNet
from models.duc import (ResentDUC, ResentDUCHDC)
from models.discriminator import Discriminator
from utils.avg import AverageMeter
from utils.evaluation import Evaluation
from utils.visualize import Dashboad
from utils.losses import *
from utils.experiments import Experiments
from utils.tools import morph_postprocess
from settings import get_arguments
import sys
import itertools

def load_data(args):

    normalize = t.Normalize(mean=[0.445, 0.287, 0.190], std=[0.31, 0.225, 0.168])
    im_transform = t.Compose([t.ToTensor(), normalize])

    # Use  the following code fo co_transformations e.g. random rotation or random flip etc.
    # co_transformer = cot.Compose([cot.RandomRotate(45)])

    dsetTrain = GIANA(args.imgdir, args.gtdir, input_size=(args.input_width, args.input_height) ,train=True, transform=im_transform, co_transform=None, target_transform=t.ToLabel())
    train_data_loader = data.DataLoader(dsetTrain, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    dsetVal = GIANA(args.imgdir, args.gtdir, train=False, transform=im_transform, co_transform=None,
                    target_transform=t.ToLabel())
    val_data_loader = data.DataLoader(dsetVal, batch_size=args.batch_size, shuffle=False,
                                      num_workers=args.num_workers)
    return train_data_loader, val_data_loader


def train(args, model):
    board = Dashboad(args.visdom_port)
    tr_losses = AverageMeter()
    tLoader, vLoader = load_data(args)

    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.99)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    for epoch in range(1, args.num_epochs+1):
        scheduler.step()
        if epoch == 1:
            tr_loss, _ , _ = evaluate(args, model, criterion, tLoader)
            vl_loss, vl_jacc, vl_dice = evaluate(args, model, criterion, vLoader)

            # Draw the loss curves
            win = None
            win = board.loss_curves([tr_loss, vl_loss], epoch, win=win)
            print('[Initial TrainLoss: {0:.4f}]'
                  '\t[Initial ValidationLoss: {1:.4f}]'
                  '\t[Initial ValidationJaccard: {2:.4f}]'
                  '\t[Initial ValidationDice: {3:.4f}]'.format(tr_loss, vl_loss, vl_jacc, vl_dice))
            print('----------------------------------------------------------------------------------------------------'
                  '--------------')

        for step, (images, labels) in enumerate(tLoader):
            model.train(True)
            if args.cuda:
                images = images.cuda()
                labels = labels.cuda()
                criterion = criterion.cuda()

            inputs = Variable(images)
            targets = Variable(labels)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            tr_losses.update(loss.data.cpu().numpy())

        if epoch % args.log_step == 0:
            vl_loss, vl_jacc, vl_dice = evaluate(args, model, criterion, vLoader)
            print('[Epoch: {0:02}/{1:02}]'
                  '\t[TrainLoss: {2:.4f}]'
                  '\t[ValidationLoss: {3:.4f}]'
                  '\t[ValidationJaccard: {4:.4f}]'
                  '\t[ValidationDice: {5:.4f}]'.format(epoch, args.num_epochs, tr_losses.avg, vl_loss, vl_jacc,
                                                       vl_dice)),

            filename = "weights/{0}-{1:02}.pth".format(args.model, epoch)
            torch.save(model.state_dict(), filename)
            print('  [Snapshot]')
        else:
            vl_loss, vl_jacc, vl_dice = evaluate(args, model, criterion, vLoader)
            print('[Epoch: {0:02}/{1:02}]'
                  '\t[TrainLoss: {2:.4f}]'
                  '\t[ValidationLoss: {3:.4f}]'
                  '\t[ValidationJaccard: {4:.4f}]'
                  '\t[ValidationDice: {5:.4f}]'.format(epoch, args.num_epochs, tr_losses.avg, vl_loss, vl_jacc,
                                                       vl_dice))

        # --- Update the loss curves ---
        win = board.loss_curves([tr_losses.avg, vl_loss], epoch, win=win)


def evaluate(args, model, criterion, val_loader):
    model.eval()
    losses = AverageMeter()
    jaccars = AverageMeter()
    dices = AverageMeter()
    eva = Evaluation()

    for i, (images, labels) in enumerate(val_loader):
        if args.cuda:
            images = images.cuda()
            labels = labels.cuda()
            criterion = criterion.cuda()

        images = Variable(images)
        labels = Variable(labels)
        outputs = model(images)
        loss = criterion(outputs, labels)
        losses.update(loss.data.cpu().numpy())

        jacc_index = eva.jaccard_similarity_coefficient(outputs.cpu().data.numpy().squeeze(),
                                                        labels.cpu().data.numpy())
        dice_index = eva.dice_coefficient(outputs.cpu().data.numpy().squeeze(),
                                                        labels.cpu().data.numpy())
        jaccars.update(jacc_index)
        dices.update(dice_index)

    return losses.avg, jaccars.avg, dices.avg


def eval(args, model):
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
    
    listImgFiles = [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join(args.imgdir, '*.bmp'))]
    for currFile in tqdm(listImgFiles):
        img = Image.open(os.path.join(args.imgdir, currFile + '.bmp'))
        img = t.ToTensor()(img)
        if args.cuda:
            img.cuda()
        output = model(Variable(img, volatile=True).unsqueeze(0))
        output = t.ToPILImage()(output[0].cpu().data)
        newfilename = os.path.join(args.savedir, currFile + '_' + args.model + '.bmp')
        output.save(newfilename, 'BMP')

def make_model(args):
    Net = None
    if args.model == 'unet':
        Net = UNet
    elif args.model == 'unet_res':
        Net = UNetRes
    elif args.model == 'fcn32':
        Net = FCN32
    elif args.model == 'fcn16':
        Net = FCN16
    elif args.model == 'fcn8':
        Net = FCN8
    elif args.model == 'pspnet':
        Net = PSPNet
    elif args.model == 'segnet':
        Net = SegNet
    elif args.model == 'fusionnet':
        Net = FusionNet
    elif args.model == 'duc':
        Net = ResentDUC
    elif args.model == 'duchdc':
        Net = ResentDUCHDC
    elif args.model == 'gcn':
        Net = ResnetGCN
    elif args.model == 'linknet':
        Net = LinkNet

    assert Net is not None, 'model {args.model} is not available'

    model = Net(1)

    if args.cuda:
        nGPUs = torch.cuda.device_count()
        model = torch.nn.DataParallel(model, device_ids=range(nGPUs)).cuda()
        model = model.cuda()
        if args.state:
            model.load_state_dict(torch.load('weights/' + args.state))

    # In case to use weights trained on multiple GPUs in CPU mode
    else:
        checkpoint = torch.load('./weights/' + args.state)
        state_dict = checkpoint['state_dict']
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[0:9] + k[16:]  # remove `module.`
            if k[0] == 'f':
                new_state_dict[name] = v
            else:
                new_state_dict[k] = v

        model.load_state_dict(new_state_dict)
        model = model.cpu()
    return model


def main(args):
    torch.backends.cudnn.enabled = False

    model = make_model(args)
    if args.mode == 'train':
        train(args, model)
    elif args.mode == 'eval':
        eval(args, model)


if __name__ == '__main__':

    args = get_arguments(sys.argv[1:])

    if args.mode == 'crossval':
        params = list(itertools.product(args.batch_size, args.optimizer, args.momentum, args.lr))
        args.batch_size, args.optimizer, args.momentum, args.lr = params[args.settings_id]

    main(args)
