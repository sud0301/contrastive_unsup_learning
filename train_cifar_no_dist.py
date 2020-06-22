"""
Code for MoCo pre-training for CIFAR-10

MoCo: Momentum Contrast for Unsupervised Visual Representation Learning

"""
import argparse
import os
import time
from pprint import pprint

from PIL import Image
import numpy as np
import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torchvision import transforms
import torchvision
import torchvision.transforms.functional as TF
import random

from lib.augment.cutout import Cutout
from lib.augment.autoaugment_extra import CIFAR10Policy

from lib.NCE import MemoryMoCo, NCESoftmaxLoss
from lib.dataset import ImageFolderInstance
 
from lib.models.resnet_cifar import ResNet18
from lib.models.wrn import wrn
from lib.util import adjust_learning_rate, AverageMeter, check_dir, DistributedShufle, set_bn_train, moment_update, RandomTranslateWithReflect

def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    # exp name
    parser.add_argument('--exp-name', type=str, default='exp',
                        help='experiment name, used to determine checkpoint/tensorboard dir')

    # optimization
    parser.add_argument('--learning-rate', '--lr', type=float, default=0.003, help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')

    # root folders
    parser.add_argument('--output-root', type=str, default='./output', help='root directory for output')

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar10')

    # resume
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    # model definition
    parser.add_argument('--model-width', type=int, default=1, help='width of resnet, eg, 1, 2, 4')
    parser.add_argument('--alpha', type=float, default=0.999, help='exponential moving average weight')

    # loss function
    parser.add_argument('--nce-k', type=int, default=4096)
    parser.add_argument('--nce-t', type=float, default=0.3)

    # misc
    parser.add_argument('--print-freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb-freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save-freq', type=int, default=10, help='save frequency')
    parser.add_argument('--batch-size', type=int, default=32, help='batch_size')
    parser.add_argument('--num-workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')
    parser.add_argument('--start-epoch', type=int, default=1, help='used for resume')
    parser.add_argument("--local_rank", type=int)

    args = parser.parse_args()

    # set the path according to the environment
    output_dir = check_dir(os.path.join(args.output_root, args.dataset, args.exp_name))
    args.model_folder = check_dir(os.path.join(output_dir, 'models'))
    args.tb_folder = check_dir(os.path.join(output_dir, 'tensorboard'))

    return args

class TransformTwice:
    def __init__(self, transform, aug_transform):
        self.transform = transform
        self.aug_transform = aug_transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.aug_transform(inp)
        return out1, out2

def get_loader(args):

    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])

    col_jitter = transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8)
    img_jitter = transforms.RandomApply([RandomTranslateWithReflect(4)], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.25)

    transform_ori = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    
    transform_aug = transforms.Compose([
        transforms.ToTensor(),
        Cutout(n_holes=1, length=16),
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        CIFAR10Policy(),
        img_jitter,
        col_jitter,
        rnd_gray,
        transforms.ToTensor(),
        normalize,
    ])

    transform_train = TransformTwice(transform_aug, transform_aug)

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
   
    # select samples from 5 classes 
    train_dataset_size = len(train_dataset)
    train_ids = np.arange(train_dataset_size)
    
    mask = np.zeros(train_ids.shape[0], dtype=np.bool)
    labels = np.array([train_dataset[i][1] for i in train_ids], dtype=np.int64)

    class_index = [i for i in range(0, 10)]
    random.seed(1993)
    random.shuffle(class_index)
    
    for i in class_index[:5]:
        mask[np.where(labels == i)[0]] = True
   
    labeled_indices = train_ids[mask]
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(labeled_indices)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        sampler=train_sampler, drop_last=True)

    return train_loader


def build_model(args):
    model = ResNet18().cuda()
    model_ema = ResNet18().cuda()

    # copy weights from `model' to `model_ema'
    moment_update(model, model_ema, 0)

    return model, model_ema


def load_checkpoint(args, model, model_ema, contrast, optimizer):
    if args.local_rank == 0:
        print("=> loading checkpoint '{}'".format(args.resume))

    checkpoint = torch.load(args.resume, map_location='cpu')
    args.start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['model'])
    model_ema.load_state_dict(checkpoint['model_ema'])
    contrast.load_state_dict(checkpoint['contrast'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    if args.local_rank == 0:
        print("=> loaded successfully '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

    del checkpoint
    torch.cuda.empty_cache()

    # make sure all process have loaded the checkpoint
    #torch.distributed.barrier()


def save_checkpoint(args, epoch, model, model_ema, contrast, optimizer):
    print('==> Saving...')
    state = {
        'opt': args,
        'model': model.state_dict(),
        'model_ema': model_ema.state_dict(),
        'contrast': contrast.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, os.path.join(args.model_folder, 'current.pth'))
    if epoch % args.save_freq == 0 or epoch ==1:
        torch.save(state, os.path.join(args.model_folder, f'ckpt_epoch_{epoch}.pth'))
    # help release GPU memory
    del state
    torch.cuda.empty_cache()


def main(args):
    train_loader = get_loader(args)
    n_data = len(train_loader.dataset)
    if args.local_rank == 0:
        print(f"length of training dataset: {n_data}")

    model, model_ema = build_model(args)
    contrast = MemoryMoCo(128, args.nce_k, args.nce_t).cuda()
    criterion = NCESoftmaxLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay) 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs*(n_data/args.batch_size), eta_min=0.0001)

    # optionally resume from a checkpoint
    if args.resume:
        assert os.path.isfile(args.resume)
        load_checkpoint(args, model, model_ema, contrast, optimizer)

    # tensorboard
    logger = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)

    # routine
    for epoch in range(args.start_epoch, args.epochs + 1):
        
        if epoch == 1:    
            save_checkpoint(args, epoch, model, model_ema, contrast, optimizer)

        tic = time.time()
        loss, prob = train_moco(epoch, train_loader, model, model_ema, contrast, criterion, optimizer, scheduler, args)

        if args.local_rank == 0:
            print('epoch {}, total time {:.2f}'.format(epoch, time.time() - tic))

            # tensorboard logger
            logger.log_value('ins_loss', loss, epoch)
            logger.log_value('ins_prob', prob, epoch)
            logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

            # save model
            save_checkpoint(args, epoch, model, model_ema, contrast, optimizer)


def train_moco(epoch, train_loader, model, model_ema, contrast, criterion, optimizer, scheduler, args):
    """
    one epoch training for moco
    """
    model.train()
    set_bn_train(model_ema)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    prob_meter = AverageMeter()

    end = time.time()
    for idx, ((x1, x2), _) in enumerate(train_loader):
        data_time.update(time.time() - end)

        bsz = x1.size(0)

        # forward
        x1.contiguous()
        x2.contiguous()
        x1 = x1.cuda(non_blocking=True)
        x2 = x2.cuda(non_blocking=True)

        feat_q = model(x1)
        with torch.no_grad():
            feat_k = model_ema(x2)

        out = contrast(feat_q, feat_k, feat_k)
        loss = criterion(out)
        prob = F.softmax(out, dim=1)[:, 0].mean()

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        moment_update(model, model_ema, args.alpha)

        # update meters
        loss_meter.update(loss.item(), bsz)
        prob_meter.update(prob.item(), bsz)
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if args.local_rank == 0 and idx % args.print_freq == 0:
            print(f'Train: [{epoch}][{idx}/{len(train_loader)}]\t'
                  f'T {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  f'loss {loss_meter.val:.3f} ({loss_meter.avg:.3f})\t'
                  f'prob {prob_meter.val:.3f} ({prob_meter.avg:.3f})')

    return loss_meter.avg, prob_meter.avg


if __name__ == '__main__':
    opt = parse_option()
    if opt.local_rank == 0:
        pprint(vars(opt))

    cudnn.benchmark = True

    main(opt)
