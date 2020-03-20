"""
Code for MoCo pre-training

MoCo: Momentum Contrast for Unsupervised Visual Representation Learning

"""
import argparse
import os
import time
import json
from pprint import pprint

from PIL import Image
import numpy as np
import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
import random

from lib.augment.cutout import Cutout
from lib.augment.autoaugment_extra import CIFAR10Policy

from lib.NCE import MemoryMoCo, NCESoftmaxLoss, ClassOracleMemoryMoCo
from lib.NCE.Contrast import MoCoNet

from lib.models.wrn import wrn
from lib.util import adjust_learning_rate, AverageMeter, check_dir, DistributedShufle, set_bn_train, moment_update
from sklearn.metrics import roc_auc_score

from lib.data import train_transform, rescale_images

global_step = 0
data_root = "/misc/lmbraid19/galessos/datasets/"


def parse_option():
    parser = argparse.ArgumentParser('arguments for training', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # task
    parser.add_argument('dataset', type=str)

    # exp name
    parser.add_argument('--exp-name', type=str, default='exp',
                        help='experiment name, used to determine checkpoint/tensorboard dir')

    # optimization
    parser.add_argument('--learning-rate', '--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--batch-size', '--bs', type=int, default=32, help='batch_size')
    parser.add_argument('--lr-decay-epochs', type=int, default=[120, 160, 200], nargs='+',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr-decay-rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight-decay', '--wd', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')

    # root folders
    parser.add_argument('--data-root', type=str, default='./data', help='root directory of dataset')
    parser.add_argument('--output-root', type=str,
                        default='/misc/lmbraid18/galessos/experiments/contrastive_unsup_learning/output',
                        help='root directory for output')

    # dataset
    parser.add_argument('--crop', type=float, default=0.2, help='minimum crop')
    parser.add_argument('--aug', type=str, default='CJ', choices=['NULL', 'CJ'])

    # resume
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    # model definition
    parser.add_argument('--model', type=str, default='ResNet18', choices=['ResNet18', 'ResNet50'])
    parser.add_argument('--model-width', type=int, default=1, help='width of resnet, eg, 1, 2, 4')
    parser.add_argument('--alpha', type=float, default=0.999, help='exponential moving average weight')

    # loss function
    parser.add_argument('--nce-k', type=int, default=2048)
    parser.add_argument('--nce-t', type=float, default=0.3)
    parser.add_argument('--class-oracle', action="store_true")

    # misc
    parser.add_argument('--print-freq', type=int, default=10, help='print frequency')
    parser.add_argument('--tb-freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save-freq', type=int, default=10, help='save frequency')
    parser.add_argument('--num-workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000, help='number of training epochs')
    parser.add_argument('--start-epoch', type=int, default=1, help='used for resume')
    parser.add_argument("--local_rank", type=int)

    args = parser.parse_args()

    # set the path according to the environment
    output_dir = check_dir(os.path.join(args.output_root, args.dataset, args.exp_name))
    args.model_folder = check_dir(os.path.join(output_dir, 'models'))
    args.tb_folder = check_dir(os.path.join(output_dir, 'tensorboard'))

    return args


def get_train_loader(args):
    import re
    if args.dataset.lower() == "cifar10":
        train_in_data = datasets.CIFAR10(data_root, train=True, transform=train_transform, download=True)
    elif args.dataset.lower() == "cifar100":
        train_in_data = datasets.CIFAR100(data_root, train=True, transform=train_transform, download=True)
    elif re.match("^cifar10_([0-9]+)$", args.dataset):
        classes = re.match("^cifar10_([0-9]+)$", args.dataset)[1]
        classes = [int(c) for c in classes]
        dataset = datasets.CIFAR10(data_root, train=True, transform=train_transform, download=True)
        train_in_data = torch.utils.data.Subset(dataset, [i for i, (_, l) in enumerate(dataset) if l in classes])
    elif re.match("^cifar80-20_set([0-9]+)$", args.dataset):
        with open(os.path.join(data_root, "cifar80-20", "{}.json".format(args.dataset)), "r") as set_file:
            classes = json.load(set_file)['cifar80']
        dataset = datasets.CIFAR100(data_root, train=True, transform=train_transform, download=True)
        train_in_data = torch.utils.data.Subset(dataset, [i for i, (_, l) in enumerate(dataset) if l in classes])
    else:
        raise NotImplementedError

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_in_data)
    train_loader = torch.utils.data.DataLoader(
        train_in_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        sampler=train_sampler, drop_last=True)
    return train_loader


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
    torch.distributed.barrier()


def save_checkpoint(args, epoch, model, optimizer):
    print('==> Saving...')
    state = {
        'opt': args,
        'model': model.state_dict(),
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
    # train_loader, in_val_loader, out_val_loaders = get_ood_loaders(args)
    train_loader = get_train_loader(args)
    n_data = len(train_loader.dataset)
    if args.local_rank == 0:
        print(f"length of training dataset: {n_data}")

    model = MoCoNet(args.alpha, 128, args.nce_k, temperature=args.nce_t)
    criterion = NCESoftmaxLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay) 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs*(n_data/args.batch_size),
                                                           eta_min=0.0001)

    model = DistributedDataParallel(model, device_ids=[args.local_rank],
                                    broadcast_buffers=False, find_unused_parameters=True)

    # optionally resume from a checkpoint
    if args.resume:
        assert os.path.isfile(args.resume)
        load_checkpoint(args, model, optimizer)

    # tensorboard
    log = SummaryWriter(args.tb_folder, flush_secs=2)

    # routine
    for epoch in range(args.start_epoch, args.epochs + 1):
        train_loader.sampler.set_epoch(epoch)
        #adjust_learning_rate(epoch, args, optimizer)
        
        if epoch == 1:    
            save_checkpoint(args, epoch, model, optimizer)

        tic = time.time()
        # prob_in, prob_out, auroc = val_ood(epoch, in_val_loader, out_val_loaders, model, criterion, args, log=log)
        loss, prob = train_moco(epoch, train_loader, model, criterion, optimizer, scheduler, args, log=log)

        if args.local_rank == 0:
            print('epoch {}, total time {:.2f}'.format(epoch, time.time() - tic))

            # tensorboard logger
            log.add_scalar('ins_loss', loss, epoch)
            log.add_scalar('ins_prob', prob, epoch)
            log.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
            # log.add_scalar('val_prob_in', prob_in, epoch)
            # log.add_scalar('val_prob_out', prob_out, epoch)
            # log.add_scalar('auroc', auroc, epoch)

            # save model
            save_checkpoint(args, epoch, model, optimizer)


def train_moco(epoch, train_loader, model, criterion, optimizer, scheduler, args, log=None):
    """
    one epoch training for moco
    """
    global global_step
    model.train()
    # set_bn_train(model_ema)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    prob_meter = AverageMeter()

    end = time.time()
    for idx, ((x1, x2), l) in enumerate(train_loader):
        data_time.update(time.time() - end)

        bsz = x1.size(0)

        # forward
        x1.contiguous()
        x2.contiguous()
        x1 = x1.cuda(non_blocking=True)
        x2 = x2.cuda(non_blocking=True)

        out = model(x1, x2)

        loss = criterion(out)
        prob = F.softmax(out, dim=1)[:, 0].mean()  # TODO bring this to the MoCo class?

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        model.module.momentum_update()

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

            log.add_images('training_samples_x1', rescale_images(x1), global_step)
            log.add_images('training_samples_x2', rescale_images(x2), global_step)

        global_step += 1

    return loss_meter.avg, prob_meter.avg


if __name__ == '__main__':
    opt = parse_option()
    if opt.local_rank == 0:
        pprint(vars(opt))

    os.environ['MASTER_PORT'] = str(random.randrange(1000, 5000))
    torch.cuda.set_device(opt.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    cudnn.benchmark = True

    main(opt)


# # *** CODE GRAVEYARD ***
# def get_ood_loaders(args):
#     train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
#                                           transforms.RandomHorizontalFlip(),
#                                           transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8),
#                                           transforms.RandomGrayscale(p=0.25),
#                                           transforms.ToTensor(),
#                                           transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
#                                                                std=[0.2471, 0.2435, 0.2616])])
#     val_transform = transforms.Compose([transforms.ToTensor(),
#                                         transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
#                                                              std=[0.2471, 0.2435, 0.2616])])
#
#     train_transform = TransformTwice(train_transform, train_transform)
#     val_transform = TransformTwice(val_transform, val_transform)
#
#     if args.dataset == 'svhn':
#         from torchvision.datasets import CIFAR10, SVHN
#         train_in_data = CIFAR10('./data', train=True, transform=train_transform, download=True)
#         val_in_data = CIFAR10('./data', train=False, transform=val_transform, download=True)
#         val_out_data = SVHN('./data', split='test', transform=val_transform, download=True)
#
#     elif 'cifar10_' in args.dataset:
#         if args.dataset == "cifar10_animals":
#             positive_classes = [2, 3, 4, 5, 6, 7]
#         else:
#             positive_classes = [int(d) for d in args.dataset.replace('cifar10_', '')]
#         negative_classes = [c for c in list(range(10)) if c not in positive_classes]
#         train_in_data = CIFAR10ClassSelect(root='./data', positive_classes=positive_classes, train=True,
#                                            download=False, transform=train_transform)
#         val_in_data = CIFAR10ClassSelect(root='./data', positive_classes=positive_classes, train=False,
#                                          download=False, transform=val_transform)
#         val_out_data = CIFAR10ClassSelect(root='./data', positive_classes=negative_classes, train=False,
#                                           download=False, transform=val_transform)
#     else:
#         raise NotImplementedError
#
#     train_sampler = torch.utils.data.distributed.DistributedSampler(train_in_data)
#     train_loader = torch.utils.data.DataLoader(
#         train_in_data, batch_size=args.batch_size, shuffle=False,
#         num_workers=args.num_workers, pin_memory=True,
#         sampler=train_sampler, drop_last=True)
#
#     in_val_sampler = torch.utils.data.distributed.DistributedSampler(val_in_data)
#     in_val_loader = torch.utils.data.DataLoader(
#         val_in_data, batch_size=64, shuffle=False,
#         num_workers=args.num_workers, pin_memory=True,
#         sampler=in_val_sampler, drop_last=True)
#
#     out_val_sampler = torch.utils.data.distributed.DistributedSampler(val_out_data)
#     out_val_loader = torch.utils.data.DataLoader(
#         val_out_data, batch_size=64, shuffle=False,
#         num_workers=args.num_workers, pin_memory=True,
#         sampler=out_val_sampler, drop_last=True)
#
#     return train_loader, in_val_loader, out_val_loader
