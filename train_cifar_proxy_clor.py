"""
Code for MoCo pre-training

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

from lib.NCE import MemoryMoCo, NCESoftmaxLoss, ProxyClassOracleMemoryMoCo
from lib.dataset import ImageFolderInstance
#from lib.models.resnet import resnet50
from lib.models.resnet_cifar import ResNet18
from lib.models.wrn import wrn
from lib.util import adjust_learning_rate, AverageMeter, check_dir, DistributedShufle, set_bn_train, moment_update

def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    # exp name
    parser.add_argument('--exp-name', type=str, default='exp',
                        help='experiment name, used to determine checkpoint/tensorboard dir')

    # optimization
    parser.add_argument('--learning-rate', '--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--lr-decay-epochs', type=int, default=[120, 160, 200], nargs='+',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr-decay-rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')

    # root folders
    parser.add_argument('--data-root', type=str, default='./data', help='root directory of dataset')
    parser.add_argument('--output-root', type=str, default='./output', help='root directory for output')

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--crop', type=float, default=0.2, help='minimum crop')
    parser.add_argument('--aug', type=str, default='CJ', choices=['NULL', 'CJ'])

    # resume
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    # model definition
    parser.add_argument('--model', type=str, default='resnet50', choices=['resnet50'])
    parser.add_argument('--model-width', type=int, default=1, help='width of resnet, eg, 1, 2, 4')
    parser.add_argument('--alpha', type=float, default=0.999, help='exponential moving average weight')

    # loss function
    parser.add_argument('--nce-k', type=int, default=4096)
    parser.add_argument('--nce-t', type=float, default=0.07)

    # supervised options
    parser.add_argument('--sup-n-samples', type=int, default=250)
    parser.add_argument('--sup-lr', type=float, default=0.01)
    parser.add_argument('--sup-rate', type=float, default=0.3)
    parser.add_argument("--sup-head-only", action="store_true")
    parser.add_argument('--sup-bs', type=int, default=None)
    parser.add_argument('--proxy-threshold', type=float, default=0.5)

    # misc
    parser.add_argument('--print-freq', type=int, default=10, help='print frequency')
    parser.add_argument('--tb-freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save-freq', type=int, default=10, help='save frequency')
    parser.add_argument('--batch-size', type=int, default=32, help='batch_size')
    parser.add_argument('--num-workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000, help='number of training epochs')
    parser.add_argument('--start-epoch', type=int, default=1, help='used for resume')
    parser.add_argument("--local_rank", type=int)

    args = parser.parse_args()

    # set the path according to the environment
    output_dir = check_dir(os.path.join(args.output_root, args.dataset, args.exp_name))
    args.model_folder = check_dir(os.path.join(output_dir, 'models'))
    args.tb_folder = check_dir(os.path.join(output_dir, 'tensorboard'))

    if args.dataset == "cifar10":
        args.num_classes = 10
    else:
        raise NotImplementedError

    if args.sup_lr is None:
        args.sup_lr = args.learning_rate

    if args.sup_bs is None:
        args.sup_bs = args.batch_size

    return args


class MyRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)


class TransformTwice:
    def __init__(self, transform, aug_transform):
        self.transform = transform
        self.aug_transform = aug_transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.aug_transform(inp)
        return out1, out2

class RandomTranslateWithReflect:
    '''
    Translate image randomly
    Translate vertically and horizontally by n pixels where
    n is integer drawn uniformly independently for each axis
    from [-max_translation, max_translation].
    Fill the uncovered blank area with reflect padding.
    '''

    def __init__(self, max_translation):
        self.max_translation = max_translation

    def __call__(self, old_image):
        xtranslation, ytranslation = np.random.randint(-self.max_translation,
                                                       self.max_translation + 1,
                                                       size=2)
        xpad, ypad = abs(xtranslation), abs(ytranslation)
        xsize, ysize = old_image.size

        flipped_lr = old_image.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_tb = old_image.transpose(Image.FLIP_TOP_BOTTOM)
        flipped_both = old_image.transpose(Image.ROTATE_180)

        new_image = Image.new("RGB", (xsize + 2 * xpad, ysize + 2 * ypad))

        new_image.paste(old_image, (xpad, ypad))

        new_image.paste(flipped_lr, (xpad + xsize - 1, ypad))
        new_image.paste(flipped_lr, (xpad - xsize + 1, ypad))

        new_image.paste(flipped_tb, (xpad, ypad + ysize - 1))
        new_image.paste(flipped_tb, (xpad, ypad - ysize + 1))

        new_image.paste(flipped_both, (xpad - xsize + 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad - xsize + 1, ypad + ysize - 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad + ysize - 1))

        new_image = new_image.crop((xpad - xtranslation,
                                    ypad - ytranslation,
                                    xpad + xsize - xtranslation,
                                    ypad + ysize - ytranslation))
        return new_image

def get_loader(args):
    # set the data loader
    #train_folder = os.path.join(args.data_root, 'train')

    #image_size = 224
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])
    rotation_transform = MyRotationTransform(angles=[-90, 0, 90, 180])

    #color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
    #rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    #rnd_gray = transforms.RandomGrayscale(p=0.2)

    col_jitter = transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8)
    img_jitter = transforms.RandomApply([RandomTranslateWithReflect(4)], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.25)

    transform_ori = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    
    transform_amdim = transforms.Compose([
        img_jitter,
        col_jitter,
        rnd_gray,
        transforms.ToTensor(),
        normalize
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
    #train_dataset = ImageFolderInstance(train_folder, transform=train_transform, two_crop=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        sampler=train_sampler, drop_last=True)

    # labeled sampler/loader
    train_ids = np.arange(len(train_dataset))
    np.random.shuffle(train_ids)
    mask = np.zeros(train_ids.shape[0], dtype=np.bool)
    labels = np.array([train_dataset[i][1] for i in train_ids], dtype=np.int64)
    for i in range(args.num_classes):
        mask[np.where(labels == i)[0][: int(args.sup_n_samples / args.num_classes)]] = True
        labeled_indices = train_ids[mask]
    train_sampler_lab = torch.utils.data.sampler.SubsetRandomSampler(labeled_indices)
    sup_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.sup_bs, shuffle=False,
                                             sampler=train_sampler_lab, num_workers=args.num_workers, drop_last=True)

    return train_loader, sup_loader


def build_model(args):
    model = ResNet18().cuda()
    model_ema = ResNet18().cuda()
    print("Built feature extractors")
    classifier = torch.nn.Linear(128, 10).cuda()
    print("Built classifier")

    # copy weights from `model' to `model_ema'
    moment_update(model, model_ema, 0)

    return model, model_ema, classifier


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
    train_loader, sup_loader = get_loader(args)
    n_data = len(train_loader.dataset)
    if args.local_rank == 0:
        print(f"length of training dataset: {n_data}")

    open(os.path.join(args.tb_folder, os.environ.pop("PBS_JOBID", "dbg")), "a").close()
    model, model_ema, classifier = build_model(args)
    contrast = ProxyClassOracleMemoryMoCo(10, 128, args.nce_k, temperature=args.nce_t).cuda()
    criterion = NCESoftmaxLoss().cuda()
    sup_criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD([{'params': model.parameters()},
                                 {'params': classifier.parameters(), "lr": args.sup_lr}],
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs*(n_data/args.batch_size), eta_min=0.0001)

    model = DistributedDataParallel(model, device_ids=[args.local_rank], broadcast_buffers=False)

    # optionally resume from a checkpoint
    if args.resume:
        assert os.path.isfile(args.resume)
        load_checkpoint(args, model, model_ema, contrast, optimizer)

    # tensorboard
    logger = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)

    # routine
    for epoch in range(args.start_epoch, args.epochs + 1):
        train_loader.sampler.set_epoch(epoch)
        #adjust_learning_rate(epoch, args, optimizer)
        
        if epoch == 1:    
            save_checkpoint(args, epoch, model, model_ema, contrast, optimizer)

        tic = time.time()
        nce_loss, prob, sup_loss, percent_proxy = train_moco(epoch, train_loader, sup_loader, model, model_ema,
                                                             classifier, contrast, criterion, sup_criterion,
                                                             optimizer, scheduler, args)

        if args.local_rank == 0:
            print('epoch {}, total time {:.2f}'.format(epoch, time.time() - tic))

            # tensorboard logger
            logger.log_value('nce_loss', nce_loss, epoch)
            logger.log_value('ce_loss', sup_loss, epoch)
            logger.log_value('ins_prob', prob, epoch)
            logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)
            logger.log_value('percent_proxy_labels', percent_proxy, epoch)

            # save model
            save_checkpoint(args, epoch, model, model_ema, contrast, optimizer)


def train_moco(epoch, train_loader, sup_loader, model, model_ema, classifier, contrast, criterion, sup_criterion, optimizer, scheduler, args):
    """
    one epoch training for moco
    """
    model.train()
    set_bn_train(model_ema)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    nce_loss_meter = AverageMeter()
    sup_loss_meter = AverageMeter()
    prob_meter = AverageMeter()

    # dbg meters
    percent_proxy_meter = AverageMeter()

    sup_iter = iter(sup_loader)
    end = time.time()
    for idx, ((x1, x2), _) in enumerate(train_loader):
        data_time.update(time.time() - end)

        #bsz = inputs.size(0)
        bsz = x1.size(0)

        # forward
        #x1, x2 = torch.split(inputs, [3, 3], dim=1)
        x1.contiguous()
        x2.contiguous()
        x1 = x1.cuda(non_blocking=True)
        x2 = x2.cuda(non_blocking=True)

        feat_q = model(x1)
        with torch.no_grad():
            # x2_shuffled, backward_inds = DistributedShufle.forward_shuffle(x2, epoch)
            # feat_k = model_ema(x2_shuffled)
            # feat_k_all, feat_k = DistributedShufle.backward_shuffle(feat_k, backward_inds, return_local=True)
            feat_k = model_ema(x2)

        # classifier/proxy forward
        try:
            (sup_x1, sup_x2), sup_l = next(sup_iter)
        except StopIteration:
            sup_iter = iter(sup_loader)  # restart sampling. is this ok?
            (sup_x1, sup_x2), sup_l = next(sup_iter)
        cls_feat = model(sup_x1.cuda())
        if args.sup_head_only:
            cls_feat = cls_feat.detach()
        sup_pred = classifier(cls_feat)
        sup_loss = sup_criterion(sup_pred, sup_l.cuda())

        proxy_labels = sup_pred.argmax(dim=-1)
        # print()
        # print(proxy_labels)
        # print(sup_pred.softmax(dim=-1).max(dim=-1)[0])
        proxy_labels[sup_pred.softmax(dim=-1).max(dim=-1)[0] < args.proxy_threshold] = -1.
        # print(proxy_labels)

        # print ('feat_k: ', feat_k.size(), ' feat_k_all: ', feat_k_all.size(), ' feat_q: ',  feat_q.size())
        out = contrast(feat_q, feat_k, feat_k, proxy_labels)
        nce_loss = criterion(out)
        prob = F.softmax(out, dim=1)[:, 0].mean()

        loss = nce_loss*(1-args.sup_rate) + sup_loss*args.sup_rate

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        moment_update(model, model_ema, args.alpha)

        # update meters
        nce_loss_meter.update(nce_loss.item(), bsz)
        sup_loss_meter.update(sup_loss.item(), bsz)
        prob_meter.update(prob.item(), bsz)
        percent_proxy_meter.update((proxy_labels != -1).float().mean().item())
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if args.local_rank == 0 and idx % args.print_freq == 0:
            print(f'Train: [{epoch}][{idx}/{len(train_loader)}]\t'
                  f'T {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  f'nce loss {nce_loss_meter.val:.3f} ({nce_loss_meter.avg:.3f})\t'
                  f'ce loss {sup_loss_meter.val:.3f} ({sup_loss_meter.avg:.3f})\t'
                  f'prob {prob_meter.val:.3f} ({prob_meter.avg:.3f})\t'
                  f'percent proxies {percent_proxy_meter.val:.3f} {percent_proxy_meter.avg:.3f}')

    return nce_loss_meter.avg, prob_meter.avg, sup_loss_meter.avg, percent_proxy_meter.avg


if __name__ == '__main__':
    opt = parse_option()
    if opt.local_rank == 0:
        pprint(vars(opt))

    os.environ['MASTER_PORT'] = str(random.randrange(1000, 5000))
    torch.cuda.set_device(opt.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    cudnn.benchmark = True

    main(opt)
