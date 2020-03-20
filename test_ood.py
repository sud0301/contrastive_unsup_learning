import argparse
import re
import os
import time
import json
import numpy as np
import torch
import torch.nn.functional as F
from pprint import pprint
from torchvision import datasets
from torchvision import transforms
from lib.NCE import NCESoftmaxLoss
from lib.NCE.Contrast import MoCoNet
from torch.utils.tensorboard import SummaryWriter
from lib.util import AverageMeter, check_dir
from lib.data import val_transform, rescale_images
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm


data_root = "/misc/lmbraid19/galessos/datasets/"
# val_transform = transforms.Compose([transforms.ToTensor(),
#                                     transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
#                                                          std=[0.2471, 0.2435, 0.2616])])


def fpr_at_tpr(scores, labels, tpr=0.95):
    fprs, tprs, _ = roc_curve(labels, scores)
    idx = np.argmin(np.abs(np.array(tprs)-tpr))
    return fprs[idx]


def parse_option():
    parser = argparse.ArgumentParser('arguments for validation', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # task
    parser.add_argument('in_dataset', type=str)
    parser.add_argument('ood_dataset', type=str)
    parser.add_argument('exp_name', type=str,
                        help='experiment name, used to determine checkpoint/tensorboard dir')
    parser.add_argument("--start-epoch", default=0, type=int, help='epoch to start testing from')

    # root folders
    parser.add_argument('--data-root', type=str, default="/misc/lmbraid19/galessos/datasets/",
                        help='root directory of dataset')
    parser.add_argument('--output-root', type=str,
                        default='/misc/lmbraid18/galessos/experiments/contrastive_unsup_learning/output',
                        help='root directory for output')

    # loss function
    # parser.add_argument('--nce-k', type=int, default=2048)
    # parser.add_argument('--nce-t', type=float, default=0.3)

    # misc
    parser.add_argument('--num-workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--batch-size', '--bs', type=int, default=64, help='test time batch size')

    args = parser.parse_args()

    # set the path according to the environment
    output_dir = check_dir(os.path.join(args.output_root, args.in_dataset, args.exp_name))
    args.model_dir = check_dir(os.path.join(output_dir, 'models'))
    # args.model_path = os.path.join(args.model_dir,
    #                                "ckpt_epoch_{}.pth".format(args.epoch) if args.epoch is not None
    #                                else "current.pth")
    args.tb_folder = check_dir(os.path.join(output_dir, 'tensorboard'))

    return args


def get_val_loaders(args):
    # IN DATASET
    if args.in_dataset.lower() == "cifar10":
        val_in_data = datasets.CIFAR10(data_root, train=False, transform=val_transform, download=True)
    elif args.in_dataset.lower() == "cifar100":
        val_in_data = datasets.CIFAR100(data_root, train=False, transform=val_transform, download=True)
    elif re.match("^cifar10_([0-9]+)$", args.in_dataset):
        classes = re.match("^cifar10_([0-9]+)$", args.in_dataset)[1]
        classes = [int(c) for c in classes]
        dataset = datasets.CIFAR10(data_root, train=False,
                                   transform=val_transform, download=True)
        val_in_data = torch.utils.data.Subset(dataset, [i for i, (_, l) in enumerate(dataset) if l in classes])
        print("In Validation data: {}, {} samples".format(args.in_dataset, len(val_in_data)))
    elif re.match("^cifar80-20_set([0-9]+)$", args.in_dataset):
        with open(os.path.join(data_root, "cifar80-20", "{}.json".format(args.in_dataset)), "r") as set_file:
            classes = json.load(set_file)['cifar80']
        dataset = datasets.CIFAR100(data_root, train=False, transform=val_transform, download=True)
        val_in_data = torch.utils.data.Subset(dataset, [i for i, (_, l) in enumerate(dataset) if l in classes])
        print("In Validation data: {}, {} samples".format(args.in_dataset, len(val_in_data)))
    else:
        raise NotImplementedError

    # OOD DATASET
    ood_datasets = {}
    if args.ood_dataset in ["non_cifar", "non-cifar"]:
        ood_datasets['svhn'] = datasets.SVHN(os.path.join(data_root, "SVHN"), split='test',
                                             transform=val_transform, download=True)
        for d in ["Imagenet_crop", "Imagenet_resize", "LSUN_crop", "LSUN_resize", "iSUN"]:
            ood_datasets[d] = datasets.ImageFolder(os.path.join(data_root, d), transform=val_transform)
    elif re.match("^cifar10_([0-9]+)$", args.ood_dataset):
        neg_classes = re.match("^cifar10_([0-9]+)$", args.ood_dataset)[1]
        if len([n for n in neg_classes if n in classes]) > 0:
            raise ValueError("Overlap between positive and negative classes!")
        neg_classes = [int(c) for c in neg_classes]
        dataset = datasets.CIFAR10(data_root, train=False, transform=val_transform, download=True)
        val_out_data = torch.utils.data.Subset(dataset, [i for i, (_, l) in enumerate(dataset) if l in neg_classes])
        ood_datasets[args.ood_dataset] = val_out_data
        print("OOD Validation data: {}, {} samples".format(args.ood_dataset, len(val_out_data)))
    elif re.match("^cifar80-20_set([0-9]+)$", args.ood_dataset):
        if "cifar80-20" in args.in_dataset and args.in_dataset != args.ood_dataset:
            raise ValueError("cifar80-20  sets must correspond!")
        with open(os.path.join(data_root, "cifar80-20", "{}.json".format(args.ood_dataset)), "r") as set_file:
            classes = json.load(set_file)['cifar20']
        dataset = datasets.CIFAR100(data_root, train=False, transform=val_transform, download=True)
        val_out_data = torch.utils.data.Subset(dataset, [i for i, (_, l) in enumerate(dataset) if l in classes])
        ood_datasets[args.ood_dataset] = val_out_data
        print("OOD Validation data: {}, {} samples".format(args.ood_dataset, len(val_out_data)))
    else:
        raise NotImplementedError

    val_in_loader = torch.utils.data.DataLoader(
        val_in_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=True)

    ood_loaders = {}
    for k, d in ood_datasets.items():
        val_out_loader = torch.utils.data.DataLoader(
            d, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True, drop_last=True)
        ood_loaders[k] = val_out_loader

    return val_in_loader, ood_loaders


def load_args(args):
    model_path = os.path.join(args.model_dir, "current.pth")
    checkpoint_opt = torch.load(model_path, map_location='cpu')["opt"]
    args.nce_k = checkpoint_opt.nce_k
    args.nce_t = checkpoint_opt.nce_t


def load_checkpoint(path, model):
    checkpoint = torch.load(path, map_location='cpu')
    epoch = checkpoint['epoch']
    checkpoint["model"] = {k.replace("module.", ""): v for k, v in checkpoint["model"].items()}
    model.load_state_dict(checkpoint['model'])

    del checkpoint
    torch.cuda.empty_cache()

    # make sure all process have loaded the checkpoint
    # torch.distributed.barrier()
    print("=> loaded checkpoint '{}'".format(path))

    return epoch


def main(args):
    val_in_loader, val_out_loaders = get_val_loaders(args)
    load_args(args)
    model = MoCoNet(1, 128, args.nce_k, temperature=args.nce_t).cuda()
    criterion = NCESoftmaxLoss().cuda()

    logs = {k: SummaryWriter(os.path.join(args.tb_folder, "val_{}".format(k))) for k in val_out_loaders.keys()}
    for epoch in range(args.start_epoch, 1000):
        try:
            model_path = os.path.join(args.model_dir, "ckpt_epoch_{}.pth".format(epoch))
            epoch = load_checkpoint(model_path, model)
        except FileNotFoundError:
            continue

        for k, val_out_loader in val_out_loaders.items():
            print("\nValidating {} on {}, epoch {}".format(args.exp_name, k, epoch))
            log = logs[k]

            # dbg_ood(epoch, val_in_loader, val_out_loader, model, criterion, args, log=None, name=k)

            prob_in, prob_out, auroc, fpr_at_95tpr = val_ood(epoch, val_in_loader, val_out_loader, model, criterion, args, log=log, name=k)
            # log.add_scalar('val_prob_in_'+k, prob_in, epoch)
            # log.add_scalar('val_prob_out_'+k, prob_out, epoch)
            # log.add_scalar('AUROC_{}_{}'.format(k, epoch), auroc, epoch)
            # log.add_scalar('FPR@95TPR_{}'.format(k), fpr_at_95tpr, epoch)
            print("AUROC = {:.5f}, FPR@95TPR = {:.5f}".format(auroc, fpr_at_95tpr))


def dbg_ood(epoch, in_loader, out_loader, model, criterion, args, log=None, name=None):
    model.eval()

    iters = {'in': iter(in_loader), 'out': iter(out_loader)}
    n = {'in': len(in_loader.dataset) / in_loader.batch_size,
         'out': len(out_loader.dataset) / out_loader.batch_size}

    ood_batch = torch.stack([out_loader.dataset[i][0][0] for i in range(-args.batch_size, 0)], dim=0).cuda()
    pos_batch = torch.stack([in_loader.dataset[i][0][0] for i in range(-args.batch_size, 0)], dim=0).cuda()
    bgs = {"in": pos_batch, "out": ood_batch}

    probs = dict()
    for k in iters.keys():
        probs[k] = dict()
        for bg_k in bgs.keys():
            probs[k][bg_k] = []

    for data_dist in iters.keys():
        with tqdm(total=n[data_dist]) as t:
            for idx, ((x1, _), _) in enumerate(iters[data_dist]):
                if idx >= 1000:
                    break
                # forward
                x1.contiguous()
                x1 = x1.cuda(non_blocking=True)

                for bg_k, bg_dist in bgs.items():
                    with torch.no_grad():
                        out = model(x1, bg_dist)

                    # loss = criterion(out)
                    prob = F.softmax(out, dim=1)[:, 0].flatten().cpu().numpy()
                    probs[data_dist][bg_k].append(prob)
                    # prob_meters["{}_{}".format(data_dist, bg_k)].update(prob.item())

                t.update()

    # [print("avg prob {}: {:.05f}".format(k, pm.avg)) for k, pm in prob_meters.items()]
    for k, p in probs.items():
        for k2, p2 in p.items():
            probs[k][k2] = np.concatenate(p2, axis=0)

    # for k, p in probs.items():
    #     for k2, p2 in p.items():
    #         print(k)
    #         print(k2)
    #         print(p2.shape)
    # probs = {k: np.concatenate(p, axis=0) for k, p in probs.items()}

    for sets in ["in_out-out_out", "in_in-out_in", "in_in-out_out"]:
        in_set, out_set = sets.split('-')
        in_probs = probs[in_set.split('_')[0]][in_set.split('_')[1]]
        out_probs = probs[out_set.split('_')[0]][out_set.split('_')[1]]
        labels = [1 for _ in in_probs] + [0 for _ in out_probs]
        tmp_probs = np.concatenate([in_probs, out_probs], axis=0)
        auroc = roc_auc_score(labels, tmp_probs)
        print("avg in_probs = {}".format(in_probs.mean()))
        print("avg out_probs = {}".format(out_probs.mean()))
        print("{}, AUROC = {}".format(sets, auroc))


def val_ood(epoch, in_loader, out_loader, model, criterion, args, log=None, name=None):
    model.eval()

    iters = {'in': iter(in_loader), 'out': iter(out_loader)}
    n = {'in': len(in_loader.dataset) / in_loader.batch_size,
         'out': len(out_loader.dataset) / out_loader.batch_size}
    prob_meters = {k: AverageMeter() for k in iters.keys()}
    probs = {k: [] for k in iters.keys()}

    # log.add_images('val_{}_nce_batch'.format(name), rescale_images(nce_batch), epoch)

    for data_dist in iters.keys():
        with tqdm(total=n[data_dist]) as t:
            for idx, ((x1, _), _) in enumerate(iters[data_dist]):
                if idx >= 1000:
                    break
                # forward
                x1.contiguous()
                x1 = x1.cuda(non_blocking=True)

                with torch.no_grad():
                    out = model(x1, x1)

                # loss = criterion(out)
                prob = F.softmax(out, dim=1)[0, 0].flatten().cpu().numpy()
                probs[data_dist].append(prob)
                prob_meters[data_dist].update(prob.mean())

                log.add_images('val_{}_{}_samples_x1'.format(data_dist, name), rescale_images(x1), epoch*n[data_dist]+idx)
                # log.add_images('val_{}_{}_samples_x2'.format(data_dist, name), rescale_images(x2), epoch*n[data_dist]+idx)
                t.update()
            print("avg prob {}: {:.05f}".format(data_dist, prob_meters[data_dist].avg))

    probs = {k: np.concatenate(p, axis=0) for k, p in probs.items()}
    labels = [1 for _ in probs['in']] + [0 for _ in probs['out']]
    probs = np.concatenate([probs['in'], probs['out']], axis=0)
    auroc = roc_auc_score(labels, probs)
    fpr_at_95tpr = fpr_at_tpr(probs, labels, tpr=0.95)

    log.add_scalar('AUROC_{}'.format(name), auroc, epoch)
    log.add_scalar('FPR@95TPR_{}'.format(name), fpr_at_95tpr, epoch)

    # log roc curve
    # fprs, tprs, ths = roc_curve(labels, probs)
    # for fpr, tpr in zip(fprs, tprs):
    #     log.add_scalar("val_{}_{}_ROC_curve".format(name, epoch), int(tpr*100), int(fpr*100))

    return prob_meters['in'].avg, prob_meters['out'].avg, auroc, fpr_at_95tpr


if __name__ == '__main__':
    opt = parse_option()
    pprint(vars(opt))
    main(opt)
