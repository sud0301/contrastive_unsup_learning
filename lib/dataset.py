from __future__ import print_function

import os
import json
import re
import torch
import torchvision.datasets as datasets

from .util import AverageMeter

data_root = "/misc/lmbraid19/galessos/datasets/"


class ImageFolderInstance(datasets.ImageFolder):
    """Folder datasets which returns the index of the image as well
    """

    def __init__(self, root, transform=None, target_transform=None, two_crop=False):
        super(ImageFolderInstance, self).__init__(root, transform, target_transform)
        self.two_crop = two_crop

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, index) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        image = self.loader(path)
        if self.transform is not None:
            img = self.transform(image)
        else:
            img = image
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.two_crop:
            img2 = self.transform(image)
            img = torch.cat([img, img2], dim=0)

        return img, target


def get_cifar_ids(dataset, classes, name):
    idsfile = "{}.csv".format(name)
    if not os.path.isfile(idsfile):
        print("Finding indices for "+name+"...")
        ids = [i for i, (_, l) in enumerate(dataset) if l in classes]
        # write ids to json for future use and return
        with open(idsfile, "w") as jsf:
            json.dump(ids, jsf)
        return ids
    # load json
    print("Found indices for " + name + ", loading...")
    with open(idsfile) as jsf:
        ids = json.load(jsf)
    return ids


# def get_ood_loaders(args):
#     # train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
#     #                                       transforms.RandomHorizontalFlip(),
#     #                                       transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8),
#     #                                       transforms.RandomGrayscale(p=0.25),
#     #                                       transforms.Resize(64),
#     #                                       transforms.ToTensor(),
#     #                                       transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
#     #                                                            std=[0.2471, 0.2435, 0.2616])])
#     # val_transform = transforms.Compose([transforms.Resize(64),
#     #                                     transforms.ToTensor(),
#     #                                     transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
#     #                                                          std=[0.2471, 0.2435, 0.2616])])
#
#     # train_transform = TransformTwice(train_transform, train_transform)
#     # val_transform = TransformTwice(val_transform, val_transform)
#
#     if args.dataset == 'svhn':
#         train_in_data = CIFAR10('./tmp_datasets', train=True, transform=train_transform, download=True)
#         val_in_data = CIFAR10('./tmp_datasets', train=False, transform=val_transform, download=True)
#         val_out_data = SVHN('./tmp_datasets', split='test', transform=val_transform, download=True)
#
#     elif 'cifar10_' in args.dataset:
#         train_dataset = CIFAR10('./tmp_datasets', train=True, transform=train_transform, download=True)
#         val_dataset = CIFAR10('./tmp_datasets', train=False, transform=val_transform, download=True)
#         if args.dataset == "cifar10_animals":
#             positive_classes = [2, 3, 4, 5, 6, 7]
#         else:
#             positive_classes = [int(d) for d in args.dataset.replace('cifar10_', '')]
#         negative_classes = [c for c in list(range(10)) if c not in positive_classes]
#
#         train_ids = get_cifar_ids(train_dataset, positive_classes, args.dataset+"_train_in")
#         val_in_ids = get_cifar_ids(val_dataset, positive_classes, args.dataset+"_val_in")
#         val_out_ids = get_cifar_ids(val_dataset, negative_classes, args.dataset+"_val_out")
#
#         train_in_data = torch.utils.data.Subset(train_dataset, train_ids)
#         val_in_data = torch.utils.data.Subset(val_dataset, val_in_ids)
#         val_out_data = torch.utils.data.Subset(val_dataset, val_out_ids)
#
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


