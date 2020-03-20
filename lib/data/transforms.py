import random
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
from PIL import Image


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


class Unnormalize(object):
    def __init__(self, mean, std):
        self.mean = torch.as_tensor(mean).cuda()
        self.std = torch.as_tensor(std).cuda()

    def __call__(self, tensor):
        return tensor * self.std[:, None, None] + self.mean[:, None, None]

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


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


train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8),
                                      transforms.RandomGrayscale(p=0.25),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                           std=[0.2471, 0.2435, 0.2616])])
val_transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                         std=[0.2471, 0.2435, 0.2616])])

train_transform = TransformTwice(train_transform, train_transform)
val_transform = TransformTwice(val_transform, val_transform)

rescale_images = Unnormalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2471, 0.2435, 0.2616])

