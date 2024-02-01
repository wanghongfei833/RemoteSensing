import numpy as np
import random

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as tfs


def pad_if_smaller(img, size, fill=0,seed=0):
    # 如果图像最小边长小于给定size，则用数值fill进行padding
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        random.seed(seed)
        starh = random.randint(0,padh)
        random.seed(seed)
        starw = random.randint(0,padw)
        img = F.pad(img, [starw, starh, padw-starw, padh-starh], fill=fill)
    return img

class Color(object):
    def __init__(self, p=0.5):
        self.p = p
        self.color = T.ColorJitter(brightness=random.randint(5, 20) * 0.05,
                                   contrast=random.randint(5, 15) * 0.05,
                                   saturation=random.randint(5,15)*0.05,
                                   # hue=random.randint(0, 10) * 0.005,
                                   )

    def __call__(self, image, target):
        if self.p < random.random():
            image = self.color(image)
        return image, target


class NDWI(object):
    def __call__(self, image, target):
        # tensor
        fenmu = image[1] + image[-1]
        ndwi = (image[1] - image[-1]) / (fenmu+1e-8)
        ndwi = (ndwi - ndwi.min()) / (ndwi.max() - ndwi.min())
        ndwi[ndwi == 0] = 1
        image[-1] = ndwi
        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomResize(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        # 这里size传入的是int类型，所以是将图像的最小边长缩放到size大小
        image = F.resize(image, [size,size])
        # 这里的interpolation注意下，在torchvision(0.9.0)以后才有InterpolationMode.NEAREST
        # 如果是之前的版本需要使用PIL.Image.NEAREST
        target = F.resize(target, [size,size], interpolation=T.InterpolationMode.NEAREST)
        return image, target


class Resizes(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = [self.min_size, self.max_size]
        # 这里size传入的是int类型，所以是将图像的最小边长缩放到size大小
        image = F.resize(image, size)
        # 这里的interpolation注意下，在torchvision(0.9.0)以后才有InterpolationMode.NEAREST
        # 如果是之前的版本需要使用PIL.Image.NEAREST
        target = F.resize(target, size, interpolation=T.InterpolationMode.NEAREST)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target

class RandomVlFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.vflip(image)
            target = F.vflip(target)
        return image, target
class RandomRotate(object):
    def __init__(self, rotate_prob=0.5, rotate=90):
        self.rotate_prob = rotate_prob
        self.rotate = rotate

    def __call__(self, image, target):
        rand = random.random()
        if rand < self.rotate_prob:
            rotate = random.randint(-self.rotate, self.rotate)
            image = F.rotate(image, rotate)
            target = F.rotate(target, rotate)
        return image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        # b, w, h = image.size()
        # if w < self.size or h < self.size:
        #     image = torch.nn.functional.interpolate(image, (300, 300),mode='bicubic')
        #     target = torch.nn.functional.interpolate(target, (300, 300),mode='nearest')
        seed = random.randint(-10000,10000)
        image = pad_if_smaller(image, self.size, fill=0,seed=seed)
        target = pad_if_smaller(target, self.size, fill=0,seed=seed)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target
