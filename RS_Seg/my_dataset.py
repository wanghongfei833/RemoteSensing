import os
import random

import numpy as np
import torch
import torch.utils.data as data
import tqdm
from PIL import Image, ImageOps
from osgeo import gdal
import transforms as T
from ulits import ini_arr
from shutil import copy

mean_glob = torch.tensor([0.1791, 0.1374, 0.1243])
std_glob = torch.tensor([0.1264, 0.1130, 0.1003])


class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, mean=None,
                 std=None):
        if std is None:
            std = std_glob
        if mean is None:
            mean = mean_glob
        min_size = int(0.5 * base_size)
        max_size = int(2.0 * base_size)

        trans = []
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
            trans.append(T.RandomVlFlip(hflip_prob))
        trans.extend([
            T.RandomResize(min_size, max_size),
            T.RandomRotate(),
            T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self, base_size, mean=None, std=None, predict=False):
        if std is None:
            std = std_glob
        if mean is None:
            mean = mean_glob
        trains = []
        if not predict:
            trains.append(T.RandomCrop(base_size))
        trains.extend([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trains)

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(train, base_size=320, crop_size=256):
    return SegmentationPresetTrain(base_size, crop_size) if train else SegmentationPresetEval(crop_size)


class DatasetSegmentation(data.Dataset):
    def __init__(self,
                 data_root,
                 transforms=None,
                 image_name='image',
                 mask_name='mask',
                 ):
        super(DatasetSegmentation, self).__init__()
        self.images = os.path.join(data_root, image_name)
        self.masks = os.path.join(data_root, mask_name)
        self.transforms = transforms

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        # img = self.readTif(self.images[index])
        # target = self.readTif(self.masks[index])
        img = self.read_img(self.images[index])
        target = self.read_img(self.masks[index])
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        target[target != 0] = 1
        return img, target

    def __len__(self):
        return len(self.images)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=0)
        return batched_imgs, batched_targets

    def read_img(self, name):
        img = Image.open(name)
        return img

    def readTif(self, fileName):
        dataset = gdal.Open(fileName)
        im_width = dataset.RasterXSize  # 栅格矩阵的列数
        im_height = dataset.RasterYSize  # 栅格矩阵的行数
        im_bands = dataset.RasterCount  # 波段数
        im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  # 获取数据
        im_data = im_data.astype(np.uint8)
        if im_bands == 1:
            return Image.fromarray(im_data)
        im_data = ini_arr(im_data)
        im_data = np.transpose(im_data, (1, 2, 0))
        return Image.fromarray(im_data[:, :, 0])


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


def split_datas(img_file_lst, mks_file_lst, sdir, split=None, title=None, seed=10):
    """
    此函数将单文件夹数据集分割为train/test/val
    :param seed: 随机种子
    :param img_file_lst: 图像的路径列表
    :param mks_file_lst: 掩膜的路径列表
    :param sdir: 保存路径
    :param split: 分割点
    :param title: 分割的title
    :return:
    """
    if title is None:
        title = ["train", "test", "val"]
    if split is None:
        split = [0.7, 0.2, 0.1]
    assert len(title) == len(split), ValueError(
        "split和title的长度必须对应，但是为title:{}和split{}".format(len(title), len(split)))
    assert len(img_file_lst) != 0, ValueError("img_file_lst is None")
    random.seed(seed)
    random.shuffle(img_file_lst)
    random.seed(seed)
    random.shuffle(mks_file_lst)
    length = len(img_file_lst)
    # split.insert(0, 0)
    # split.insert(-1, 1)
    split = [int(length * i) for i in split]  # 换算为cat
    for i in range(len(title)):
        if i == 0:
            cat_img = img_file_lst[0, split[0]]
            cat_msk = mks_file_lst[0, split[0]]
        elif i == len(title) - 1:
            cat_img = img_file_lst[split[i]:]
            cat_msk = mks_file_lst[split[i]:]

        else:
            cat_img = img_file_lst[split[i]:split[i + 1]]
            cat_msk = mks_file_lst[split[i], split[i + 1]]
        tit = title[i]
        img_path = os.path.join(sdir, f"{tit}/image")
        msk_path = os.path.join(sdir, f"{tit}/mask")
        os.makedirs(img_path, exist_ok=True)
        os.makedirs(msk_path, exist_ok=True)

        for i, m in tqdm.tqdm(zip(cat_img, cat_msk), desc=tit):
            copy(i, img_path)
            copy(m, msk_path)


def create_datasets(args):
    batch_size = args.batch_size
    train_root = os.path.join(args.data_path, args.train_name)
    test_root = os.path.join(args.data_path, args.test_name)

    train_dataset = DatasetSegmentation(train_root,
                                        transforms=get_transform(train=True,
                                                                 base_size=args.image_size,
                                                                 crop_size=args.image_size),
                                        image_name=args.image_name,
                                        mask_name=args.mask_name,
                                        )

    val_dataset = DatasetSegmentation(test_root,
                                      transforms=get_transform(train=False,
                                                               base_size=args.image_size,
                                                               crop_size=args.image_size),
                                      image_name=args.image_name,
                                      mask_name=args.mask_name,
                                      )

    num_workers = args.number_works if args.number_works is not None else min(
        [os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               pin_memory=True,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             shuffle=True,
                                             collate_fn=val_dataset.collate_fn)
    return train_loader, val_loader
