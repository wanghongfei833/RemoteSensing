import os
import glob
import sys
import threading

import cv2
from math import ceil

from matplotlib import pyplot as plt
from tqdm import tqdm
from osgeo import gdal, ogr
import numpy as np
from models.creat_model import create_model as cm
import torch


def compute_metrics(confusion_matrix):
    # 计算真阳性、假阳性和假阴性
    true_positive = confusion_matrix.diag()
    false_positive = confusion_matrix.sum(dim=0) - true_positive
    false_negative = confusion_matrix.sum(dim=1) - true_positive
    # 计算精确度（Precision）
    precision = true_positive / (true_positive + false_positive)
    # 计算召回率（Recall）
    recall = true_positive / (true_positive + false_negative)
    # 计算F1分数（F1-score）
    f1_score = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1_score


def ini_dirs(model_name):
    for name in ["log", "save_weights", "sample"]:
        os.makedirs(os.path.join(name, model_name), exist_ok=True)


def get_path(resume, name="exp"):
    if not resume:
        os.makedirs("runs", exist_ok=True)

        dir_list = [int(i.replace(f"{name}", "")) for i in os.listdir("runs") if
                    len(name) == i or i[:len(name)] == name]
        if len(dir_list):
            root = f"runs/{name}{max(dir_list) + 1}"
        else:
            root = f"runs/{name}{1}"
        os.makedirs(root)
        return root
    else:
        dir_list = [int(i.replace(f"{name}", "")) for i in os.listdir("runs")]
        return f"runs/{name}{max(dir_list)}"


# 0.00026949458523585647

def ini_arr(arr):
    return arr


class MergeTIF(object):
    """
    主要是批量拼接tif,主函数有 run_dir 和 run_dirs两种函数
    """

    @staticmethod
    def GetExtent(infile):
        ds = gdal.Open(infile)
        geotrans = ds.GetGeoTransform()
        xsize = ds.RasterXSize
        ysize = ds.RasterYSize
        min_x, max_y = geotrans[0], geotrans[3]
        max_x, min_y = geotrans[0] + xsize * geotrans[1], geotrans[3] + ysize * geotrans[5]
        ds = None
        return min_x, max_y, max_x, min_y

    def boundary_acquisition(self, file_list):
        """
        获取边界信息
        Args:
            file_list:
        Returns:

        """
        min_x, max_y, max_x, min_y = self.GetExtent(file_list[0])
        for infile in file_list:
            minx, maxy, maxx, miny = self.GetExtent(infile)
            min_x, min_y = min(min_x, minx), min(min_y, miny)
            max_x, max_y = max(max_x, maxx), max(max_y, maxy)
        return min_x, max_y, max_x, min_y

    def RasterMosaic(self, file_list: list, out_path: str, name=""):
        Open = gdal.Open
        min_x, max_y, max_x, min_y = self.boundary_acquisition(file_list)
        in_ds = Open(file_list[0])
        bands = in_ds.RasterCount
        in_band = in_ds.GetRasterBand(bands)
        geotrans = list(in_ds.GetGeoTransform())
        width, height = geotrans[1], geotrans[5]
        columns = ceil((max_x - min_x) / width)  # 列数
        rows = ceil((max_y - min_y) / (-height))  # 行数
        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(out_path, columns, rows, bands, in_band.DataType,
                               options=["TILED=YES", "COMPRESS=LZW", "BIGTIFF=YES"])
        out_ds.SetProjection(in_ds.GetProjection())
        geotrans[0] = min_x  # 更正左上角坐标
        geotrans[3] = max_y
        out_ds.SetGeoTransform(geotrans)
        inv_geotrans = gdal.InvGeoTransform(geotrans)
        for index, in_fn in enumerate(file_list):
            in_ds = Open(in_fn)
            in_gt = in_ds.GetGeoTransform()
            offset = gdal.ApplyGeoTransform(inv_geotrans, in_gt[0], in_gt[3])
            x, y = map(int, offset)
            for i in tqdm(range(bands), desc=f'{name}:\t{index + 1}/{len(file_list)}'):  # 每个波段都要考虑
                data = in_ds.GetRasterBand(i + 1).ReadAsArray()
                out_ds.GetRasterBand(i + 1).WriteArray(data, x, y)  # x，y是开始写入时左上角像元行列号
        del in_ds, out_ds

    def run_dirs(self, root_dir: str, save_dir: str, remove=False):
        """
        对root中存在的多个文件夹进行拼接
        文件路径为
            -- root
                -- cat1
                    -- 1.tif , 2.tif , ....
                --cat2
                    -- 1.tif , 2.tif
        Args:
            root_dir: 根路径
            save_dir: 保存的路径
            remove: 是否删除1.tif 2.tif ....
        Returns:

        """
        assert len(os.listdir(root_dir)), print("错误...,{} 中没有文件夹".format(root_dir))
        for root in os.listdir(root_dir):
            tif_path = os.path.join(root_dir, root)  # 拿到 对应文件夹
            if not os.path.isdir(tif_path): continue
            tif_dirs = glob.glob(os.path.join(tif_path, "*.tif"))
            if len(tif_dirs) == 0: continue
            self.RasterMosaic(tif_dirs, os.path.join(save_dir, root + ".tif"), root)
            if remove:
                for i in tif_dirs:
                    os.remove(i)

    def run_dir(self, tif_dir: str, save_path: str, remove=False):
        """
        对root中存在的多个文件夹进行拼接
        文件路径为
            -- tif_dir
                    -- 1.tif , 2.tif , ....
        Args:
            tif_dir:  tif 的文件夹路径
            save_path: 保存路径
            remove: 是否删除1.tif 2.tif ....
        Returns:

        """
        tif_dirs = glob.glob(os.path.join(tif_dir, "*.tif"))
        self.RasterMosaic(tif_dirs, save_path, os.path.basename(tif_dir))
        if remove:
            for i in tif_dirs:
                os.remove(i)


def create_model(num_classes, in_channels, model_name="swin"):
    return cm(num_classes, in_channels, model_name)


def get_mean_std(loader):
    # Var[x] = E[X**2]-E[X]**2
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in loader:
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1
    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    print('mean:', mean, '\n', 'std:', std)
    sys.exit()


def data_load(dataloader):
    from my_dataset import mean_glob, std_glob
    mean_glob = mean_glob.view(1, len(mean_glob), 1, 1)
    std_glob = std_glob.view(1, len(std_glob), 1, 1)
    for image, label in dataloader:
        image = image * std_glob + mean_glob
        image = np.transpose(image[0].numpy(), (1, 2, 0))
        label = label[0].numpy()
        plt.subplot(131)
        plt.imshow(image)
        plt.subplot(132)
        plt.imshow(label)
        plt.subplot(133)
        plt.imshow(image)
        plt.imshow(label, alpha=0.7)
        plt.show()


class MakeDataSets(object):
    """
    项目应用于 存在遥感影像image.tif以及对应的标签文件 mask.shp
    1、首先利用 shp2mask 函数 将shp转换到和image.tif相同尺寸的 mask.tif
    2、利用 full_picture_clipping 函数 将image.tif/mask.tif 进行裁剪
          同时 跳过 image_arr 中非0 像素超过80%的 图像
    """

    def __init__(self):
        pass

    @staticmethod
    def shp2mask(image_path: str, mask_path: str, shp_path: str):
        """
            根据给定的形状文件创建与图像尺寸相同的遮罩。

            参数：
            image_path (str)：输入图像文件的路径。
            shapefile_path (str)：输入形状文件的路径。
            output_mask_path (str)：输出遮罩文件的路径。
            """

        # 打开输入图像文件
        image_ds = gdal.Open(image_path, gdal.GA_ReadOnly)
        image_width = image_ds.RasterXSize
        image_height = image_ds.RasterYSize

        # 创建输出遮罩文件
        driver = gdal.GetDriverByName('GTiff')
        mask_ds = driver.Create(mask_path, image_width, image_height, 1, gdal.GDT_Byte)
        mask_ds.SetProjection(image_ds.GetProjection())
        mask_ds.SetGeoTransform(image_ds.GetGeoTransform())

        # 将遮罩文件的像素值填充为0（默认值）
        mask_band = mask_ds.GetRasterBand(1)
        mask_band.Fill(0)

        # 打开输入形状文件
        shapefile_ds = ogr.Open(shp_path)
        layer = shapefile_ds.GetLayer()

        # 将形状文件中的多边形区域对应的像素值设置为1
        gdal.RasterizeLayer(mask_ds, [1], layer, burn_values=[1])
        # 关闭数据集
        mask_ds = None
        image_ds = None
        shapefile_ds = None

    def full_picture_clipping(self, image: str, mask: str, blocks: int, save_dir: str,
                              image_save: str = "images", label_save: str = "labels",
                              thread: bool = False):
        """
        将相同尺寸的 mask.tif和image.tif 按照block_size 进行裁剪
        :param image: image.tif的路径
        :param mask:  mask.tif的路径
        :param blocks: 裁剪的尺寸
        :param image_save: 图像文件名
        :param label_save: 标签文件名
        :param save_dir: 文件的保存的根路径
        :param thread: 是否采用多线程裁剪
        :return: None
        """
        assert os.path.exists(image), FileNotFoundError("{}文件未找到".format(image))
        assert os.path.exists(mask), FileNotFoundError("{}文件未找到".format(mask))
        os.makedirs(os.path.join(save_dir, image_save), exist_ok=True)
        os.makedirs(os.path.join(save_dir, label_save), exist_ok=True)

        image_dataset = gdal.Open(image)
        masks_dataset = gdal.Open(mask)
        width = image_dataset.RasterXSize
        height = image_dataset.RasterYSize
        for col in range(0, width, blocks):
            for row in range(0, height, blocks):
                x_off = col if col + blocks <= width else width - col
                y_off = row if row + blocks <= height else height - row
                image_arr = image_dataset.ReadAsArray(x_off, y_off, blocks, blocks)
                masks_arr = masks_dataset.ReadAsArray(x_off, y_off, blocks, blocks)
                masks_arr = masks_arr.astype(np.uint8)
                if thread:
                    threading.Thread(target=self.save_one_image, args=(image_arr, masks_arr, save_dir, image_save, label_save, row, col)).start()
                else:
                    self.save_one_image(image_arr, masks_arr, save_dir, image_save, label_save, row, col)

    @staticmethod
    def save_one_image(image_arr, masks_arr, save_dir, image_save, label_save, row, col):
        if (image_arr != 0) < image_arr.size * 0.8 and masks_arr.sum() == 0:  # 黑色图像 并且mask没有
            pass
        else:
            cv2.imwrite(os.path.join(save_dir, f"{image_save}/{row}_{col}.jpg"), np.transpose(image_arr[:3], (1, 2, 0)))
            cv2.imwrite(os.path.join(save_dir, f"{label_save}/{row}_{col}.jpg"), np.transpose(masks_arr * 255, (1, 2, 0)))


if __name__ == '__main__':
    p = r"D:\Projects\RemoteSensing\RS_Seg"
    print(os.path.basename(p))
