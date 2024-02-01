import os
import glob
import sys

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
        os.makedirs(os.path.join(name, model_name),exist_ok=True)


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


class Shp2Mask(object):
    def __init__(self, shp_path, shp_list, result_path, pixel_size=0.00026949458523585647):
        """
        将 shp_list 转为 mask
        :param result_path: 数据的保存路径
        :param shp_list: 年份的 list 最后会在此文件夹下去进行检索
        """
        self.shp_path = shp_path
        self.shp_list = shp_list
        self.result_path = result_path
        self.pixel_size = pixel_size

    def shp_to_mask(self, shapefile_path, output):
        # 打开矢量形状文件
        driver = ogr.GetDriverByName("ESRI Shapefile")
        dataset = driver.Open(shapefile_path, 0)
        layer = dataset.GetLayer()

        # 获取图层的空间参考信息
        spatial_ref = layer.GetSpatialRef()

        # 定义栅格图像的范围
        x_min, x_max, y_min, y_max = layer.GetExtent()
        width = int((x_max - x_min) / self.pixel_size)
        height = int((y_max - y_min) / self.pixel_size)

        # 创建栅格图像
        target_ds = gdal.GetDriverByName('MEM').Create('', width, height, 1, gdal.GDT_Byte)
        target_ds.SetGeoTransform((x_min, self.pixel_size, 0, y_max, 0, -self.pixel_size))
        band = target_ds.GetRasterBand(1)
        band.SetNoDataValue(0)

        # 设置栅格图像的投影信息
        target_ds.SetProjection(spatial_ref.ExportToWkt())

        # 将矢量形状文件中的要素转换为栅格图像
        gdal.RasterizeLayer(target_ds, [1], layer, burn_values=[255])

        # 读取栅格图像数据并将其转换为掩码
        mask_data = band.ReadAsArray()
        mask = np.where(mask_data == 255, 1, 0)

        # 保存掩码为TIFF文件
        driver = gdal.GetDriverByName("GTiff")
        output_ds = driver.Create(output, width, height, 1, gdal.GDT_Int32)
        output_ds.SetGeoTransform((x_min, self.pixel_size, 0, y_max, 0, -self.pixel_size))
        output_ds.SetProjection(spatial_ref.ExportToWkt())
        output_band = output_ds.GetRasterBand(1)
        output_band.WriteArray(mask)

    def run(self):
        for shp in self.shp_list:
            shp_name = os.path.join(self.shp_path, shp)
            result_masks = os.path.join(self.result_path, shp.replace("shp", "tif"))
            self.shp_to_mask(shp_name, result_masks)


class ClipShpTif(object):
    """
    # 对tif和对应的shp裁剪成为数据集，需要在 MergeDataSets 运行以后执行
    """

    def __init__(self, pad_pix=0, min_length=200):
        self.pad_pix = pad_pix  # 在自己的 shp 周边各延展的像素值
        self.min_length = min_length  # 定义最小的 尺寸样本

    def save_tif(self, out_dir, years, index, pos_x, pox_y, bands, transfrom, pro, data):
        # # 创建新的保存影像的文件
        output_tiff_path = os.path.join(out_dir, f'{years}_{index}.tif')
        output_tiff = gdal.GetDriverByName('GTiff').Create(output_tiff_path, pos_x, pox_y, bands,
                                                           gdal.GDT_Float32)

        # 设置新文件的地理变换和投影信息
        output_tiff.SetGeoTransform(transfrom)
        output_tiff.SetProjection(pro)

        # 设置新文件的 NoData 值
        # 将影像数据写入新文件
        for band in range(bands):
            output_tiff.GetRasterBand(band + 1).SetNoDataValue(0)
            output_tiff.GetRasterBand(band + 1).WriteArray(data[band])

        # 关闭新文件
        output_tiff = None

    def get_pos(self, envelope, geo_transform, row, col):
        ulx = int((envelope[0] - geo_transform[0]) / geo_transform[1])
        uly = int((envelope[3] - geo_transform[3]) / geo_transform[5])
        lrx = int((envelope[1] - geo_transform[0]) / geo_transform[1])
        lry = int((envelope[2] - geo_transform[3]) / geo_transform[5])
        return max(ulx - self.pad_pix, 0), max(0, uly - self.pad_pix), min(col, lrx + self.pad_pix), min(row,
                                                                                                         lry + self.pad_pix)

    def get_row_col(self, dataset):
        cols = dataset.RasterXSize  # 图像长度
        rows = dataset.RasterYSize  # 图像宽度
        return rows, cols

    def clip_tif_mask(self, tif, mask, shp, out, name):
        """

        :param tif: tif的路径
        :param mask: mask 的路径
        :param shp: 裁剪的shp
        :param out: 输出路径
        :param name: 名称
        :return:
        """
        # 打开 Shapefile 文件
        shp_dataset = ogr.Open(shp)
        shp_layer = shp_dataset.GetLayer()
        # 打开 TIFF 文件
        tiff_dataset = gdal.Open(tif)
        mask_dataset = gdal.Open(mask)
        tiff_geo_transform = tiff_dataset.GetGeoTransform()
        mask_geo_transform = mask_dataset.GetGeoTransform()
        row_tif, col_tif = self.get_row_col(tiff_dataset)  # 图像长度
        row_mask, col_mask = self.get_row_col(mask_dataset)  # 图像宽度
        for index, feature in tqdm(enumerate(shp_layer, 1), total=len(shp_layer), desc=f"{name}"):
            geom = feature.GetGeometryRef()
            envelope = geom.GetEnvelope()
            # 计算要素在 TIFF 文件中的像素坐标
            ulx1, uly1, lrx1, lry1 = self.get_pos(envelope, tiff_geo_transform, row_tif, col_tif)
            ulx2, uly2, lrx2, lry2 = self.get_pos(envelope, mask_geo_transform, row_mask, col_mask)
            x_, y_ = lrx1 - ulx1, lry1 - uly1
            if x_ > self.min_length and y_ > self.min_length:
                tiff_data = tiff_dataset.ReadAsArray(ulx1, uly1, lrx1 - ulx1, lry1 - uly1)
                tiff_data = ini_arr(tiff_data)
                mask_data = mask_dataset.ReadAsArray(ulx2, uly2, lrx2 - ulx2, lry2 - uly2)
                if tiff_data is not None and mask_data is not None:
                    img_data = np.transpose(tiff_data, (1, 2, 0))
                    cv2.imwrite(os.path.join(out, f'image/{name}_{index}.png'), img_data)
                    cv2.imwrite(os.path.join(out, f'mask/{name}_{index}.png'), mask_data)


class MergeTIF(object):
    def __init__(self, root, save_root, remove=True):
        """

        :param root: tif 的根路径
        :param save_root: 保存的根路径
        :param remove: 拼接完成后是否删除
        """
        self.root = root
        self.save_root = save_root
        self.remove = remove

    def GetExtent(self, infile):
        ds = gdal.Open(infile)
        geotrans = ds.GetGeoTransform()
        xsize = ds.RasterXSize
        ysize = ds.RasterYSize
        min_x, max_y = geotrans[0], geotrans[3]
        max_x, min_y = geotrans[0] + xsize * geotrans[1], geotrans[3] + ysize * geotrans[5]
        ds = None
        return min_x, max_y, max_x, min_y

    def RasterMosaic(self, file_list, outpath, name):
        Open = gdal.Open
        min_x, max_y, max_x, min_y = self.GetExtent(file_list[0])
        for infile in file_list:
            minx, maxy, maxx, miny = self.GetExtent(infile)
            min_x, min_y = min(min_x, minx), min(min_y, miny)
            max_x, max_y = max(max_x, maxx), max(max_y, maxy)

        in_ds = Open(file_list[0])
        bands = in_ds.RasterCount
        in_band = in_ds.GetRasterBand(bands)
        # if in_band is None: in_band = in_ds.GetRasterBand(1)
        # bands = in_ds.RasterCount  # 波段数
        geotrans = list(in_ds.GetGeoTransform())
        width, height = geotrans[1], geotrans[5]
        columns = ceil((max_x - min_x) / width)  # 列数
        rows = ceil((max_y - min_y) / (-height))  # 行数

        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(outpath, columns, rows, bands, in_band.DataType,
                               options=["TILED=YES", "COMPRESS=LZW", "BIGTIFF=YES"])
        out_ds.SetProjection(in_ds.GetProjection())
        geotrans[0] = min_x  # 更正左上角坐标
        geotrans[3] = max_y
        out_ds.SetGeoTransform(geotrans)
        inv_geotrans = gdal.InvGeoTransform(geotrans)
        for index, in_fn in enumerate(file_list[3:]):
            in_ds = Open(in_fn)
            in_gt = in_ds.GetGeoTransform()
            offset = gdal.ApplyGeoTransform(inv_geotrans, in_gt[0], in_gt[3])
            x, y = map(int, offset)
            for i in tqdm(range(bands), desc=f'{name}:\t{index + 1}/{len(file_list)}'):  # 每个波段都要考虑
                data = in_ds.GetRasterBand(i + 1).ReadAsArray()
                # h,w = data.shape
                # ww = w+x
                # hh = h+y
                # x_size = out_ds.RasterXSize
                # y_size = out_ds.RasterYSize
                out_ds.GetRasterBand(i + 1).WriteArray(data, x, y)  # x，y是开始写入时左上角像元行列号
        del in_ds, out_ds

    def run(self):
        for root in os.listdir(self.root):
            tif_path = os.path.join(self.root, root)  # 拿到 对应文件夹
            if not os.path.isdir(tif_path):
                continue
            tif_dirs = glob.glob(os.path.join(tif_path, "*.tif"))
            if len(tif_dirs) == 0:
                continue
            self.RasterMosaic(tif_dirs, os.path.join(self.save_root, root + ".tif"), root)
            if self.remove:
                for i in tif_dirs:
                    os.remove(i)
                    # os.remove(tif_path)


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
