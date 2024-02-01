import glob
import os
import sys
import time
from datetime import timedelta, datetime
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from PIL import Image
from osgeo import gdal, osr, ogr
from my_dataset import SegmentationPresetEval
import warnings
from ulits import ini_arr, create_model

warnings.filterwarnings("ignore")


def write_tif(new_path, im_data, im_Geotrans, im_proj, datatype=gdal.GDT_Byte):
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape
    diver = gdal.GetDriverByName('GTiff')
    new_dataset = diver.Create(new_path, im_width, im_height, im_bands, datatype)
    new_dataset.SetGeoTransform(im_Geotrans)
    new_dataset.SetProjection(im_proj)

    if im_bands == 1:
        new_dataset.GetRasterBand(1).WriteArray(im_data)
        new_dataset.FlushCache()

    else:
        for i in range(im_bands):
            datas = im_data[i]
            new_dataset.GetRasterBand(i + 1).WriteArray(datas)
            new_dataset.FlushCache()
    del new_dataset, im_data


def format_time_delta(end_time, start_time):
    delta = timedelta(seconds=end_time - start_time)
    hours, remainder = divmod(delta.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    result = "{:^4}h{:^2}m{:^2}s".format(int(hours), int(minutes), int(seconds))
    return result


def pad_to_multiple(arr, length):
    c, h, w = arr.shape
    if h == length and w == length: return arr
    pad_h = (length - h % length) % length
    pad_w = (length - w % length) % length
    padded_arr = np.pad(arr, ((0, 0), (0, pad_h), (0, pad_w)), mode='constant')
    return padded_arr


def raster2vector(raster_path, vecter_path, field_name="class", ignore_values=None):
    print('开始转为SHP...')
    # 读取路径中的栅格数据
    raster = gdal.Open(raster_path)
    # in_band 为想要转为矢量的波段,一般需要进行转矢量的栅格都是单波段分类结果
    # 若栅格为多波段,需要提前转换为单波段
    band = raster.GetRasterBand(1)
    # 读取栅格的投影信息,为后面生成的矢量赋予相同的投影信息
    prj = osr.SpatialReference()
    prj.ImportFromWkt(raster.GetProjection())

    drv = ogr.GetDriverByName("ESRI Shapefile")
    # 若文件已经存在,删除
    if os.path.exists(vecter_path):
        drv.DeleteDataSource(vecter_path)

    # 创建目标文件
    polygon = drv.CreateDataSource(vecter_path)
    # 创建面图层
    poly_layer = polygon.CreateLayer(vecter_path[:-4], srs=prj, geom_type=ogr.wkbMultiPolygon)
    # 添加浮点型字段,用来存储栅格的像素值
    field = ogr.FieldDefn(field_name, ogr.OFTReal)
    poly_layer.CreateField(field)

    # FPolygonize将每个像元转成一个矩形，然后将相似的像元进行合并
    # 设置矢量图层中保存像元值的字段序号为0
    print('Shp 开始转换')
    s = time.time()
    gdal.FPolygonize(band, None, poly_layer, 0)
    print('转换完成,耗时', format_time_delta(time.time(), s))
    polygon.SyncToDisk()
    polygon = None
    raster = None
    # os.remove(raster_path)
    print(f'\nShp 写入完成 耗时:{format_time_delta(time.time(), s)}')


class PredictTif():
    def __init__(self,
                 weights_path: str,
                 in_channels: int = 3,
                 nums_class: int = 2,
                 model_name: str = "swin",
                 upper_predict: bool = False,
                 base_size: int = 224,
                 save_dir: str = "result"):
        """

        :param weights_path: 对应的模型权重
        :param in_channels: 模型的输入通道
        :param nums_class: 模型的输出通道
        :param model_name: 模型名称
        :param upper_predict: 模型是否进行上采样验证 如 112 -->224 --> predict --> 112
        :param base_size: 上述的112 即模型的基础大小
        :param save_dir:  保存路径
        """
        print("the save root is : {}".format(save_dir))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = create_model(nums_class, in_channels, model_name)
        weights_dict = torch.load(weights_path, map_location='cpu')['model']
        self.model.load_state_dict(weights_dict)
        self.model.to(self.device)
        self.model.eval()
        self.upper_predict = upper_predict
        self.base_size = base_size
        self.save_dir = save_dir
        self.txt_name = save_dir + "/user_time_log_{}.txt".format(datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.txt = open(self.txt_name, "w")
        self.txt.close()

    def main(self, file_name, **kw):
        """
        对影像进行单batch预测
        :param file_name:
        :return:
        """
        self.txt = open(self.txt_name, "a")
        dataset = gdal.Open(file_name)
        transform = SegmentationPresetEval(self.base_size, predict=True)
        base_name = os.path.basename(file_name)
        cols = dataset.RasterXSize  # 图像宽度
        rows = dataset.RasterYSize  # 图像高度
        im_proj = dataset.GetProjection()  # 读取投影
        im_Geotrans = dataset.GetGeoTransform()  # 读取仿射变换
        h, w = rows, cols
        result = np.zeros((h, w), dtype=np.uint8)
        n_h = h // self.base_size + 1 if h % self.base_size else h // self.base_size
        n_w = w // self.base_size + 1 if w % self.base_size else w // self.base_size
        pbars = tqdm.tqdm(range(n_h), file=sys.stdout)
        one = time.time()
        stars = kw["stars"]
        mask = Image.fromarray(np.zeros((self.base_size, self.base_size)))
        total = 0
        s = time.time()
        for top in pbars:
            for left in range(n_w):
                total += 1
                l = left * self.base_size
                t = top * self.base_size
                r = l + self.base_size if l + self.base_size < cols else cols
                b = t + self.base_size if t + self.base_size < rows else rows
                x_, y_ = r - l, b - t
                datas = dataset.ReadAsArray(l, t, x_, y_).astype(np.uint8)
                datas = ini_arr(datas)[::-1]
                if (datas == 0).sum() < self.base_size * self.base_size // 2:
                    _, h, w = datas.shape
                    if w != self.base_size or h != self.base_size:
                        datas = pad_to_multiple(datas, self.base_size)
                    # 单影像输入
                    datas = Image.fromarray(np.transpose(datas, (1, 2, 0)))
                    if self.upper_predict:
                        datas = datas.resize((2 * self.base_size, 2 * self.base_size))
                    datas, _ = transform(datas, mask)
                    datas = datas.unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        predicts_data = self.model(datas)
                        if self.upper_predict:
                            predicts_data = F.interpolate(predicts_data,
                                                          size=(self.base_size,
                                                                self.base_size),
                                                          mode="bicubic")
                        predicts_data = torch.argmax(predicts_data, 1)
                        predicts_data = predicts_data[0].cpu().numpy()
                        result[t:b, l:r] = predicts_data[:h, :w]
                pbars.set_description_str(
                    f'{base_name} | all_time:{format_time_delta(time.time(), stars)} |'
                    f'one_time:{format_time_delta(time.time(), one)} | speed:{(time.time() - s) / total:.3f}S')
        dataset = None
        self.txt.write(f"{os.path.basename(file_name)}:\tpredict:\t" + format_time_delta(time.time(), s))
        s = time.time()
        write_tif(new_path=os.path.join(self.save_dir, base_name),
                  im_data=result,
                  im_proj=im_proj,
                  im_Geotrans=im_Geotrans)
        self.txt.write(f"\twrite_tif:\t" + format_time_delta(time.time(), s) + "\n")


if __name__ == '__main__':
    predicts = PredictTif(
        save_dir="result",
        in_channels=3,
        nums_class=2,
        model_name="swin",
        base_size=224,
        upper_predict=False,
        weights_path="save_weights/model_199.pth",
    )
    # tiff_list = glob.glob("image_tif/*.tif")
    tiff_list = [".tif",".tif"]
    stars = time.time()
    for tif_name in tiff_list:
        predicts.main(tif_name, stars=stars)
