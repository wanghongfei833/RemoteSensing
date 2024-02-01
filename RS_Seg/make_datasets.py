import os.path
from ulits import ClipShpTif


def main(tif_list, mask_list, shp_list, out_dirs):
    """

    :param tif_list: 图像路径集合
    :param mask_list: mask 路径集合
    :param shp_list:  shp 路径集合
    :param out_dirs: 输出的根目录
    :return: None
    """
    print('开始裁剪数据...')
    clip = ClipShpTif()
    os.makedirs(os.path.join(out_dirs, "image"), exist_ok=True)
    os.makedirs(os.path.join(out_dirs, "mask"), exist_ok=True)
    for tif, mask, shp in zip(tif_list, mask_list, shp_list):
        assert os.path.exists(tif), FileNotFoundError("文件{}找不到".format(tif))
        assert os.path.exists(mask), FileNotFoundError("文件{}找不到".format(mask))
        assert os.path.exists(shp), FileNotFoundError("文件{}找不到".format(shp))
        base_name = os.path.basename(tif.replace(".tif", ""))
        clip.clip_tif_mask(tif, mask, shp, out=out_dirs, name=base_name)


if __name__ == '__main__':
    tif_list = ["1.tif"]  # tif的list
    mask_list = ["1_mask.tif"]  # mask的list
    shp_list = ["1.shp"]  # shp的list
    main(tif_list, mask_list, shp_list, "image_label")
