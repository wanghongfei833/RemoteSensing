import os.path
from ulits import ClipShpTif, MakeDataSets


def crop_by_shp(tif_list, mask_list, shp_list, out_dirs):
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


def crop_by_block(image, mask, block_size, save_dir, shp=None, **kwargs):
    """
    本函数会将shp绘制为和image一样大小的mask，然后进行block_size 裁剪
    Args:
        save_dir:   数据集的保存路径
        image:      遥感影像的tif路径
        mask:        掩膜的路径，第一次运行没有掩膜，则是会保存到对应路径
        block_size: 裁剪的大小
        shp:            利用arcgis或是其他方式得到的标签shp

    Returns:

    """
    make_data = MakeDataSets()
    if shp is not None:
        make_data.shp2mask(image, mask, shp)  # shp-->mask
    make_data.full_picture_clipping(image, mask, block_size, save_dir, **kwargs)


if __name__ == '__main__':
    tif_lists = ["1.tif"]  # tif的list
    mask_lists = ["1_mask.tif"]  # mask的list
    shp_lists = ["1.shp"]  # shp的list
    crop_by_shp(tif_lists, mask_lists, shp_lists, "image_label")
