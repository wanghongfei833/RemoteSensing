# Remote Sensing Segment
##  数据集制作
    利用make_dataset.py进行数据集制作，其中有两个方式。
    分别为通过shp进行分割和按照特定尺寸进行裁剪
        -- 利用shp裁剪  crop_by_shp.py
            可以使用自己绘制的矩形shp对image和mask进行裁剪,或者利用数据集标签的shp进行裁剪
        -- 利用 block裁剪 crop_by_block.py
            将image和mask 按照 block的大小按着裁剪
## 模型训练
    """
    train.py
    对应修改数据
    直接pycharm 右键训练,模型训练后会在runs中得到对应的训练过程数据
    save_weights 是训练出保存的权重信息
    """
## 模型验证
    """
    predict.py
       
    
    """