import os
import time
import datetime
import torch
from train_utils import train_one_epoch, evaluate, create_lr_scheduler
from models.creat_model import create_model
from my_dataset import create_datasets
from ulits import get_path, get_mean_std, ini_dirs
import warnings

warnings.filterwarnings("ignore")
from torch.utils.tensorboard import SummaryWriter


def main(args):
    # 创建学习率更新策略，这里是每个step更新一次(不是每个epoch)
    train_loader, val_loader = create_datasets(args)
    # -------------------- 查看数据集--------------
    # data_load(train_loader)
    # ------------     计算 均值方差 ----------------
    # get_mean_std(train_loader)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # segmentation nun_classes + background
    num_classes = args.num_classes
    in_channels = args.in_channels
    model_name = args.model_name
    # 用来保存训练以及验证过程中信息
    weight_root = "save_weights/{}".format(model_name)
    results_file = "log/{}_results{}.txt".format(model_name, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    model = create_model(num_classes, in_channels, model_name)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    params_to_optimize = [
        {"params": params},
        # {"params": [p for p in model.classifier.parameters() if p.requires_grad]}
    ]
    optimizer = torch.optim.Adam(
        params_to_optimize,
        lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs,
                                       warmup=True, warmup_epochs=5, warmup_factor=1e-7)

    # import matplotlib.pyplot as plt
    # lr_list = []
    # for _ in range(args.epochs):
    #     for _ in range(len(train_loader)):
    #         lr_scheduler.step()
    #         lr = optimizer.param_groups[0]["lr"]
    #         lr_list.append(lr)
    # plt.plot(range(len(lr_list)), lr_list)
    # plt.show()

    if args.resume:
        weight_path = os.path.join(weight_root, f'model_{args.resume}.pth')
        checkpoint = torch.load(weight_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])
    start_time = time.time()
    write = SummaryWriter(get_path(args.resume, name=model.model_name))

    for epoch in range(args.start_epoch, args.epochs):
        mean_loss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch,
                                        lr_scheduler, args.print_freq, scaler)
        confmat = evaluate(model, val_loader, device, num_classes, epoch, args.print_freq)
        result = confmat.result()
        val_info = ""
        for index, (classes, source) in enumerate(result.items()):
            source_mean = source.mean().item()
            for d in range(len(source)):
                write.add_scalar(f"{classes}/class_{d}", source[d], epoch)
            write.add_scalar(f"{classes}/{classes}_mean", source_mean, epoch)
            list_str = ' '.join('{:.1f}'.format(i * 100) for i in source.tolist())
            val_info += "[ {}: '{:.1f}'\t{} ]\t".format(classes, source_mean * 100, list_str)
        # write into txt
        with open(results_file, "a") as f:
            # 记录每个epoch对应的train_loss、lr以及验证集各指标
            train_info = f"[epoch: {epoch}]\t" \
                         f"train_loss: {mean_loss:.4f}\t" \
                         f"lr: {lr:.6f}\t"
            f.write(train_info + val_info + "\n")
        print(train_info + val_info + "\n")
        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args
                     }
        if args.amp:
            save_file["scaler"] = scaler.state_dict()
        if epoch % args.frequency == 0:
            torch.save(save_file, os.path.join(weight_root, f"model_{epoch}.pth"))
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch unet training")
    parser.add_argument("--data-path", default=r"image_label\result", help="data root")
    parser.add_argument("--train_name", default=r"train", help="训练集的文件名")
    parser.add_argument("--test_name", default=r"test", help="测试集的文件名")
    parser.add_argument("--image_name", default=r"image", help="图像的名称")
    parser.add_argument("--mask_name", default=r"mask", help="掩膜的名称")
    parser.add_argument("--number_works", default=4, help="线程数量")
    parser.add_argument("--image_size", default=224, help="训练验证的图像尺寸")

    # Model
    parser.add_argument("--in_channels", default=3, help="模型的输入通道数量")
    parser.add_argument("--nums_class", default=2, help="模型的分类数量")
    parser.add_argument("--model_name", default="swin",
                        choices=["swin", "unet", "deep"], help="模型名称")

    # H P
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=32, type=int)
    parser.add_argument("--epochs", default=200, type=int, metavar="N", help="number of total epochs to train")
    parser.add_argument('--print-freq', default=3, type=int, help='需要打印的轮次数目')
    # Optime
    parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--wd', '--weight-decay', default=0, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--resume', default='', help='继续从第 i 个epoch进行训练')
    parser.add_argument('--frequency', default=1, help='模型的保存频率')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='start epoch')
    # Mixed precision training parameters frequency
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    ini_dirs(args.model_name)
    print(args)
    main(args)
