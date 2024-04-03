import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from torch import nn
import train_utils.distributed_utils as utils
from torchvision.utils import save_image
from my_dataset import mean_glob, std_glob


def data_load(dataloader):
    mean = torch.tensor(mean_glob).resize(1, len(mean_glob), 1, 1)
    std = torch.tensor(std_glob).resize(1, len(std_glob), 1, 1)
    for index, (image, label) in enumerate(dataloader):
        image = image * std + mean
        image = image[0]
        rgb = np.transpose(image[:3].numpy(), (1, 2, 0))
        # plt.figure(dpi=100)
        plt.clf()
        plt.subplot(2, 2, 1)
        plt.imshow(rgb)
        plt.subplot(2, 2, 2)
        plt.title("ndwi")
        plt.imshow(image[-1].numpy())
        plt.subplot(2, 2, 3)
        plt.title('r')
        plt.imshow(image[0].numpy())
        plt.subplot(2, 2, 4)
        plt.imshow(label[0].numpy())
        # plt.savefig(f'data_show/{index}.png')
        # plt.close()
        plt.show()
        # plt.pause(0.1)
    sys.exit()


def get_mean_std(loader):
    # Var[x] = E[X**2]-E[X]**2
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in tqdm.tqdm(loader):
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1
    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    print(mean, std)
    sys.exit()


def criterion(inputs, target):
    return nn.functional.cross_entropy(inputs, target, ignore_index=255)


def bce(inputs, target):
    return torch.mean((inputs - target) ** 2)


def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler, print_freq=3, scaler=None, loss_fun="bce"):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'train : Epoch: [{}]'.format(epoch)
    for image, target in metric_logger.log_every(data_loader, max(len(data_loader) // print_freq, 1), header):
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            try:
                output = model(image)
            except Exception as e:
                output = model(image.repeat(2, 1, 1, 1))
                target = target.repeat(2, 1, 1)
            loss = criterion(output, target) if loss_fun == "criterion" else bce(output, target)
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        lr_scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss, lr=lr)
    return metric_logger.meters["loss"].global_avg, lr


def evaluate(model, data_loader, device, num_classes, epoch, print_freq=3):
    model.eval()
    os.makedirs('sample', exist_ok=True)
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    root = rf"sample/{model.model_name}"
    if epoch % 5 == 4:
        os.makedirs(fr"{root}/{epoch}", exist_ok=True)
    step = 0
    std = torch.tensor([0.0820, 0.1102], device=device).view(1, -1, 1, 1)
    mean = torch.tensor([0.6851, 0.5235], device=device).view(1, -1, 1, 1)
    with torch.no_grad():
        # step = 0
        for image, target in metric_logger.log_every(data_loader, max(1, len(data_loader) // print_freq), header):
            # [0.6851, 0.5235], [0.0820, 0.1102]
            step += 1
            image, target = image.to(device), target.to(device)
            output = model(image)
            confmat.update(target.flatten(), output.argmax(1).flatten())
            if epoch % 5 != 4:
                continue

            image = image * std + mean
            image = torch.cat([
                image,
                target.unsqueeze(1).repeat(1, image.size(1), 1, 1),
                output.argmax(1).unsqueeze(1).repeat(1, image.size(1), 1, 1)],
                dim=-1).to(torch.float32)
            for s in range(image.size(0)):
                save_image(image[s][1], fr'{root}/{epoch}/{epoch}_{step}_{s}.png')
        confmat.reduce_from_all_processes()

    return confmat


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=20,
                        warmup_factor=1e-5):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            ls = warmup_factor * (1 - alpha) + alpha
            return ls
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            x_w_n = x - warmup_epochs * num_step
            e_w = epochs - warmup_epochs
            l = (1 - (x_w_n) / ((e_w) * num_step))
            return l ** 0.5

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)

#
# def get_mean_std(loader):
#     # Var[x] = E[X**2]-E[X]**2
#     channels_sum, channels_squared_sum, num_batches = 0, 0, 0
#     for data, _ in loader:
#         channels_sum += torch.mean(data, dim=[0, 2, 3])
#         channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
#         num_batches += 1
#     mean = channels_sum / num_batches
#     std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
#     print('mean:', mean, '\n', 'std:', std)
#     sys.exit()
