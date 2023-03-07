import cv2
import numpy as np
import os

import torch
import torchvision
from PIL import Image 


def imwrite(path, figure):
    """归一化到了[-1, 1]的图片矩阵保存为图片
    """
    figure = (figure + 1) / 2 * 255
    figure = np.round(figure, 0).astype('uint8')
    cv2.imwrite(path, figure)

def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def save_model(dirname, model, optimizer, epoch, step, loss):
    
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "step": step,
        "loss": loss,
    }
    
    path = os.path.join(dirname, f"ddpm_epoch{epoch}_step{step}_loss{loss}.pth")
    torch.save(ckpt, path)
    
def load_model(path, model, optimizer, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    assert isinstance(ckpt, dict)
    
    model.load_state_dict(ckpt["model"])
    if optimizer and ckpt["optimizer"]:
        optimizer.load_state_dict(ckpt["optimizer"])
    epoch = ckpt["epoch"] if "epoch" in ckpt else 0
    step = ckpt["step"] if "step" in ckpt else 0
    loss = ckpt["loss"] if "loss" in ckpt else float("inf")
    return epoch, step, loss 
