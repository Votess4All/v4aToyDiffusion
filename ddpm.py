import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import glob
import os
import cv2
import numpy as np
from tqdm import tqdm


max_epochs = 100000
img_path = "/root/autodl-nas/yuyan/data/celeba_hq/train"
img_size = 128
batch_size = 48  # 如果显存不够，可以降低为32、16，但不建议低于16
embedding_size = 128
learning_rate = 1e-3

T = 1000
alpha = np.sqrt(1 - 0.02 * np.arange(1, T+1) / T)
beta = np.sqrt(1 - alpha ** 2)
bar_alpha = np.cumprod(alpha)
bar_beta = np.sqrt(1 - bar_alpha ** 2) 
sigma = beta.copy()

class Swish(nn.Module):
    """
    https://github.com/seraphzl/swish-pytorch/blob/master/swish.py
    Args:
        nn (_type_): _description_
    """
    def __init__(self, num_features):
        """
            num_features: int, the number of input feature dimensions.
        """
        super(Swish, self).__init__()
        shape = (1, num_features) + (1, ) * 2
        self.beta = nn.Parameter(torch.Tensor(*shape))
        self.reset_parameters()

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

    def reset_parameters(self):
        nn.init.ones_(self.beta)
        

class DiffusionDataset(Dataset):
    def __init__(self, img_path, crop_size=None, img_size=128) -> None:
        super().__init__()
        self.crop_size = crop_size
        self.img_size = img_size
        
        self.images = glob.glob(os.path.join(img_path, "male", "*.jpg")) + \
            glob.glob(os.path.join(img_path, "female", "*.jpg"))
        
    
    def __getitem__(self, index):
        img = cv2.imread(self.images[index])
        h, w, _ = img.shape
        if self.crop_size is None:
            self.crop_size = min(h, w)
        else:
            self.crop_size = min(self.crop_size, h, w)
        
        h_start, w_start = (h - self.crop_size + 1) // 2, (w - self.crop_size + 1) // 2
        img = img[h_start: h_start+self.crop_size, w_start:w_start+self.crop_size, :]
        if img.shape[:2] != (self.img_size, self.img_size):
            img = cv2.resize(img, (self.img_size, self.img_size))
        
        img = img.astype('float32')
        img = img / 255 * 2 - 1  # 归一化到[-1,1]之间
        img = img.transpose(2, 0, 1)
        return img

    def __len__(self,):
        return len(self.images)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch=None, embedding_size=None) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch if out_ch else in_ch
        
        if self.in_ch == self.out_ch:
            self.skip_conv = nn.Identity()
        else:
            self.skip_conv = nn.Conv2d(in_ch, out_ch, 1)
        
        self.conv_t = nn.Conv2d(embedding_size, out_ch, 1)
        self.conv1 = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1), Swish(out_ch))
        self.conv2 = nn.Sequential(nn.Conv2d(out_ch, out_ch, 3, padding=1), Swish(out_ch))
        
        self.group_norm = nn.GroupNorm(32, out_ch)
    
    def forward(self, x, t):
        res = x
        x = self.conv1(x)
        x += self.conv_t(t)
        x = self.conv2(x)
        
        x = x + self.skip_conv(res)
        x = self.group_norm(x)
        
        return x


class ToyDiffusionModel(nn.Module):
    def __init__(self, T=1000, embedding_size=128) -> None:
        super().__init__()
        self.embedding_t = nn.Embedding(T, embedding_size)
        self.conv_in = nn.Conv2d(3, embedding_size, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(embedding_size, 3, kernel_size=3, padding=1)
         
        self.avg_pooling = nn.AvgPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2)
        self.group_norm = nn.GroupNorm(32, embedding_size)
        
        self.residual_block1 = ResidualBlock(embedding_size, embedding_size, embedding_size)
        self.residual_block2 = ResidualBlock(embedding_size, embedding_size, embedding_size)
        self.residual_block3 = ResidualBlock(embedding_size, embedding_size * 2, embedding_size)
        self.residual_block4 = ResidualBlock(embedding_size * 2, embedding_size * 2, embedding_size)
        self.residual_block5 = ResidualBlock(embedding_size * 2, embedding_size * 4, embedding_size)
        self.residual_block6 = ResidualBlock(embedding_size * 4, embedding_size * 4, embedding_size)
        
        self.residual_block_bottom = ResidualBlock(embedding_size * 4, embedding_size * 4, embedding_size) 

        self.residual_block1_de = ResidualBlock(embedding_size, embedding_size, embedding_size)
        self.residual_block2_de = ResidualBlock(embedding_size, embedding_size, embedding_size)
        self.residual_block3_de = ResidualBlock(embedding_size * 2, embedding_size, embedding_size)
        self.residual_block4_de = ResidualBlock(embedding_size * 2, embedding_size * 2, embedding_size)
        self.residual_block5_de = ResidualBlock(embedding_size * 4, embedding_size * 2, embedding_size)
        self.residual_block6_de = ResidualBlock(embedding_size * 4, embedding_size * 4, embedding_size)
    
    def forward(self, x, t):
        x = self.conv_in(x)                                     # [64, 128, 128, 128]
        embedding_t = self.embedding_t(t)[:, :, None, None]     # [64, 128]
        #TODO  这里为啥每一层都要加 embedding_t 
        x1 = self.residual_block1(x, embedding_t) 
        x1 = self.avg_pooling(x1)
        
        x2 = self.residual_block2(x1, embedding_t)
        x2 = self.avg_pooling(x2)
        
        x3 = self.residual_block3(x2, embedding_t)
        x3 = self.avg_pooling(x3)
        
        x4 = self.residual_block4(x3, embedding_t)
        x4 = self.avg_pooling(x4)
        
        x5 = self.residual_block5(x4, embedding_t)
        x5 = self.avg_pooling(x5)
        
        x6 = self.residual_block6(x5, embedding_t)
        x6 = self.avg_pooling(x6)
        
        x_bottom = self.residual_block_bottom(x6, embedding_t)
        
        x6_de = self.residual_block6_de(x_bottom+x6, embedding_t)
        x5_de = self.upsample(x6_de)

        x5_de = self.residual_block5_de(x5_de+x5, embedding_t)
        x4_de = self.upsample(x5_de)

        x4_de = self.residual_block4_de(x4_de+x4, embedding_t)
        x3_de = self.upsample(x4_de)

        x3_de = self.residual_block3_de(x3_de+x3, embedding_t)
        x2_de = self.upsample(x3_de)

        x2_de = self.residual_block2_de(x2_de+x2, embedding_t)
        x1_de = self.upsample(x2_de)

        x1_de = self.residual_block1_de(x1_de+x1, embedding_t)
        x1_de = self.upsample(x1_de)
        
        x_out = self.group_norm(x1_de)
        x_out = self.conv_out(x_out)
        
        return x_out

def l2_loss(y_true, y_pred):
    return torch.sum((y_true - y_pred) ** 2, dim=[1,2,3])


def imwrite(path, figure):
    """归一化到了[-1, 1]的图片矩阵保存为图片
    """
    figure = (figure + 1) / 2 * 255
    figure = np.round(figure, 0).astype('uint8')
    cv2.imwrite(path, figure)
    

def sample(path=None, n=4, z_samples=None, t0=0, model=None):
    """随机采样函数
    """
    if z_samples is None:
        z_samples = np.random.randn(n**2, 3, img_size, img_size)
    else:
        z_samples = z_samples.copy()
    for t in tqdm(range(t0, T), ncols=0):
        # 从[x_{T}, x_{T-1}, ..., x0]恢复
        t = T - t - 1
        # 每次推理 z_samples.shape[0] 个样本，这里就是16个
        bt = np.array([t] * z_samples.shape[0])
        z_samples_tensor = torch.from_numpy(z_samples).float().cuda()
        bt_tensor = torch.from_numpy(bt).cuda()
        
        out = model(z_samples_tensor, bt_tensor)
        out = out.detach().cpu().numpy()
        
        z_samples -= beta[t]**2 / bar_beta[t] * out
        z_samples /= alpha[t]
        z_samples += np.random.randn(*z_samples.shape) * sigma[t]
    x_samples = np.clip(z_samples, -1, 1)
    
    x_samples = x_samples.transpose(0, 2, 3, 1)
    if path is None:
        return x_samples
    figure = np.zeros((img_size * n, img_size * n, 3))
    for i in range(n):
        for j in range(n):
            digit = x_samples[i * n + j]
            figure[i * img_size:(i + 1) * img_size,
                   j * img_size:(j + 1) * img_size] = digit
    imwrite(path, figure)
    
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
    
def load_model(path, model, optimizer, epoch, step, loss, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    assert isinstance(ckpt, dict)
    
    model.load_state_dict(ckpt["model"])
    if optimizer and ckpt["optimizer"]:
        optimizer.load_state_dict(ckpt["optimizer"])
    epoch = ckpt["epoch"] if "epoch" in ckpt else 0
    step = ckpt["step"] if "step" in ckpt else 0
    loss = ckpt["loss"] if "loss" in ckpt else float("inf")
    return epoch, step, loss 

def train():
    # torch.autograd.set_detect_anomaly(True)
    os.makedirs("samples", exist_ok=True)
    os.makedirs("ckpt", exist_ok=True)
    
    dataset = DiffusionDataset(img_path, img_size=img_size)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=24, drop_last=True)
    model = ToyDiffusionModel(embedding_size=embedding_size)
    model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    for epoch in range(max_epochs):
        model.train()
        for iter, data in enumerate(data_loader):
            
            batch_imgs = data 
            batch_steps = np.random.choice(T, batch_size)
            batch_bar_alpha = bar_alpha[batch_steps][:, None, None, None]
            batch_bar_beta = bar_beta[batch_steps][:, None, None, None]
            batch_noise = np.random.randn(*batch_imgs.shape)
            batch_noisy_imgs = batch_imgs * batch_bar_alpha + batch_noise * batch_bar_beta 
            
            batch_noisy_imgs = batch_noisy_imgs.float().cuda()
            batch_noise = torch.from_numpy(batch_noise).float().cuda()
            batch_steps = torch.from_numpy(batch_steps).cuda()
            
            optimizer.zero_grad()
            
            pred_noise = model(batch_noisy_imgs, batch_steps)
            loss = torch.mean(l2_loss(pred_noise, batch_noise))
            loss.backward()
            optimizer.step()
            
            if iter % 20 == 0:
                print(f"#epoch: {epoch}, #iter: {iter}, loss: {loss}")

        model.eval()
        sample('samples/%05d.png' % (epoch + 1), model=model)
        save_model("ckpt", model, optimizer, epoch, iter, loss)
        model.train()

if __name__ == "__main__":
    train()