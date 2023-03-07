import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os
from tqdm import tqdm

from utils import save_images, load_model, save_model
from dataset import DiffusionDataset
from kexue_toy import ToyDiffusionModel
# from unet import NaiveUnet
from modules import UNet


max_epochs = 100000
T = 1000
img_path = "/root/autodl-nas/yuyan/data/celeba_hq/train"
img_size = 64
batch_size = 6  # 如果显存不够，可以降低为32、16，但不建议低于16
learning_rate = 3e-4
device = "cuda:0"


class DiffusionProgress:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, stride=1, eta=1, img_size=128, device="cpu") -> None:
        self.noise_steps = T
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.stride = stride
        self.eta = eta
        self.img_size = img_size
        self.device = device
        
        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
    
    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def get_noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        noise = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise, noise

    def sample(self, n=4, model=None):

        self.sample_beta = self.prepare_noise_schedule()[::self.stride].to(device)
        self.sample_alpha = 1. - self.sample_beta
        self.sample_alpha_hat = torch.cumprod(self.sample_alpha, dim=0)
        
        self.sample_beta_hat = torch.sqrt(1 - self.sample_alpha_hat)
        self.sample_alpha_hat_pre = torch.concat([torch.tensor([1]).to(device), self.sample_alpha_hat[:-1,]], dim=0)
        self.sample_beta_hat_pre = torch.sqrt(1 - self.sample_alpha_hat_pre)
        self.sample_alpha_ = self.sample_alpha_hat / self.sample_alpha_hat_pre
        self.sample_sigma_ = self.sample_beta_hat_pre / self.sample_beta_hat * torch.sqrt(1 - self.sample_alpha_**2) * self.eta
        self.sample_epsilon_ = self.sample_beta_hat - self.sample_alpha_ * torch.sqrt(self.sample_beta_hat_pre**2 - self.sample_sigma_**2)
        self.sample_noise_steps_ = len(self.sample_beta)
        
        model.eval()
        with torch.no_grad():
            x = torch.randn(n, 3, self.img_size, self.img_size).to(self.device)
            for i in tqdm(reversed(range(1, self.sample_noise_steps_)), position=0):
                
                t = (torch.ones(n) * i).long().to(self.device)
                pred_noise = model(x, t * self.stride)
                
                noise = torch.randn_like(x)
                
                _alpha = self.sample_alpha_[t][:, None, None, None]
                _epsilon = self.sample_epsilon_[t][:, None, None, None]
                _sigma = self.sample_sigma_[t][:, None, None, None]

                x = 1 / _alpha * (x - _epsilon * pred_noise) + _sigma * noise
                
        model.train()  
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x
    
def train():
    # torch.autograd.set_detect_anomaly(True)
    sample_dir = "samples"
    ckpt_dir = "/home/save_ckpt/ddim/"
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    ckpt_path = None
    
    dataset = DiffusionDataset(img_path, img_size=img_size)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=20, drop_last=True)
    progress = DiffusionProgress(img_size=img_size, device=device, eta=0, stride=100)
    # model = NaiveUnet(3, 3, 128)
    # model = UNet()
    # model = ToyDiffusionModel(device=device) 
    model = UNet(device=device) 
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    if ckpt_path:
        start_epoch, _, _, = load_model(ckpt_path, model, optimizer, map_location=device)
    else:
        start_epoch = 0
    
    for epoch in range(start_epoch, max_epochs):
        model.train()
        
        iter = 0
        
        pbar = tqdm(data_loader)
        for data in pbar:
            images = data.to(device)
            batch_steps = progress.sample_timesteps(images.shape[0]).to(device) 
            batch_noisy_imgs, batch_noise =  progress.get_noise_images(images, batch_steps)
            
            pred_noise = model(batch_noisy_imgs, batch_steps)
            
            loss = criterion(pred_noise, batch_noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pbar.set_description(f"loss: {loss.item():.4f}")
            iter += 1

        model.eval()
        with torch.no_grad():

            sampled_imgs = progress.sample(n=images.shape[0], model=model)
            save_images(sampled_imgs, f'{sample_dir}/%05d.png' % (epoch + 1))
            save_model(ckpt_dir, model, optimizer, epoch, iter, loss.item())

if __name__ == "__main__":
    train()