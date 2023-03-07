import glob
import os
import cv2

from torch.utils.data import Dataset

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