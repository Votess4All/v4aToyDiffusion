import torch.nn as nn
import torch


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
        

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch=None, embedding_size=None) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch if out_ch else in_ch
        self.embedding_size = embedding_size
        
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
    def __init__(self, T=1000, embedding_size=256, device="cpu") -> None:
        super().__init__()
        
        self.device = device
        
        self.embedding_size = embedding_size
        # self.embedding_t = nn.Embedding(T+1, embedding_size)
        # self.embedding_t = TimeSiren(embedding_size)
        self.conv_in = nn.Conv2d(3, embedding_size, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(embedding_size, 3, kernel_size=3, padding=1)
         
        self.avg_pooling = nn.AvgPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2)
        self.group_norm = nn.GroupNorm(32, embedding_size)
        
        self.residual_block1 = ResidualBlock(embedding_size, embedding_size, embedding_size)
        self.residual_block2 = ResidualBlock(embedding_size, embedding_size, embedding_size)
        self.residual_block3 = ResidualBlock(embedding_size, embedding_size * 2, embedding_size)
        # self.residual_block4 = ResidualBlock(embedding_size * 2, embedding_size * 2, embedding_size)
        # self.residual_block5 = ResidualBlock(embedding_size * 2, embedding_size * 4, embedding_size)
        # self.residual_block6 = ResidualBlock(embedding_size * 4, embedding_size * 4, embedding_size)
        
        # self.residual_block_bottom = ResidualBlock(embedding_size * 4, embedding_size * 4, embedding_size) 
        self.residual_block_bottom = ResidualBlock(embedding_size * 2, embedding_size * 2, embedding_size) 

        self.residual_block1_de = ResidualBlock(embedding_size, embedding_size, embedding_size)
        self.residual_block2_de = ResidualBlock(embedding_size, embedding_size, embedding_size)
        self.residual_block3_de = ResidualBlock(embedding_size * 2, embedding_size, embedding_size)
        # self.residual_block4_de = ResidualBlock(embedding_size * 2, embedding_size * 2, embedding_size)
        # self.residual_block5_de = ResidualBlock(embedding_size * 4, embedding_size * 2, embedding_size)
        # self.residual_block6_de = ResidualBlock(embedding_size * 4, embedding_size * 4, embedding_size)
    
    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc
    
    def forward(self, x, t):
        x = self.conv_in(x)                                     # [64, 128, 128, 128]
        
        t = t.unsqueeze(-1).type(torch.float)
        embedding_t = self.pos_encoding(t, self.embedding_size)[:, :, None, None] 
        # embedding_t = self.embedding_t(t)[:, :, None, None]     # [64, 128]
        #TODO  这里为啥每一层都要加 embedding_t 
        x1 = self.residual_block1(x, embedding_t) 
        x1 = self.avg_pooling(x1)
        
        x2 = self.residual_block2(x1, embedding_t)
        x2 = self.avg_pooling(x2)
        
        x3 = self.residual_block3(x2, embedding_t)
        x3 = self.avg_pooling(x3)
        
        # x4 = self.residual_block4(x3, embedding_t)
        # x4 = self.avg_pooling(x4)
        
        # x5 = self.residual_block5(x4, embedding_t)
        # x5 = self.avg_pooling(x5)
        
        # x6 = self.residual_block6(x5, embedding_t)
        # x6 = self.avg_pooling(x6)
        
        # x_bottom = self.residual_block_bottom(x6, embedding_t)
        x_bottom = self.residual_block_bottom(x3, embedding_t)
        
        # x6_de = self.residual_block6_de(x_bottom+x6, embedding_t)
        # x5_de = self.upsample(x6_de)

        # x5_de = self.residual_block5_de(x5_de+x5, embedding_t)
        # x4_de = self.upsample(x5_de)

        # x4_de = self.residual_block4_de(x4_de+x4, embedding_t)
        # x3_de = self.upsample(x4_de)

        x3_de = x_bottom
        x3_de = self.residual_block3_de(x3_de+x3, embedding_t)
        x2_de = self.upsample(x3_de)

        x2_de = self.residual_block2_de(x2_de+x2, embedding_t)
        x1_de = self.upsample(x2_de)

        x1_de = self.residual_block1_de(x1_de+x1, embedding_t)
        x1_de = self.upsample(x1_de)
        
        x_out = self.group_norm(x1_de)
        x_out = self.conv_out(x_out)
        
        return x_out