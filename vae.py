# Variational Auto Encoder
# Encoder: compress the image into a latent space (smaller than the original)
# Decoder: extract the latent image and convert it back to original

import torch
from torch import nn
from torch.nn import functional as F

from attention import SelfAttention


class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, x):
        # x: (batch, in_channels, height, width)
        residue = x
        
        x = self.groupnorm_1(x)
        x = F.silu(x)
        
        # (batch, in_channels, height, width) -> (batch, out_channels, height, width)
        x = self.conv_1(x)
        
        x = self.groupnorm_2(x)
        x = F.silu(x)
        
        # (batch, out_channels, height, width) -> (batch, out_channels, height, width)
        x = self.conv_2(x)
        return x + self.residual_layer(residue)
    
    
    
    
class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)
    
    def forward(self, x):
        n,c,h,w = x.shape
        
        # x: (batch, channels, height, width)
        residue = x
        
        x = self.groupnorm(x)
        # (batch, channels, height, width) -> (batch, channels, height*width)
        x = x.view((n, c, h*w))
        # (batch, channels, height*width) -> (batch, height*width, channels)
        x = x.transpose(-1, -2)
        # (batch, height*width, channels) -> (batch, height*width, channels)
        x = self.attention(x)
        # (batch, height*width, channels) -> (batch, channels, height*width)
        x = x.transpose(-1, -2)
        # (batch, channels, height*width) -> (batch, channels, height, width)
        x = x.view((n, c, h, w))
        
        return x + residue


class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            # (batch, channel, height, width) -> (batch, channel, height, width)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            # (batch, 128, height, width) -> (batch, 128, height, width)
            VAE_ResidualBlock(128, 128),
            # (batch, 128, height, width) -> (batch, 128, height, width)
            VAE_ResidualBlock(128, 128),
            # (batch, 128, height, width) -> (batch, 128, height//2, width//2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            # (batch, 128, height//2, width//2) -> (batch, 256, height//2, width//2)
            VAE_ResidualBlock(128, 256),
            # (batch, 256, height//2, width//2) -> (batch, 256, height//2, width//2)
            VAE_ResidualBlock(256, 256),
            # (batch, 256, height//2, width//2) -> (batch, 256, height//4, width//4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            # (batch, 256, height//4, width//4) -> (batch, 512, height//4, width//4)
            VAE_ResidualBlock(256, 512),
            # (batch, 512, height//4, width//4) -> (batch, 512, height//4, width//4)
            VAE_ResidualBlock(512, 512),
            # (batch, 512, height//4, width//4) -> (batch, 512, height//8, width//8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0), 
            # (batch, 512, height//8, width//8) -> (batch, 512, height//8, width//8)
            VAE_ResidualBlock(512, 512),
            # (batch, 512, height//8, width//8) -> (batch, 512, height//8, width//8)
            VAE_ResidualBlock(512, 512),
            # (batch, 512, height//8, width//8) -> (batch, 512, height//8, width//8)
            VAE_ResidualBlock(512, 512),
            # (batch, 512, height//8, width//8) -> (batch, 512, height//8, width//8)
            VAE_AttentionBlock(512),
            # (batch, 512, height//8, width//8) -> (batch, 512, height//8, width//8)
            VAE_ResidualBlock(512, 512),
            nn.GroupNorm(32, 512),
            nn.SiLU(),
            
            # bottle neck
            # (batch, 512, height//8, width//8) -> (batch, 8, height//8, width//8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            # (batch, 8, height//8, width//8) -> (batch, 8, height//8, width//8)
            nn.Conv2d(8, 8, kernel_size=1, padding=0),
        )
    
    
    def forward(self, x, noise):
        # x: (batch, channel, height, width)
        # noise: (batch, out_channel, height//8, width//8)
        
        for module in self:
            if getattr(module, 'stride', None) == (2,2):
                x = F.pad(x, (0, 1, 0, 1)) # left, right, top, bottom
            x = module(x)
        
        # (batch, 8, height/8, height/8) -> 2 * (batch, 4, height/8, height/8)
        mean, log_var = torch.chunk(x, 2, dim=1)
        # clamp the the log_var into defined range
        log_var = torch.clamp(log_var, -30, 20)
        # calculate the standard distribution
        var = log_var.exp()
        std = var.sqrt()
        # sample from distribution
        x = mean + std * noise
        
        # scale the output by a constant
        x *= 0.18215
        return x
        


class VAE_Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            # (batch, 4, height//8, width//8) -> (batch, 4, height//8, width//8)
            nn.Conv2d(4, 4, kernel_size=1, padding=0), 
            
            nn.Conv2d(4, 512, kernel_size=3, padding=1), 
            
            VAE_ResidualBlock(512, 512),
            
            VAE_AttentionBlock(512),
            # (batch, 512, height//8, width//8) -> (batch, 512, height//8, width//8)
            VAE_ResidualBlock(512, 512),
            # (batch, 512, height//8, width//8) -> (batch, 512, height//8, width//8)
            VAE_ResidualBlock(512, 512),
            # (batch, 512, height//8, width//8) -> (batch, 512, height//8, width//8)
            VAE_ResidualBlock(512, 512),
            # (batch, 512, height//8, width//8) -> (batch, 512, height//8, width//8)
            VAE_ResidualBlock(512, 512),
            # (batch, 512, height//8, width//8) -> (batch, 512, height//4, width//4)
            nn.Upsample(scale_factor=2), 
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            
            # (batch, 512, height//4, width//4) -> (batch, 512, height//2, width//2)
            nn.Upsample(scale_factor=2), 
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            
            VAE_ResidualBlock(512, 256),
            VAE_ResidualBlock(256, 256),
            VAE_ResidualBlock(256, 256),
            
            # (batch, 256, height//2, width//2) -> (batch, 256, height, width)
            nn.Upsample(scale_factor=2),
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1), 
            
            
            VAE_ResidualBlock(256, 128),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),
            
            nn.GroupNorm(32, 128),
            
            nn.SiLU(),
            # (batch, 128, height, width) -> (batch, 3, height, width)
            nn.Conv2d(128, 3, kernel_size=3, padding=1)
            
        )
        
        
    def forward(self, x):
        # x: (batch, 4, height//8, width//8)
        
        # reverse scaling
        x /= 0.18215
        
        for module in self:
            x = module(x)
        
        # batch, 3, height, width
        return x