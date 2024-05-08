# Diffusion: Use UNet for extracting noise from Latent Space

import torch
from torch import nn
from torch.nn import functional as F


from attention import SelfAttention, CrossAttention


class TimeEmbedding(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.linear_1 = nn.Linear(n_embd, 4*n_embd)
        self.linear_2 = nn.Linear(4*n_embd, 4*n_embd)
        
    def forward(self, x):
        # x: (1, 320)
        x = self.linear_1(x)
        x = F.silu(x)
        x = self.linear_2(x)
        # (1, 1280)
        return x
    
    
    
class UpSample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        # (batch, features, height, width) -> (batch, features, height*2, width*2)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)
        
        
        
class SwitchSequential(nn.Sequential):
    def forward(self, x, context, time):
        for layer in self:
            if isinstance(layer, UNet_AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNet_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x
    

class UNet_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_time=1280):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channels)
        
        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        
        
    def forward(self, feature, time):
        # feature: (batch, in_channels, height, width)
        # time: (1, 1280)
        
        residue = feature
        feature = self.groupnorm_feature(feature)
        feature = F.silu(feature)
        feature = self.conv_feature(feature)
        
        time = F.silu(time)
        time = self.linear_time(time)
        merged = feature + time.unsqueeze(-1).unsqueeze(-1)
        
        merged = self.groupnorm_merged(merged)
        merged = F.silu(merged)
        merged = self.conv_merged(merged)
        
        return merged + self.residual_layer(residue)
    
    
    

class UNet_AttentionBlock(nn.Module):
    def __init__(self, n_head, n_embd, d_context=768):
        super().__init__()
        channels = n_head * n_embd
        
        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
        
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False)
        
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels, 4*channels*2)
        self.linear_geglu_2 = nn.Linear(4*channels, channels)
        
        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        
        
    def forward(self, x, context):
        # x: (batch, features, height, width)
        # context: (batch, seq_len, dim)
        residue_long = x
        
        x = self.groupnorm(x)
        x = self.conv_input(x)
        n, c, h, w = x.shape
        
        # (batch, features, height, width) -> (batch, height*width, features)
        x = x.view((n,c,h*w))
        x = x.transpose(-1, -2)
        
        # normalization + self_attention with skip connection
        residue_short = x
        x = self.layernorm_1(x)
        x = self.attention_1(x)
        x += residue_short
        residue_short = x
        
        # normalization + cross_attention with skip connection
        x = self.layernorm_2(x)
        x = self.attention_2(x, context) # cross attention
        x += residue_short
        residue_short = x
        
        # normalization + feedforward with GELU and skip connection
        x = self.layernorm_3(x)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)
        x = self.linear_geglu_2(x)
        x += residue_short
        
        # (batch, height*width, features) -> (batch, features, height, width) 
        x = x.transpose(-1, -2)
        x = x.view((n, c, h, w))
        
        return self.conv_output(x) + residue_long


class UNet_OutputLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        # (batch, 320, height//8, width//8)
        x = self.groupnorm(x)
        x = F.silu(x)
        x = self.conv(x)
        # (batch, 4, height//8, width//8)
        return x
    
    
    
    
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # UNet Encoder
        self.encoders = nn.ModuleList([
            # (batch, 4, height//8, width//8)
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
            
            SwitchSequential(UNet_ResidualBlock(320, 320), UNet_AttentionBlock(8, 40)),
            
            SwitchSequential(UNet_ResidualBlock(320, 320), UNet_AttentionBlock(8, 40)),
            
            # (batch, 320, height//8, width//8) -> (batch, 320, height//16, width//16)
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),
            
            SwitchSequential(UNet_ResidualBlock(320, 640), UNet_AttentionBlock(8, 80)),
            
            SwitchSequential(UNet_ResidualBlock(640, 640), UNet_AttentionBlock(8, 80)),
            
            # (batch, 640, height//16, width//16) -> (batch, 640, height//32, width//32)
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),
            
            SwitchSequential(UNet_ResidualBlock(640, 1280), UNet_AttentionBlock(8, 160)),
            
            SwitchSequential(UNet_ResidualBlock(1280, 1280), UNet_AttentionBlock(8, 160)),
            
            # (batch, 1280, height//32, width//32) -> (batch, 1280, height//64, width//64)
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),
            
            SwitchSequential(UNet_ResidualBlock(1280, 1280)),
            
            # (batch, 1280, height//64, width//64) -> (batch, 1280, height//64, width//64)
            SwitchSequential(UNet_ResidualBlock(1280, 1280)),
            
        ])
        
        # bottleneck
        self.bottleneck = SwitchSequential(
            UNet_ResidualBlock(1280, 1280),
            UNet_AttentionBlock(8, 160),
            UNet_ResidualBlock(1280, 1280),
        )
        
        
        # UNet Decoder
        self.decoders = nn.ModuleList([
            # (batch, 2560, height//64, width//64) -> (batch, 1280, height//64, width//64)
            SwitchSequential(UNet_ResidualBlock(2560, 1280)),
            SwitchSequential(UNet_ResidualBlock(2560, 1280)),
            SwitchSequential(UNet_ResidualBlock(2560, 1280), UpSample(1280)),
            
            SwitchSequential(UNet_ResidualBlock(2560, 1280,), UNet_AttentionBlock(8, 160)),
            SwitchSequential(UNet_ResidualBlock(2560, 1280,), UNet_AttentionBlock(8, 160)),
            SwitchSequential(UNet_ResidualBlock(1920, 1280,), UNet_AttentionBlock(8, 160), UpSample(1280)),
            
            SwitchSequential(UNet_ResidualBlock(1920, 640), UNet_AttentionBlock(8, 80)),
            
            SwitchSequential(UNet_ResidualBlock(1280, 640), UNet_AttentionBlock(8, 80)),
            
            SwitchSequential(UNet_ResidualBlock(960, 640,), UNet_AttentionBlock(8, 80), UpSample(640)),
            
            SwitchSequential(UNet_ResidualBlock(960, 320,), UNet_AttentionBlock(8, 40)),
            
            SwitchSequential(UNet_ResidualBlock(640, 320,), UNet_AttentionBlock(8, 40)),
            
            SwitchSequential(UNet_ResidualBlock(640, 320,), UNet_AttentionBlock(8, 40)),
        ])
        
        
    def forward(self, x, context, time):
        # x: (batch, 4, height//8, width//8)
        # context: (batch, seq_len, dim)
        # time: (1, 1280)
        
        skip_connections = []
        for layers in self.encoders:
            x = layers(x, context, time)
            skip_connections.append(x)
            
        x = self.bottleneck(x, context, time)
        
        for layers in self.decoders:
            x = torch.cat((x, skip_connections.pop()), dim=1)
            x = layers(x, context, time)
        return x
        
        

class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNet()
        self.final = UNet_OutputLayer(320, 4)
        
    def forward(self, latent, context, time):
        # latent: (batch, 4, height//8, width//8)
        # context: (batch, seq_len, seq_len, dim)
        # time: (1, 320)
        
        # (1, 320) -> (1, 1280)
        time = self.time_embedding(time)
        
        # (batch, 4, height//8, width//8) -> (batch, 320, height//8, width//8)
        output = self.unet(latent, context, time)
        
        #  (batch, 320, height//8, width//8) -> (batch, 4, height//8, width//8)
        output = self.final(output)
        
        return output