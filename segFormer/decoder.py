import torch
import torch.nn as nn
import torch.nn.functional as F

import einops
import warnings

warnings.filterwarnings("ignore")


class ResizeLayer(nn.Module):

    def __init__(self, size) -> None:
        super().__init__()
        self.size = size

    def forward(self, x):

        x = einops.rearrange(x, "b h w c -> b c h w")
        x = F.interpolate(
            x, 
            size=self.size, 
            scale_factor=None, 
            mode="bilinear", 
            align_corners=False)
        x = einops.rearrange(x, "b c h w -> b h w c")
        return x
    

class SegFormerDecoder(nn.Module):
    def __init__(self, out_dims, num_blocks, channel_ins, channel_out, num_classes) -> None:
        super().__init__()

        self.blocks = []
        for idx in range(num_blocks):
            self.blocks.append(nn.Linear(channel_ins[idx], channel_out))

        self.resizer = ResizeLayer(out_dims)
        self.mlp = nn.Linear(num_blocks*channel_out, num_classes)

    def forward(self, features):
        _features = []

        for feature, block in zip(features, self.blocks):
            _features.append(self.resizer(block(feature)))
            
        _features = torch.cat(_features, dim=-1)
        _features = self.mlp(_features)

        return _features