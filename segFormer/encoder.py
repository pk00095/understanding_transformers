import torch
import torch.nn as nn

import einops
import warnings

warnings.filterwarnings("ignore")


class overlapPatchEmbedding(nn.Module):

    def __init__(self, patch_size, stride, num_channels, hidden_size):        
        super().__init__()

        self.proj = nn.Conv2d(
            in_channels=num_channels,
            out_channels=hidden_size,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size//2
        )

        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor):
        """create embedding on x by performing Convolution with overlaps,
          this leaks neighbour info into patch embedding

        Args:
            x (torch.tensor): tensor of shape (batch, height, width, channels)

        Returns:
            tuple(torch.Tensor, int, int): embeddings(batch, height x width, channel_out), height, width
        """

        # channel first for convolution
        x = einops.rearrange(x, "b h w c -> b c h w")
        x = self.proj(x)
        x = einops.rearrange(x, "b c h w -> b h w c")

        return self.norm(x)


class efficientMultiHeadAttention(nn.Module):

    def __init__(self, num_heads: int, d_model: int, sr_ratio: int) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model should be divisible for num_heads"
        self.d_model_per_head = d_model//num_heads
        self.num_heads = num_heads
        self.d_model = d_model

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        self.sr = nn.Conv2d(d_model, d_model, sr_ratio, sr_ratio)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):

        B, H, W, C = x.shape

        Q = self.query(x)
        Q = einops.rearrange(Q, "B H W (d n) -> B (H W) d n", d=self.d_model_per_head, n=self.num_heads)
        Q = einops.rearrange(Q, "B S d n -> B n S d")

        x = einops.rearrange(x, "B H W D -> B D H W")
        x_sr = self.sr(x)
        x_sr = einops.rearrange(x_sr, "B D H W -> B H W D")
        x_sr = einops.rearrange(x_sr, "B H W D -> B (H W) D")
        x_sr = self.layer_norm(x_sr)

        K = self.key(x_sr)
        K = einops.rearrange(K, "B S (n d) -> B S n d", d=self.d_model_per_head, n=self.num_heads)
        K_transpose = einops.rearrange(K, "B S n d -> B n d S")

        V = self.value(x_sr)
        V = einops.rearrange(V, "B S (n d) -> B S n d", d=self.d_model_per_head, n=self.num_heads)
        V = einops.rearrange(V, "B S n d -> B n S d")

        proj = torch.matmul(Q, K_transpose)/(self.d_model_per_head ** 0.5)
        attn = torch.matmul(nn.functional.softmax(proj, dim=-1), V)

        attn = einops.rearrange(attn, "B n S d -> B S n d")
        attn = einops.rearrange(attn, "B (H W) n d -> B H W (n d)", H=H, W=W)

        return attn
        

class MixFFN(nn.Module):
    """ 
    This layer adds spatial context. In Segformer no positional encoding is used.
    mix_feedforward = MLP(GELU(Conv(MLP(x))))
    """

    def __init__(self, intermediate_dim, d_model) -> None:
        super().__init__()

        self.mlp1 = nn.Linear(in_features=d_model, out_features=intermediate_dim)
        self.conv = nn.Conv2d(
            in_channels=intermediate_dim, 
            out_channels=intermediate_dim,
            kernel_size=3,
            stride=1,
            padding=1)
        self.act = nn.GELU()
        self.mlp2 = nn.Linear(in_features=intermediate_dim, out_features=d_model)

    def forward(self, x: torch.Tensor):
        """perform MLP(GELU(Conv(MLP(x))))

        Args:
            x (torch.tensor): (batch, h, w, d_model)

        Returns:
            torch.tensor: (batch, seq, d_model)
        """

        x = self.mlp1(x)

        # convert channel-last to channel-first for Convolution
        x = einops.rearrange(x, "b h w d -> b d h w")
        x = self.conv(x)
        x = einops.rearrange(x, "b d h w -> b h w d")

        x = self.act(x)
        x = self.mlp2(x)

        return x        


class efficientAttentionLayer(nn.Module):

    def __init__(self, num_heads, d_model, sr_ratio, intermediate_dim) -> None:
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.efficient_attn = efficientMultiHeadAttention(num_heads, d_model, sr_ratio)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.mlp = MixFFN(intermediate_dim, d_model)

    def forward(self, x):

        _x = self.layer_norm_1(x)
        _x = self.efficient_attn(_x)

        # First residual connection
        _x = _x + x

        # Second residual conn b/w MixFFN output and first residual output
        _x_norm = self.layer_norm_2(_x)
        out = _x + self.mlp(_x_norm)

        return out


class efficientAttentionBlock(nn.Module):

    def __init__(self, 
                 patch_merge_patch_size: int, 
                 patch_merge_stride: int, 
                 patch_merge_channels_in: int, 
                 patch_merge_channels_out: int, 
                 block_depth: int, 
                 num_heads: int, 
                 d_model: int, 
                 sr_ratio: int, 
                 intermediate_dim: int) -> None:
        """_summary_

        Args:
            patch_merge_patch_size (int): the kernel size for Conv of overlap patch merging layer
            patch_merge_stride (int): stride size for Conv of overlap patch merging layer
            patch_merge_channels_in (int): channel_in for Conv of overlap patch merging layer
            patch_merge_channels_out (int): channel_out for Conv of overlap patch merging layer
            block_depth (int): the number of efficient attention blocks in this layer
            num_heads (int): the number of heads in multihead attention
            d_model (int): the depth of multihead attention
            sr_ratio (int): the spatial reduction ratio for efficient self attention layer to reduce key and value
            intermediate_dim (int): the depth of intermediate linear layers in MixFFN
        """
        super().__init__()

        self.blocks = [efficientAttentionLayer(
                        num_heads, 
                        d_model, 
                        sr_ratio, 
                        intermediate_dim) for _ in range(block_depth)]
        
        self.overlap_patch_merging = overlapPatchEmbedding(
                                        patch_size=patch_merge_patch_size, 
                                        stride=patch_merge_stride, 
                                        num_channels=patch_merge_channels_in, 
                                        hidden_size=patch_merge_channels_out
                                    )
       
    def forward(self, x: torch.Tensor):
        """pass feature of dim (B, HxW, C_i) through N attention Blocks 
        then perform overlap patch merging to obtain features of 
        dim (B, HxW/R**2, C_i+1)

        Args:
            x (torch.Tensor): feature tensor of dim (B, H, W, C)

        Returns:
            torch.Tensor: _description_
        """

        x = self.overlap_patch_merging(x)

        for block in self.blocks:
            x = block(x)
        
        return x


class SegFormerEncoder(nn.Module):

    def __init__(self,
                 image_dims,
                 num_attn_blocks,
                 patch_merge_dims, 
                 patch_merge_patch_sizes,
                 patch_merge_strides,
                 attn_depths,
                 blocks_depths,
                 attn_num_heads,
                 efficient_attn_sr_ratios,
                 mixffn_depths
                 ) -> None:
        super().__init__()

        patch_merge_dims.insert(0, image_dims[-1])

        self.blocks = []
        self.layer_norms = []

        for idx in range(num_attn_blocks):
            self.blocks.append(
                efficientAttentionBlock(
                    patch_merge_patch_size=patch_merge_patch_sizes[idx], 
                    patch_merge_stride=patch_merge_strides[idx], 
                    patch_merge_channels_in=patch_merge_dims[idx], 
                    patch_merge_channels_out=patch_merge_dims[idx+1], 
                    block_depth=blocks_depths[idx], 
                    num_heads=attn_num_heads[idx], 
                    d_model=attn_depths[idx], 
                    sr_ratio=efficient_attn_sr_ratios[idx], 
                    intermediate_dim=mixffn_depths[idx]
                )
            )

            self.layer_norms.append(nn.LayerNorm(patch_merge_dims[idx+1]))

        self.blocks = nn.ModuleList(self.blocks)
        self.layer_norms = nn.ModuleList(self.layer_norms)

    def forward(self, x: torch.Tensor):
        """Pass input x through multiple stages to obtain multi scale features

        Args:
            x (torch.Tensor): Input Tensor of shape (BATCH, HEIGHT, WIDTH, IMAGE_CHANNELS)

        Returns:
            List[torch.Tensor]: MultiScale features from each stage
        """

        features = []

        for block, norm_layer in zip(self.blocks, self.layer_norms):
            feat = norm_layer(block(x))
            features.append(feat)
            x = feat

        return features