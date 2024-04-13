import torch
import torch.nn as nn
from torch.nn import functional as F

import einops
import math
import warnings

warnings.filterwarnings("ignore")


class PositionalEncoding(nn.Module):
    # https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        # self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = einops.rearrange(x, "b s d -> s b d")
        x = x + self.pe[:x.size(0)]
        x = einops.rearrange(x, "s b d -> b s d")
        return x


class PatchLayer(nn.Module):

    def __init__(self, in_channels, out_channels, patch_size, add_token=True) -> None:
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=patch_size,
            stride=patch_size)
        
        self.add_token = add_token

        if self.add_token:
            self.token = nn.Parameter(torch.rand(size=(1, 1, out_channels)))
        
    def forward(self, x):
        """creates patches of size patches_size and adds tokens if required

        Args:
            x (torch.tensor): The raw image tensor of dims (Batch_size, in_channels, height, width)

        Returns:
            torch.tensor: The image embeddings flattened out with shape (Batch_size, seq(+1), out_channels)
        """
        B, _, _, _ = x.shape
        # x = einops.rearrange(x, "B H W C -> B C H W")
        x = self.conv(x)
        x = einops.rearrange(x, "B C H W -> B (H W) C")

        if self.add_token:
            token = self.token.expand(B, -1, -1)
            x = torch.concat((token, x), dim=1)

        return x


class MLP(nn.Module):

    def __init__(self, d_model, d_intermediate):

        super().__init__()

        self.mlp1 = nn.Linear(d_model, d_intermediate)
        self.act = nn.GELU()
        self.mlp2 = nn.Linear(d_intermediate, d_model)

    def forward(self, x):

        x = self.mlp1(x)
        x = self.act(x)
        x = self.mlp2(x)

        return x


class Attention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model_per_head = d_model//num_heads
        self.d_model = d_model
        self.num_heads = num_heads

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

    def forward(self, x):

        Q = self.query(x)
        Q = einops.rearrange(Q, "B S (d n) -> B S d n", n=self.num_heads, d=self.d_model_per_head)
        Q = einops.rearrange(Q, "B S d n -> B n S d")

        K = self.value(x)
        K = einops.rearrange(K, "B S (d n) -> B S d n", n=self.num_heads, d=self.d_model_per_head)
        K_transpose = einops.rearrange(K, "B S d n -> B n d S")

        V = self.value(x)
        V = einops.rearrange(V, "B S (d n) -> B S d n", n=self.num_heads, d=self.d_model_per_head)
        V = einops.rearrange(V, "B S d n -> B n S d")

        proj = torch.matmul(Q, K_transpose)/(self.d_model_per_head**0.5)
        attn = torch.matmul(F.softmax(proj, dim=-1), V)

        attn = einops.rearrange(attn, "B n S d -> B S n d")
        attn = einops.rearrange(attn, "B S n d -> B S (n d)")

        return attn


class AttentionBlock(nn.Module):

    def __init__(self, d_model, num_heads, d_intermediate):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = Attention(d_model=d_model, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model=d_model, d_intermediate=d_intermediate)

    def forward(self, x):

        x_ = self.norm1(x)
        x_ = self.attention(x_)
        # Residual conn between input to norm 1 and attn
        x_ = x + x_        
        # Residual conn between input to norm 2 and output of mlp
        x_ = self.mlp(self.norm2(x_)) + x_

        return x_


class VisionTransformer(nn.Module):

    def __init__(self, image_dims, n_layers, patch_size, d_model, d_intermediate, num_heads, num_classes) -> None:
        super().__init__()

        height, width, n_channels = image_dims
        sequence_length = height*width//patch_size**2

        self.patcher = PatchLayer(
            in_channels=n_channels,
            out_channels=d_model,
            patch_size=patch_size,
            add_token=True
        )

        self.pos_emb = PositionalEncoding(
            d_model=d_model,
            max_len=sequence_length+1
        )

        blocks = []

        for _ in range(n_layers):
            blocks.append(
                AttentionBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_intermediate=d_intermediate
                )
            )

        self.blocks = nn.ModuleList(blocks)

        self.global_avg_pooling = nn.Conv1d(
            in_channels=sequence_length+1,
            out_channels=1,
            kernel_size=1,
            stride=1
        )

        self.activation = nn.Tanh()
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):

        x = self.patcher(x)
        x = self.pos_emb(x)

        for block in self.blocks:
            x = block(x)

        x = self.global_avg_pooling(x)
        x = x.squeeze_(1)

        x = self.activation(x)

        x = self.classifier(x)

        return x




