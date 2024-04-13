import torch

from encoder import SegFormerEncoder
from decoder import SegFormerDecoder


if __name__ == "__main__":

    height = 256
    width = 256
    channels = 3
    batch_size = 16
    d_model = 32
    num_heads = 1
    sequence_reduction_ratio = 2
    block_depth = 2

    image = torch.rand((batch_size, height, width, channels))
    print("input shape -> ", image.shape)

    encoder = SegFormerEncoder(
                image_dims=(height, width, channels),
                num_attn_blocks=4,
                patch_merge_dims=[32, 64, 160, 256], 
                patch_merge_patch_sizes=[7, 3, 3, 3],
                patch_merge_strides=[4, 2, 2, 2],
                attn_depths=[32, 64, 160, 256],
                blocks_depths=[2, 2, 2, 2],
                attn_num_heads=[1, 2, 5, 8],
                efficient_attn_sr_ratios=[8, 4, 2, 1],
                mixffn_depths=[32*4, 64*4, 160*4, 256*4]
    )

    (f1, f2, f3, f4) = encoder(image)
    print("Shape after Stage 1 :: ", f1.shape)
    print("Shape after Stage 2 :: ", f2.shape)
    print("Shape after Stage 3 :: ", f3.shape)
    print("Shape after Stage 4 :: ", f4.shape)

    decoder = SegFormerDecoder(
        out_dims=(height//4, width//4), 
        num_blocks=4, 
        channel_ins=[32, 64, 160, 256], 
        channel_out=256, 
        num_classes=2
    )

    out = decoder((f1, f2, f3, f4))

    print("Decoder out shape :: ", out.shape)
