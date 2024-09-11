import torch
from torch import nn, Tensor
from .layers import DownSampleBlock, MidBlock, UpSampleBlock, TimeEmbedding
from typing import Literal

class UNet(nn.Module):

    def __init__(self,
        in_channels : int,
        out_channels : int,
        down_channels : list[int],
        mid_channels : list[int],
        num_layers : int,
        norm_channels : int,
        num_heads : int,
        t_emb_dim : int,
        output_activation : Literal['sigmoid','tanh','linear'] = 'linear',        
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.down_channels = down_channels
        self.mid_channels = mid_channels
        self.num_layers = num_layers
        self.norm_channels = norm_channels
        self.num_heads = num_heads
        self.output_activation = output_activation
        self.t_emb_dim = t_emb_dim

        self.activations = nn.ModuleDict({
            'sigmoid' : nn.Sigmoid(),
            'tanh' : nn.Tanh(),
            'linear' : nn.Identity(),
        })

        self.time_embedding = TimeEmbedding(t_emb_dim)

        self.t_proj = nn.Sequential(
            nn.Linear(t_emb_dim,t_emb_dim),
            nn.SiLU(),
            nn.Linear(t_emb_dim,t_emb_dim),
        )

        self.conv_in = nn.Conv2d(in_channels=in_channels,out_channels=down_channels[0],kernel_size=3,padding=1)

        self.down_blocks = nn.ModuleList([
            DownSampleBlock(
                in_channels = down_channels[i],
                out_channels = down_channels[i+1],
                downsample = True,
                num_layers = num_layers,
                norm_channels = norm_channels,
                num_heads = num_heads,
                t_emb_dim = t_emb_dim,
            )
            for i in range(len(down_channels)-1)
        ])

        self.mid_blocks = nn.ModuleList([
            MidBlock(
                in_channels = mid_channels[i],
                out_channels = mid_channels[i+1],
                num_layers = num_layers,
                norm_channels = norm_channels,
                num_heads = num_heads,
                t_emb_dim = t_emb_dim,
            )
            for i in range(len(mid_channels)-1)
        ])

        self.up_blocks = nn.ModuleList([
            UpSampleBlock(
                in_channels = 2 * down_channels[i],
                out_channels = down_channels[i-1] if i != 0 else out_channels,
                upsample = True,
                num_layers = num_layers,
                norm_channels = norm_channels,
                num_heads = num_heads,
                t_emb_dim = t_emb_dim,
                expects_down=True
            )
            for i in reversed(range(len(down_channels) - 1))
        ])

        self.norm_out = nn.GroupNorm(num_groups=norm_channels,num_channels=out_channels)
        self.silu = nn.SiLU()
        self.conv_out = nn.Conv2d(in_channels=out_channels,out_channels=in_channels,kernel_size=3,padding=1)

    def forward(self, x : Tensor, t : Tensor) -> Tensor:

        x = self.conv_in(x)

        t = self.time_embedding(t)
        t = self.t_proj(t)

        downs = []

        for block in self.down_blocks:
            downs.append(x)
            x = block(x,t)

        for block in self.mid_blocks:
            x = block(x,t)

        for block in self.up_blocks:
            x = block(x,t,downs.pop())

        x = self.norm_out(x)
        x = self.silu(x)
        x = self.conv_out(x)

        activation = self.activations[self.output_activation]

        x = activation(x)

        return x