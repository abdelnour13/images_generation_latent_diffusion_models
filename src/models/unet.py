from torch import nn, Tensor
from .layers import DownSampleBlock, MidBlock, UpSampleBlock, TimeEmbedding
from typing import Literal
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class UNetConfig:
    in_channels : int = 3
    out_channels : int = 3
    last_channels : int = 32
    down_channels : list[int] = field(default_factory=lambda: [128,192,256,384])
    mid_channels : list[int] = field(default_factory=lambda: [384,256])
    num_layers : int = 2
    norm_channels : int = 32
    num_heads : int = 8
    t_emb_dim : int = 512
    context_dim : Optional[int] = None
    output_activation : Literal['sigmoid','tanh','linear'] = 'linear'
    cross_att_num_heads : Optional[int] = None

class UNet(nn.Module):

    def __init__(self,config : UNetConfig) -> None:
        super().__init__()

        self.config = config

        self.activations = nn.ModuleDict({
            'sigmoid' : nn.Sigmoid(),
            'tanh' : nn.Tanh(),
            'linear' : nn.Identity(),
        })

        self.time_embedding = TimeEmbedding(config.t_emb_dim)

        self.t_proj = nn.Sequential(
            nn.Linear(config.t_emb_dim,config.t_emb_dim),
            nn.SiLU(),
            nn.Linear(config.t_emb_dim,config.t_emb_dim),
        )

        self.conv_in = nn.Conv2d(in_channels=config.in_channels,out_channels=config.down_channels[0],kernel_size=3,padding=1)

        self.down_blocks = nn.ModuleList([
            DownSampleBlock(
                in_channels = config.down_channels[i],
                out_channels = config.down_channels[i+1],
                downsample = True,
                num_layers = config.num_layers,
                norm_channels = config.norm_channels,
                num_heads = config.num_heads,
                t_emb_dim = config.t_emb_dim,
                context_dim = config.context_dim,
                cross_att_num_heads = config.cross_att_num_heads
            )
            for i in range(len(config.down_channels)-1)
        ])

        self.mid_blocks = nn.ModuleList([
            MidBlock(
                in_channels = config.mid_channels[i],
                out_channels = config.mid_channels[i+1],
                num_layers = config.num_layers,
                norm_channels = config.norm_channels,
                num_heads = config.num_heads,
                t_emb_dim = config.t_emb_dim,
                context_dim = config.context_dim,
                cross_att_num_heads = config.cross_att_num_heads
            )
            for i in range(len(config.mid_channels)-1)
        ])

        self.up_blocks = nn.ModuleList([
            UpSampleBlock(
                in_channels = 2 * config.down_channels[i],
                out_channels = config.down_channels[i-1] if i != 0 else config.last_channels,
                upsample = True,
                num_layers = config.num_layers,
                norm_channels = config.norm_channels,
                num_heads = config.num_heads,
                t_emb_dim = config.t_emb_dim,
                expects_down=True,
                context_dim = config.context_dim,
                cross_att_num_heads = config.cross_att_num_heads
            )
            for i in reversed(range(len(config.down_channels) - 1))
        ])

        self.norm_out = nn.GroupNorm(num_groups=config.norm_channels,num_channels=config.last_channels)
        self.silu = nn.SiLU()
        self.conv_out = nn.Conv2d(in_channels=config.last_channels,out_channels=config.out_channels,kernel_size=3,padding=1)

    def forward(self, 
        x : Tensor, 
        t : Tensor,
        condition : Optional[Tensor] = None
    ) -> Tensor:

        x = self.conv_in(x)

        t = self.time_embedding(t)
        t = self.t_proj(t)

        downs = []

        for block in self.down_blocks:
            downs.append(x)
            x = block(x,t,condition)

        for block in self.mid_blocks:
            x = block(x,t,condition)

        for block in self.up_blocks:
            x = block(x,t,downs.pop(),condition)

        x = self.norm_out(x)
        x = self.silu(x)
        x = self.conv_out(x)

        activation = self.activations[self.config.output_activation]

        x = activation(x)

        return x