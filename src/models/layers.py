import torch
from torch import nn,Tensor

class ResBlock(nn.Module):

    def __init__(self,
        in_channels : int,
        out_channels : int,
        norm_channels : int,
        kernel_size : int,
        stride : int,
        padding : int,
        num_heads : int | None = None,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_channels = norm_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.num_heads = num_heads

        self.gn1 = nn.GroupNorm(norm_channels,in_channels)
        self.silu1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding)

        self.gn2 = nn.GroupNorm(norm_channels,out_channels)
        self.silu2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding)

        self.conv_3 = nn.Conv2d(in_channels,out_channels,kernel_size=1)

        self.att_gn = nn.GroupNorm(norm_channels,out_channels) if num_heads is not None else None
        self.att = nn.MultiheadAttention(out_channels,num_heads) if num_heads is not None else None

    def forward(self,x : Tensor) -> Tensor:
        
        residual = x

        x = self.gn1(x)
        x = self.silu1(x)
        x = self.conv1(x)

        x = self.gn2(x)
        x = self.silu2(x)
        x = self.conv2(x)

        residual = self.conv_3(residual)

        x = x + residual

        if self.num_heads is not None:

            B,C,H,W = x.shape

            att_in = x.view(B,C,H*W)
            att_in = self.att_gn(att_in)
            att_in = torch.transpose(att_in,1,2)

            att_out,_ = self.att(att_in,att_in,att_in)
            att_out = torch.transpose(att_out,1,2)
            att_out = att_out.view(B,C,H,W)

            x = x + att_out

        return x

class DownSampleBlock(nn.Module):

    def __init__(self,
        in_channels : int,
        out_channels : int,
        downsample : bool,
        num_layers : int,
        norm_channels : int,
    ) -> None:
        
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample
        self.num_layers = num_layers
        self.norm_channels = norm_channels

        self.layers = nn.Sequential(*[
            ResBlock(
                in_channels = in_channels if i == 0 else out_channels,
                out_channels = out_channels,
                norm_channels = norm_channels,
                kernel_size = 3,
                stride = 1,
                padding = 1,
            )
            for i in range(num_layers)
        ])

        self.downsample = nn.Conv2d(out_channels,out_channels,kernel_size=4,stride=2,padding=1) if downsample else nn.Identity()

    def forward(self,x : Tensor) -> Tensor:
        x = self.layers(x)
        x = self.downsample(x)
        return x
    
class MidBlock(nn.Module):

    def __init__(self,
        in_channels : int,
        out_channels : int,
        num_layers : int,
        norm_channels : int,
        num_heads : int | None = None,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.norm_channels = norm_channels

        self.layers = nn.Sequential(*[
            ResBlock(
                in_channels = out_channels,
                out_channels = out_channels,
                norm_channels = norm_channels,
                kernel_size = 3,
                stride = 1,
                padding = 1,
                num_heads = num_heads,
            )
            for i in range(num_layers)
        ])

        self.resnet_last = ResBlock(
            in_channels = in_channels,
            out_channels = out_channels,
            norm_channels = norm_channels,
            kernel_size = 3,
            stride = 1,
            padding = 1,
        )

    def forward(self,x : Tensor) -> Tensor:
        x = self.layers(x)
        x = self.resnet_last(x)
        return x
    
class UpSampleBlock(nn.Module):

    def __init__(self,
        in_channels : int,
        out_channels : int,
        upsample : bool,
        num_layers : int,
        norm_channels : int,
    ) -> None:
        
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample = upsample
        self.num_layers = num_layers
        self.norm_channels = norm_channels

        self.layers = nn.Sequential(*[
            ResBlock(
                in_channels = in_channels if i == 0 else out_channels,
                out_channels = out_channels,
                norm_channels = norm_channels,
                kernel_size = 3,
                stride = 1,
                padding = 1,
            )
            for i in range(num_layers)
        ])

        self.upsample = nn.ConvTranspose2d(in_channels,in_channels,kernel_size=4,stride=2,padding=1) if upsample else nn.Identity()

    def forward(self,x : Tensor) -> Tensor:
        x = self.upsample(x)
        x = self.layers(x)
        return x