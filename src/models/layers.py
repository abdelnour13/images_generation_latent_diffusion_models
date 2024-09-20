import torch
from torch import nn,Tensor
from torch.nn import functional as F

class TimeEmbedding(nn.Module):

    def __init__(
        self,
        dim : int,
    ) -> None:
        super().__init__()

        self.dim = dim

        self._verify()

        weight = torch.pow(10000, torch.arange(dim // 2).float() / (dim // 2))
        self.register_buffer('weight', weight)

    def _verify(self):
        assert self.dim % 2 == 0, "dimension must be divisible by 2"

    def forward(self,x : Tensor) -> Tensor:

        x = x[:,None].repeat(1, self.dim // 2) / self.get_buffer('weight')
        x = torch.cat([x.sin(), x.cos()], dim=-1)

        return x

class ResBlock(nn.Module):

    def __init__(self,
        in_channels : int,
        out_channels : int,
        norm_channels : int,
        kernel_size : int,
        stride : int,
        padding : int,
        num_heads : int | None = None,
        t_emb_dim : int | None = None,
        context_dim : int | None = None,
        cross_att_num_heads : int | None = None,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_channels = norm_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.num_heads = num_heads
        self.t_emb_dim = t_emb_dim
        self.context_dim = context_dim
        self.cross_att_num_heads = cross_att_num_heads

        self.gn1 = nn.GroupNorm(norm_channels,in_channels)
        self.silu1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding)

        self.t_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim,out_channels)
        ) if t_emb_dim is not None else None
            
        self.gn2 = nn.GroupNorm(norm_channels,out_channels)
        self.silu2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding)

        self.conv_3 = nn.Conv2d(in_channels,out_channels,kernel_size=1)

        self.att_gn = nn.GroupNorm(norm_channels,out_channels) if num_heads is not None else None
        self.att = nn.MultiheadAttention(out_channels,num_heads,batch_first=True) if num_heads is not None else None

        self.cross_attn_gn = context_dim and nn.GroupNorm(norm_channels,out_channels)
        self.context_proj = context_dim and nn.Linear(context_dim,out_channels)
        self.cross_attn = context_dim and nn.MultiheadAttention(out_channels,cross_att_num_heads,batch_first=True)

    def forward(self,input : tuple[Tensor,Tensor | None,Tensor | None]) -> Tensor:
        
        x,t,context = input
                
        residual = x

        x = self.gn1(x)
        x = self.silu1(x)
        x = self.conv1(x)

        if self.t_emb_dim is not None:

            assert t is not None, "t must be provided if t_emb_dim is not None"

            t = self.t_emb(t)
            x = x + t[:,:,None,None]

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

        if context is not None:

            assert self.context_dim is not None, "context_dim must be provided when context is not None"

            B,C,H,W = x.shape

            att_in = x.view(B,C,H*W)
            att_in = self.cross_attn_gn(att_in)
            att_in = torch.transpose(att_in,1,2)

            context = self.context_proj(context)

            att_out,_ = self.cross_attn(att_in,context,context)
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
        num_heads : int | None = None,
        t_emb_dim : int | None = None,
        context_dim : int | None = None,
        cross_att_num_heads : int | None = None,
    ) -> None:
        
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample
        self.num_layers = num_layers
        self.norm_channels = norm_channels
        self.num_heads = num_heads
        self.t_emb_dim = t_emb_dim
        self.context_dim = context_dim
        self.cross_att_num_heads = cross_att_num_heads

        self.layers = nn.Sequential(*[
            ResBlock(
                in_channels = in_channels if i == 0 else out_channels,
                out_channels = out_channels,
                norm_channels = norm_channels,
                kernel_size = 3,
                stride = 1,
                padding = 1,
                t_emb_dim = t_emb_dim,
                num_heads = num_heads,
                context_dim = context_dim,
                cross_att_num_heads = cross_att_num_heads
            )
            for i in range(num_layers)
        ])

        self.downsample = nn.Conv2d(out_channels,out_channels,kernel_size=4,stride=2,padding=1) if downsample else nn.Identity()

    def forward(self,
        x : Tensor,
        t : Tensor | None = None,
        context : Tensor | None = None,
    ) -> Tensor:
        
        for layer in self.layers:
            x = layer((x,t,context))

        x = self.downsample(x)

        return x
    
class MidBlock(nn.Module):

    def __init__(self,
        in_channels : int,
        out_channels : int,
        num_layers : int,
        norm_channels : int,
        num_heads : int | None = None,
        t_emb_dim : int | None = None,
        context_dim : int | None = None,
        cross_att_num_heads : int | None = None,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.norm_channels = norm_channels
        self.num_heads = num_heads
        self.t_emb_dim = t_emb_dim
        self.context_dim = context_dim
        self.cross_att_num_heads = cross_att_num_heads

        self.layers = nn.Sequential(*[
            ResBlock(
                in_channels = in_channels if i == 0 else out_channels,
                out_channels = out_channels,
                norm_channels = norm_channels,
                kernel_size = 3,
                stride = 1,
                padding = 1,
                num_heads = num_heads,
                t_emb_dim = t_emb_dim,
                context_dim = context_dim,
                cross_att_num_heads = cross_att_num_heads
            )
            for i in range(num_layers)
        ])

        self.resnet_last = ResBlock(
            in_channels = out_channels,
            out_channels = out_channels,
            norm_channels = norm_channels,
            kernel_size = 3,
            stride = 1,
            padding = 1,
            t_emb_dim=t_emb_dim,
            context_dim=context_dim,
            cross_att_num_heads = cross_att_num_heads
        )

    def forward(self,
        x : Tensor,
        t : Tensor | None = None,
        context : Tensor | None = None
    ) -> Tensor:
        
        for layer in self.layers:
            x = layer((x,t,context))

        x = self.resnet_last((x,t,context))

        return x
    
class UpSampleBlock(nn.Module):

    def __init__(self,
        in_channels : int,
        out_channels : int,
        upsample : bool,
        num_layers : int,
        norm_channels : int,
        num_heads : int | None = None,
        t_emb_dim : int | None = None,
        context_dim : int | None = None,
        cross_att_num_heads : int | None = None,
        expects_down : bool = False,
    ) -> None:
        
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample = upsample
        self.num_layers = num_layers
        self.norm_channels = norm_channels
        self.num_heads = num_heads
        self.t_emb_dim = t_emb_dim
        self.expects_down = expects_down
        self.context_dim = context_dim
        self.cross_att_num_heads = cross_att_num_heads

        self.layers = nn.Sequential(*[
            ResBlock(
                in_channels = in_channels if i == 0 else out_channels,
                out_channels = out_channels,
                norm_channels = norm_channels,
                kernel_size = 3,
                stride = 1,
                padding = 1,
                t_emb_dim = t_emb_dim,
                num_heads = num_heads,
                context_dim = context_dim,
                cross_att_num_heads = cross_att_num_heads
            )
            for i in range(num_layers)
        ])

        if not upsample and expects_down:
            raise ValueError("expects_down is True but upsample is False")
        
        if not upsample:
            self.upsample = nn.Identity()
        else:
            if not expects_down:
                self.upsample = nn.ConvTranspose2d(in_channels,in_channels,kernel_size=4,stride=2,padding=1)
            else:
                self.upsample = nn.ConvTranspose2d(in_channels // 2,in_channels // 2,kernel_size=4,stride=2,padding=1)

    def forward(self,
        x : Tensor,
        t : Tensor | None = None,
        out_down : Tensor | None = None,
        context : Tensor | None = None,
    ) -> Tensor:
        
        x = self.upsample(x)

        if out_down is not None:
            x = torch.cat([x,out_down],dim=1)

        for layer in self.layers:
            x = layer((x,t,context))

        return x
    
class CategoricalFeaturesEncoder(nn.Module):

    def __init__(self, 
        config : dict[str,tuple[int,int]],
        dim : int,
        num_layers : int,
        dropout : float = 0.0
    ) -> None:
        super().__init__()

        self.config = config
        self.dim = dim
        self.num_layers = num_layers
        self.fc_in = sum([dim for _,dim in config.values()])
        self.fc_out = len(config) * dim
        self.dropout = dropout

        self.embeddings = nn.ModuleDict({
            name : nn.Embedding(num_classes+1,dim)
            for name,(num_classes,dim) in config.items()
        })

        self.dnn = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(self.fc_in if i == 0 else self.fc_out,self.fc_out),
                nn.ReLU() if i != num_layers - 1 else nn.Identity()
            )
            for i in range(self.num_layers)
        ])

    def forward(self, x : dict[str,Tensor]) -> Tensor:

        y = []

        for name,values in x.items():
            
            if self.dropout > 0.0 and self.training:

                num_classes = self.config[name][0]
                mask = torch.rand(values.shape, device=values.device) < self.dropout
                values = torch.where(mask,torch.zeros_like(values, device=values.device).fill_(num_classes).long(),values)
            
            embedding = self.embeddings[name](values)

            y.append(embedding)

        y = torch.cat(y,dim=-1)
        y = self.dnn(y)
        y = y.view(y.shape[0],-1,self.dim)

        return y