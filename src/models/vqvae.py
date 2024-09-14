import torch
from torch import nn,Tensor
from .layers import DownSampleBlock,MidBlock,UpSampleBlock
from dataclasses import dataclass,field

@dataclass
class VQVAEConfig:
    in_channels : int = 3
    down_channels : list[int] = field(default_factory=lambda:[64,64,128])
    mid_channels : list[int] = field(default_factory=lambda:[128,128])
    num_layers : int = 2
    norm_channels : int = 32
    z_dim : int = 3
    codebook_size : int = 2048
    num_heads : int = 4
    output_activation : str = 'tanh'

class VQVAE(nn.Module):

    def __init__(self,config : VQVAEConfig) -> None:
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings = config.codebook_size,embedding_dim = config.z_dim)

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=config.in_channels,out_channels=config.down_channels[0],kernel_size=3,padding=1),
            *
            [
                DownSampleBlock(
                    in_channels = config.down_channels[i],
                    out_channels = config.down_channels[i+1],
                    downsample = True,
                    num_layers = config.num_layers,
                    norm_channels = config.norm_channels,
                )
                for i in range(len(config.down_channels)-1)
            ],
            *
            [
                MidBlock(
                    in_channels = config.mid_channels[i],
                    out_channels = config.mid_channels[i+1],
                    num_layers = config.num_layers,
                    norm_channels = config.norm_channels,
                    num_heads = config.num_heads,
                )
                for i in range(len(config.mid_channels)-1)
            ],
            nn.GroupNorm(num_groups=config.norm_channels,num_channels=config.down_channels[-1]),
            nn.SiLU(),
            nn.Conv2d(in_channels=config.down_channels[-1],out_channels=config.z_dim,kernel_size=3,padding=1),
            nn.Conv2d(in_channels=config.z_dim,out_channels=config.z_dim,kernel_size=1),
        )

        self.activations = {
            'tanh' : nn.Tanh(),
            'sigmoid' : nn.Sigmoid(),
            'linear' : nn.Identity(),
        }

        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels=config.z_dim,out_channels=config.z_dim,kernel_size=1),
            nn.Conv2d(in_channels=config.z_dim,out_channels=config.mid_channels[-1],kernel_size=3,padding=1),
            *
            [
                MidBlock(
                    in_channels = config.mid_channels[i],
                    out_channels = config.mid_channels[i-1],
                    num_layers = config.num_layers,
                    norm_channels = config.norm_channels,
                    num_heads = config.num_heads,
                )
                for i in reversed(range(1,len(config.mid_channels)))
            ],
            *
            [
                UpSampleBlock(
                    in_channels = config.down_channels[i],
                    out_channels = config.down_channels[i-1],
                    upsample = True,
                    num_layers = config.num_layers,
                    norm_channels = config.norm_channels,
                )
                for i in reversed(range(1,len(config.down_channels)))
            ],
            nn.GroupNorm(num_groups=config.norm_channels,num_channels=config.down_channels[0]),
            nn.SiLU(),
            nn.Conv2d(in_channels=config.down_channels[0],out_channels=config.in_channels,kernel_size=3,padding=1),
            self.activations[config.output_activation],
        )

    def quantize(self,x : Tensor) -> dict:
        
        B,C,H,W = x.shape

        ### Flatten x
        x = x.permute(0,2,3,1) # B,H,W,C
        x = x.reshape(B,H*W,C) # B,H*W,C

        ### Calculate distances
        distances = torch.cdist(x,self.embedding.weight[None,:].repeat(B,1,1))
        indices = torch.argmin(distances,dim=-1)

        ### Quantize
        quant_out = torch.index_select(self.embedding.weight, 0, indices.view(-1)) # B*H*W,C

        ### Commitment loss
        x = x.reshape(-1,C) # B*H*W,C
        commitment_loss = x - quant_out.detach()
        commitment_loss = torch.pow(commitment_loss,2)
        commitment_loss = commitment_loss.mean()

        ### Codebook loss
        codebook_loss = x.detach() - quant_out
        codebook_loss = torch.pow(codebook_loss,2)
        codebook_loss = commitment_loss.mean()

        ### Quantize output
        quant_out = x + (quant_out - x).detach()

        ### Reshape
        quant_out = quant_out.reshape(B,H,W,C) # B,H,W,C
        quant_out = quant_out.permute(0,3,1,2) # B,C,H,W
        indices = indices.reshape(B,H,W)

        output = {
            "quant_out" : quant_out,
            "indices" : indices,
            "commitment_loss" : commitment_loss,
            "codebook_loss" : codebook_loss,
        }

        return output

    def forward(self,x : Tensor) -> dict:

        ### Encoder
        enc_out = self.encoder(x)

        ### Quantize
        quant_out = self.quantize(enc_out)

        ### Decoder
        dec_out = self.decoder(quant_out["quant_out"])

        ### Output
        out = {
            "dec_out" : dec_out,
            "quant_out" : quant_out["quant_out"],
            "commitment_loss" : quant_out["commitment_loss"],
            "codebook_loss" : quant_out["codebook_loss"],
        }

        return out