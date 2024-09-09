import torch
from torch import nn,Tensor
from .layers import DownSampleBlock,MidBlock,UpSampleBlock

class VQVAE(nn.Module):

    def __init__(self,
        in_channels : int,
        down_channels : list[int],
        mid_channels : list[int],
        num_layers : int,
        norm_channels : int,
        z_dim : int,
        codebook_size : int,
        num_heads : int,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings = codebook_size,embedding_dim = z_dim)

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=down_channels[0],kernel_size=3,padding=1),
            *
            [
                DownSampleBlock(
                    in_channels = down_channels[i],
                    out_channels = down_channels[i+1],
                    downsample = True,
                    num_layers = num_layers,
                    norm_channels = norm_channels,
                )
                for i in range(len(down_channels)-1)
            ],
            *
            [
                MidBlock(
                    in_channels = mid_channels[i],
                    out_channels = mid_channels[i+1],
                    num_layers = num_layers,
                    norm_channels = norm_channels,
                    num_heads = num_heads,
                )
                for i in range(len(mid_channels)-1)
            ],
            nn.GroupNorm(num_groups=norm_channels,num_channels=down_channels[-1]),
            nn.SiLU(),
            nn.Conv2d(in_channels=down_channels[-1],out_channels=z_dim,kernel_size=3,padding=1),
            nn.Conv2d(in_channels=z_dim,out_channels=z_dim,kernel_size=1),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels=z_dim,out_channels=z_dim,kernel_size=1),
            nn.Conv2d(in_channels=z_dim,out_channels=mid_channels[-1],kernel_size=3,padding=1),
            *
            [
                MidBlock(
                    in_channels = mid_channels[i],
                    out_channels = mid_channels[i-1],
                    num_layers = num_layers,
                    norm_channels = norm_channels,
                    num_heads = num_heads,
                )
                for i in reversed(range(1,len(mid_channels)))
            ],
            *
            [
                UpSampleBlock(
                    in_channels = down_channels[i],
                    out_channels = down_channels[i-1],
                    upsample = True,
                    num_layers = num_layers,
                    norm_channels = norm_channels,
                )
                for i in reversed(range(1,len(down_channels)))
            ],
            nn.GroupNorm(num_groups=norm_channels,num_channels=down_channels[0]),
            nn.SiLU(),
            nn.Conv2d(in_channels=down_channels[0],out_channels=in_channels,kernel_size=3,padding=1),
            nn.Tanh(),
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