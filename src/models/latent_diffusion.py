import sys
import torch
sys.path.append('../..')
from torch import nn,Tensor
from src.transforms import NoiseScheduler,NoiseSchedulerConfig
from dataclasses import dataclass,field
from typing import Literal,Optional
from .layers import CategoricalFeaturesEncoder
from .vqvae import VQVAEConfig,VQVAE
from .unet import UNetConfig,UNet
from tqdm.auto import tqdm

@dataclass
class MetadataCondConfig:
    config : dict = field(default_factory=dict)
    num_layers : int = 4
    dropout : float = 0.1
    condition_mask_rate : float = 0.1

@dataclass
class MaskCondConfig:
    mask_dim_in : int = 1
    mask_dim_out : int = 1
    condition_mask_rate : float = 0.1

@dataclass
class ColorPaletteCondConfig:
    dim_in : int = 3
    dim_out : int = 32
    condition_mask_rate : float = 0.1

@dataclass
class LatentDiffusionConfig:

    vqvae : VQVAEConfig = field(default_factory=VQVAEConfig)
    unet : UNetConfig = field(default_factory=UNetConfig)
    noise_scheduler : NoiseSchedulerConfig = field(default_factory=NoiseSchedulerConfig)
    input_type : Literal['image','latent'] = 'latent'

    metadata_cond : Optional[MetadataCondConfig] = None
    mask_cond : Optional[MaskCondConfig] = None
    color_palette_cond : Optional[ColorPaletteCondConfig] = None

    def __post_init__(self):

        if isinstance(self.vqvae,dict):
            self.vqvae = VQVAEConfig(**self.vqvae)

        if isinstance(self.unet,dict):
            self.unet = UNetConfig(**self.unet)

        if isinstance(self.noise_scheduler,dict):
            self.noise_scheduler = NoiseSchedulerConfig(**self.noise_scheduler)

        if self.metadata_cond is not None and isinstance(self.metadata_cond,dict):
            self.metadata_cond = MetadataCondConfig(**self.metadata_cond)

        if self.mask_cond is not None and isinstance(self.mask_cond,dict):
            self.mask_cond = MaskCondConfig(**self.mask_cond)

        if self.color_palette_cond is not None and isinstance(self.color_palette_cond,dict):
            self.color_palette_cond = ColorPaletteCondConfig(**self.color_palette_cond)

class LatentDiffusion(nn.Module):

    def __init__(self, config : LatentDiffusionConfig) -> None:
        
        super().__init__()

        self.config = config

        self.vqvae = VQVAE(config.vqvae).eval()
        self.unet = UNet(config.unet)
        self.noise_scheduler = NoiseScheduler(config.noise_scheduler)

        self.metadata_cond = CategoricalFeaturesEncoder(
            config=config.metadata_cond.config,
            dim=config.unet.context_dim,
            num_layers=config.metadata_cond.num_layers,
            dropout=config.metadata_cond.dropout
        ) if config.metadata_cond is not None else None

        self.mask_conv = nn.Conv2d(
            in_channels=config.mask_cond.mask_dim_in,
            out_channels=config.mask_cond.mask_dim_out,
            kernel_size=1,
        ) if config.mask_cond is not None else None

        self.color_palette_cond = nn.Linear(
            config.color_palette_cond.dim_in,
            config.color_palette_cond.dim_out
        ) if config.color_palette_cond is not None else None

        if config.input_type == 'latent':
            self.vqvae.encoder = nn.Identity()

    def forward(self, 
        x : Tensor,
        metadata : Optional[dict[str,Tensor]] = None,
        mask : Optional[Tensor] = None,
        color_palette : Optional[Tensor] = None
    ) -> tuple[Tensor,Tensor,Tensor]:
        
        # Get the compressed representation of the input
        # if the input type is a raw image
        if self.config.input_type == 'image':
            with torch.inference_mode():
                x = self.vqvae.encoder(x)
                x = self.vqvae.quantize(x)
                x = x['quant_out']

        # Handle conditioning with metadata
        if metadata is not None:

            assert self.metadata_cond is not None, 'Metadata Condition Config must be provided'

            if torch.rand(1).item() > self.config.metadata_cond.condition_mask_rate or not self.training:
                metadata = self.metadata_cond(metadata)
            else:
                metadata = None

        # Add noise to the input
        noised_x,noise,t = self.noise_scheduler(x)

        # Handle conditioning with mask
        if mask is not None and self.config.mask_cond is None:
            raise ValueError('Mask Condition Config must be provided')

        if self.config.mask_cond is not None:

            if self.training:

                a = torch.rand(mask.size(0),device=mask.device) < self.config.mask_cond.condition_mask_rate
                a = a.float().reshape(-1,1,1,1)
                mask = mask * a

            mask = self.mask_conv(mask)
            noised_x = torch.cat([noised_x,mask],dim=1)

        # Handle conditioning with color palette
        if color_palette is not None:

            assert self.color_palette_cond is not None, 'Color Palette Condition Config must be provided'

            if torch.rand(1).item() > self.config.color_palette_cond.condition_mask_rate or not self.training:
                color_palette = self.color_palette_cond(color_palette)
            else:
                color_palette = None

            if metadata is not None:
                metadata = torch.cat([metadata,color_palette],dim=1)
            else:
                metadata = color_palette
        
        # UNet forward pass
        predicted_noise = self.unet(noised_x,t,metadata)

        return predicted_noise,noise,t
    
    def generate(self, 
        x : Tensor,
        metadata : Optional[dict[str,Tensor]] = None,
        mask : Optional[Tensor] = None,
        color_palette : Optional[Tensor] = None,
        *,
        cf_scale : float = 1.0,
        decode_every : Optional[int] = None,
        progress : bool = True,
        device : Optional[str] = None
    ) -> tuple[Tensor,list[Tensor]]:

        assert not self.training, 'Model must be in eval mode to generate samples'
        assert cf_scale == 1.0 or metadata is not None, 'Metadata must be provided for classifier free guidance'
        assert mask is None or self.config.mask_cond is not None, 'Mask Condition Config must be provided'

        decoded = []
        timesteps = self.config.noise_scheduler.timesteps

        with torch.inference_mode():

            if metadata is not None:

                for key,(num_classes,_) in self.config.metadata_cond.config.items():
                    if key not in metadata:
                        metadata[key] = torch.zeros(x.size(0),dtype=torch.long,device=x.device).fill_(num_classes)

                metadata = self.metadata_cond(metadata)

            if mask is None and self.config.mask_cond is not None:
                dim_in = self.config.mask_cond.mask_dim_in
                mask = torch.zeros(x.size(0),dim_in,x.size(2),x.size(3),device=x.device)

            if color_palette is not None:
                    
                color_palette = self.color_palette_cond(color_palette)
    
                if metadata is not None:
                    metadata = torch.cat([metadata,color_palette],dim=1)
                else:
                    metadata = color_palette

            iterator = reversed(range(timesteps))

            if progress:
                iterator = tqdm(iterator,total=timesteps)
            
            for t in iterator:

                t = torch.tensor(t).repeat(x.size(0)).to(x.device)

                unet_in = torch.cat([x,mask],dim=1) if mask is not None else x
                noise = self.unet.forward(unet_in,t,metadata)

                if cf_scale > 1.0:
                    noise_uncond = self.unet.forward(x,t,None)
                    noise = noise_uncond + cf_scale * (noise - noise_uncond)

                x,x0 = self.noise_scheduler.denoise(x,noise,t)

                if decode_every is not None and t[0].item() % decode_every == 0:

                    x0 = self.vqvae.decoder(x0)

                    if device is not None:
                        x0 = x0.to(device)

                    decoded.append(x0)

            x = self.vqvae.decoder(x)

            if device is not None:
                x = x.to(device)

        decoded = torch.stack(decoded) if len(decoded) > 0 else x.unsqueeze(0)

        ### Post Processing ###
        match self.config.vqvae.output_activation:
            case 'tanh':
                x = (x + 1.0) / 2.0
                decoded = (decoded + 1.0) / 2.0
            case 'linear':
                x = torch.clamp(x,0.0,1.0)
                decoded = torch.clamp(decoded,0.0,1.0)
            case 'sigmoid':
                pass

        return x,decoded