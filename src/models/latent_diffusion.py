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

@dataclass
class LatentDiffusionConfig:

    vqvae : VQVAEConfig = field(default_factory=VQVAEConfig)
    unet : UNetConfig = field(default_factory=UNetConfig)
    noise_scheduler : NoiseSchedulerConfig = field(default_factory=NoiseSchedulerConfig)
    input_type : Literal['image','latent'] = 'latent'

    metadata_cond : Optional[MetadataCondConfig] = None

    condition_mask_rate : float = 0.1

    def __post_init__(self):

        if isinstance(self.vqvae,dict):
            self.vqvae = VQVAEConfig(**self.vqvae)

        if isinstance(self.unet,dict):
            self.unet = UNetConfig(**self.unet)

        if isinstance(self.noise_scheduler,dict):
            self.noise_scheduler = NoiseSchedulerConfig(**self.noise_scheduler)

        if self.metadata_cond is not None and isinstance(self.metadata_cond,dict):
            self.metadata_cond = MetadataCondConfig(**self.metadata_cond)

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

        if config.input_type == 'latent':
            self.vqvae.encoder = nn.Identity()

    def forward(self, 
        x : Tensor,
        metadata : Optional[dict[str,Tensor]] = None
    ) -> tuple[Tensor,Tensor,Tensor]:
        
        if self.config.input_type == 'image':
            with torch.inference_mode():
                x = self.vqvae.encoder(x)
                x = self.vqvae.quantize(x)
                x = x['quant_out']

        if metadata is not None:

            assert self.metadata_cond is not None, 'Metadata Condition Config must be provided'

            if torch.rand(1).item() > self.config.condition_mask_rate:
                metadata = self.metadata_cond(metadata)
            else:
                metadata = None

        noised_x,noise,t = self.noise_scheduler(x)
        predicted_noise = self.unet(noised_x,t,metadata)

        return predicted_noise,noise,t
    
    def generate(self, 
        x : Tensor,
        decode_every : Optional[int] = None,
        progress : bool = True,
        device : Optional[str] = None
    ) -> tuple[Tensor,list[Tensor]]:

        assert not self.training, 'Model must be in eval mode to generate samples'

        decoded = []
        timesteps = self.config.noise_scheduler.timesteps

        with torch.inference_mode():

            iterator = reversed(range(timesteps))

            if progress:
                iterator = tqdm(iterator,total=timesteps)
            
            for t in iterator:

                t = torch.tensor(t).repeat(x.size(0)).to(x.device)
                noise = self.unet(x,t)
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