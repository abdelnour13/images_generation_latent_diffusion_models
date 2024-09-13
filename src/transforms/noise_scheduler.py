import torch
from torch import nn
from typing import Optional
from torch import Tensor

class NoiseScheduler(nn.Module):

    def __init__(self,
        timesteps : int,
        beta_start : float,
        beta_end : float,
    ) -> None:
        super(NoiseScheduler, self).__init__()

        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps) ** 2
        alphas = 1.0 - betas

        alpha_cumprod = alphas.cumprod(dim=0)
        sqrt_alpha_cumprod = alpha_cumprod ** 0.5
        sqrt_one_minus_alpha_cumprod = (1 - alpha_cumprod) ** 0.5

        self.register_buffer('alphas', alphas)
        self.register_buffer('betas', betas)
        self.register_buffer('alpha_cumprod', alpha_cumprod)
        self.register_buffer('sqrt_alpha_cumprod', sqrt_alpha_cumprod)
        self.register_buffer('sqrt_one_minus_alpha_cumprod', sqrt_one_minus_alpha_cumprod)

    def forward(self, image : Tensor, t : Optional[Tensor] = None) -> tuple[Tensor,Tensor,Tensor]:

        batch_mode = True

        if len(image.shape) == 3:

            batch_mode = False
            image = image.unsqueeze(0)

            if t is not None:
                t = t.unsqueeze(0)

        B,C,H,W = image.shape

        if t is not None and (t.min() < 0 or t.max() >= self.timesteps):
            raise ValueError(f"t must be None or in the range [0, {self.timesteps})")

        t = torch.randint(0, self.timesteps, (B,), device=image.device) if t is None else t
        noise = torch.randn_like(image).to(image.device)

        sqrt_alpha_cumprod = self.get_buffer('sqrt_alpha_cumprod')[t].reshape(B,1,1,1)
        sqrt_one_minus_alpha_cumprod = self.get_buffer('sqrt_one_minus_alpha_cumprod')[t].reshape(B,1,1,1)

        noised = sqrt_alpha_cumprod * image + sqrt_one_minus_alpha_cumprod * noise

        if not batch_mode:
            noised = noised.squeeze(0)
            noise = noise.squeeze(0)
            t = t.squeeze(0)
        
        return noised,noise,t
    
    def denoise(self, noised : Tensor,noise : Tensor,t : Tensor) -> tuple[Tensor,Tensor]:

        batch_mode = True

        if len(noised.shape) == 3:
            batch_mode = False
            noised = noised.unsqueeze(0)
            noise = noise.unsqueeze(0)

        B,C,H,W = noised.shape
        
        sqrt_alpha_cumprod = self.get_buffer('sqrt_alpha_cumprod')[t].reshape(B,1,1,1)
        sqrt_one_minus_alpha_cumprod = self.get_buffer('sqrt_one_minus_alpha_cumprod')[t].reshape(B,1,1,1)
        betas = self.get_buffer('betas')[t].reshape(B,1,1,1)
        alphas = self.get_buffer('alphas')[t].reshape(B,1,1,1)

        x0 = (noised - (sqrt_one_minus_alpha_cumprod * noise)) / sqrt_alpha_cumprod
        # x0 = torch.clamp(x0,-1,1)

        mean = noised - ((betas * noise) / sqrt_one_minus_alpha_cumprod)
        mean = mean / torch.sqrt(alphas)

        if t[0].item() == 0:
            xt = mean
        else:
            sigma = torch.sqrt(betas)
            z = torch.randn_like(noised).to(noised.device)
            xt = mean + sigma * z

        if not batch_mode:
            x0 = x0.squeeze(0)
            xt = xt.squeeze(0)

        return xt,x0