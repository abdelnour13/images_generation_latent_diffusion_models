import torch
from torch import nn, Tensor
from dataclasses import dataclass,field

@dataclass
class GeneratorConfig:
    latent_dim : int = 128
    channels : list[int] = field(default_factory=lambda: [3,128,64,32])
    out_channels : int = 3,
    output_activation : str = 'sigmoid'

class Generator(nn.Module):

    def __init__(self,config : GeneratorConfig = GeneratorConfig()) -> None:
        super(Generator, self).__init__()

        self.config = config

        self.conv_in = nn.Sequential(
            nn.ConvTranspose2d(config.latent_dim, config.channels[0], 4, 1, 0, bias=False),
            nn.BatchNorm2d(config.channels[0]),
            nn.ReLU(True),
        )

        self.blocks = nn.Sequential(*[
            nn.Sequential(
                nn.ConvTranspose2d(config.channels[i], config.channels[i+1], 4, 2, 1, bias=False),
                nn.BatchNorm2d(config.channels[i+1]),
                nn.ReLU(True),
            )
            for i in range(len(config.channels) - 1)
        ])

        self.conv_out =  nn.ConvTranspose2d(config.channels[-1], config.out_channels, 4, 2, 1, bias=False)

        self.activations = nn.ModuleDict({
            'sigmoid' : nn.Sigmoid(),
            'tanh' : nn.Tanh(),
        })

    def forward(self, x: Tensor) -> Tensor:

        x = self.conv_in(x)
        x = self.blocks(x)
        x = self.conv_out(x)

        activation = self.activations[self.config.output_activation]
        x = activation(x)

        return x
    
    def generate(self,x : Tensor) -> Tensor:

        assert not self.training, 'generate should be called in eval mode'

        with torch.inference_mode():
            x = self.forward(x)

        if self.config.output_activation == 'tanh':
            x = (x + 1) / 2

        return x