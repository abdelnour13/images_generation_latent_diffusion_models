from torch import nn,Tensor
from dataclasses import dataclass,field

@dataclass
class DiscriminatorConfig:
    channels : list[int] = field(default_factory=lambda: [3,64,128,256])
    filters : list[int] = field(default_factory=lambda: [4,4,4,4])
    strides : list[int] = field(default_factory=lambda: [2,2,2,1])
    paddings : list[int] = field(default_factory=lambda: [1,1,1,1])

class Discriminator(nn.Module):

    def __init__(self,config : DiscriminatorConfig = DiscriminatorConfig()) -> None:
        super(Discriminator, self).__init__()

        self.config = config
        self.n_layers = len(config.channels)

        self.layers = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(
                    in_channels=config.channels[i],
                    out_channels=config.channels[i+1],
                    kernel_size=config.filters[i],
                    stride=config.strides[i],
                    padding=config.paddings[i],
                    bias=False
                ),
                nn.BatchNorm2d(config.channels[i+1]) if i not in [0,self.n_layers - 2] else nn.Identity(),
                nn.LeakyReLU(0.2,inplace=True) if i != self.n_layers - 2 else nn.Identity()
            )
            for i in range(self.n_layers - 1)
        ])
        
        
    def forward(self, x : Tensor) -> Tensor:
        return self.layers(x)