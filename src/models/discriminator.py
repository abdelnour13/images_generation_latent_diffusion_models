from torch import nn,Tensor

class Discriminator(nn.Module):

    def __init__(self,
        channels : list = [3,64,128,256],
        filters : list = [4,4,4,4],
        strides : list = [2,2,2,1],
        paddings : list = [1,1,1,1],
    ) -> None:
        super(Discriminator, self).__init__()

        self.channels = channels + [1]
        self.filters = filters
        self.strides = strides
        self.paddings = paddings
        self.n_layers = len(self.channels)

        self.layers = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(
                    in_channels=self.channels[i],
                    out_channels=self.channels[i+1],
                    kernel_size=self.filters[i],
                    stride=self.strides[i],
                    padding=self.paddings[i],
                    bias=False
                ),
                nn.BatchNorm2d(self.channels[i+1]) if i not in [0,self.n_layers - 2] else nn.Identity(),
                nn.LeakyReLU(0.2,inplace=True) if i != self.n_layers - 2 else nn.Identity()
            )
            for i in range(self.n_layers - 1)
        ])
        
        
    def forward(self, x : Tensor) -> Tensor:
        return self.layers(x)