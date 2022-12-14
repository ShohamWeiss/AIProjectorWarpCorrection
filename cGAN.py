# Generator Code
import torch
import torch.nn as nn
from torchsummary import summary
from torchviz import make_dot
import torch.nn.functional as F

def downsample(in_channels, out_channels, size, apply_batchnorm=True):
    result = nn.Sequential()
    result.add_module("layer",
        nn.Conv2d(in_channels, out_channels, size, stride=2, padding=(0,0,2,1)))

    if apply_batchnorm:
        result.add_module("norm",nn.BatchNorm2d(out_channels))

    result.add_module("relu", nn.LeakyReLU())

    return result

def upsample(in_channels, out_channels, size, apply_dropout=False):
    result = nn.Sequential()
    result.add_module("convtrans",
    nn.ConvTranspose2d(in_channels, out_channels, size, stride=2, padding='same'))

    result.add_module("norm", nn.BatchNorm2d(out_channels))

    if apply_dropout:
        result.add_module("dropout", nn.Dropout(0.5))

    result.add_module("relu", nn.ReLU())

    return result

class Generator(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        down_stack = [
            downsample(3, 64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
            downsample(64, 128, 4),  # (batch_size, 64, 64, 128)
            downsample(128, 256, 4),  # (batch_size, 32, 32, 256)
            downsample(256, 512, 4),  # (batch_size, 16, 16, 512)
            downsample(512, 512, 4),  # (batch_size, 8, 8, 512)
            downsample(512, 512, 4),  # (batch_size, 4, 4, 512)
            downsample(512, 512, 4),  # (batch_size, 2, 2, 512)
            downsample(512, 512, 4),  # (batch_size, 1, 1, 512)
        ]

        up_stack = [
            upsample(512, 512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
            upsample(512, 512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
            upsample(512, 512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
            upsample(512, 512, 4),  # (batch_size, 16, 16, 1024)
            upsample(512, 256, 4),  # (batch_size, 32, 32, 512)
            upsample(256, 128, 4),  # (batch_size, 64, 64, 256)
            upsample(128, 64, 4),  # (batch_size, 128, 128, 128)
        ]
        
        last = nn.ConvTranspose2d(128, 3, 4, stride=2, padding='same')  # (batch_size, 256, 256, 3)
        last_activation = nn.Tanh()
        
        self.layers = nn.Sequential()
        # Downsampling through the model
        skips = []
        for down in down_stack:
            self.layers.add_module(down)
            skips.append(x)

        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            self.layers.add_module()
            x = torch.concatenate([x, skip])

        x = last(x)
        x = last_activation(x)

        self.layers = x
    
    def forward(self, input):
        return self.layers(input)
    
if __name__=="__main__":
    gmod = Generator()
    summary(gmod, (3,256,256))
    
    # dmod = Discriminator()
    # summary(dmod, (3,256,256))