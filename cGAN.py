# Generator Code
import torch
import torch.nn as nn
from torchsummary import summary
from torchviz import make_dot

class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_channels, inner_channels, input_channels=None,
                 submodule=None, is_outermost=False, is_innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = is_outermost
        if input_channels is None:
            input_channels = outer_channels
        downconv = nn.Conv2d(input_channels, inner_channels, kernel_size=4,
                             stride=2, padding=1, bias=False)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_channels)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_channels)
 
        if is_outermost:
            upconv = nn.ConvTranspose2d(inner_channels * 2, outer_channels,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif is_innermost:
            upconv = nn.ConvTranspose2d(inner_channels, outer_channels,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=False)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_channels * 2, outer_channels,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=False)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
 
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up
 
        self.model = nn.Sequential(*model)
 
    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)

class Generator(nn.Module):
    def __init__(self, input_channels, output_channels, num_filters=16, norm_layer=nn.BatchNorm2d, use_dropout=False):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        super(Generator, self).__init__()
        # construct unet structure
        # add the innermost block
        unet_block = UnetSkipConnectionBlock(num_filters * 64, num_filters * 128, input_channels=None, submodule=None, norm_layer=norm_layer, is_innermost=True) 
        #print(unet_block)
 
        # add intermediate block with nf * 8 filters
        unet_block = UnetSkipConnectionBlock(num_filters * 32, num_filters * 64, input_channels=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(num_filters * 16, num_filters * 32, input_channels=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(num_filters * 8, num_filters * 16, input_channels=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
 
        # gradually reduce the number of filters from nf * 8 to nf. 
        unet_block = UnetSkipConnectionBlock(num_filters * 4, num_filters * 8, input_channels=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(num_filters * 2, num_filters * 4, input_channels=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(num_filters, num_filters * 2, input_channels=None, submodule=unet_block, norm_layer=norm_layer)
         
        # add the outermost block
        self.model = UnetSkipConnectionBlock(output_channels, num_filters, input_channels=input_channels, submodule=unet_block, is_outermost=True, norm_layer=norm_layer)

    def forward(self, input):
        return self.model(input)
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        ndf = 8
        self.main = nn.Sequential(
            # input is (nc) x 256 x 256
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 128 x 128
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 64 x 64
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 32 x 32
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 16 x 16
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            # flatten 169
            nn.Flatten(),
            nn.LeakyReLU(),
            # 1 output
            nn.Linear(13*13,1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
    
if __name__=="__main__":
    gmod = Generator(3,3)
    summary(gmod, (3,256,256))
    
    dmod = Discriminator()
    summary(dmod, (3,256,256))