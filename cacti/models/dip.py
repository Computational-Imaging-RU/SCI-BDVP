import torch
import torch.nn as nn
import torch.nn.functional as F

""" U-Net model """


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool_conv = nn.Sequential(
            #nn.MaxPool2d(2),
            nn.AvgPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.pool_conv(x)


class Up_noskip(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x):
        x = self.up(x)
        return self.conv(x)     

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.tanh = torch.nn.Tanh()
    def forward(self, x):
        return self.tanh(self.conv(x))


class UNet_noskip(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_noskip, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024)
        self.up1 = Up_noskip(1024, 512, bilinear)
        self.up2 = Up_noskip(512, 256, bilinear)
        self.up3 = Up_noskip(256, 128, bilinear)
        self.up4 = Up_noskip(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        logits = self.outc(x)
        return logits



""" Decoder-Net model """

def conv(in_f, out_f, kernel_size, stride=1, pad='zero'):
    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == 'reflection':
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0
  
    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=False)

    layers = filter(lambda x: x is not None, [padder, convolver]) #extract values of non-None padder and convolver 
    return nn.Sequential(*layers)

     
        
class Downsample(torch.nn.Module): 
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
        super(Downsample, self).__init__()
        self.name = type(self).__name__
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, input):
        return F.interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners)




class autoencodernet(torch.nn.Module):
    def __init__(self, num_output_channels, num_channels_up, need_sigmoid=True, pad='reflection', upsample_mode='bilinear', bn_affine=True, decodetype='upsample', kernel_size=1):
        super(autoencodernet,self).__init__()
        self.decodetype = decodetype
        
        n_scales = len(num_channels_up)
        # print('n_scales=', n_scales, 'num_channels_up=', num_channels_up)
        
        if decodetype=='upsample':
            #decoder
            self.decoder = nn.Sequential()

            for i in range(n_scales-1):
                
                #if i!=0:
                module_name = 'dconv'+str(i)    
                self.decoder.add_module(module_name, conv(num_channels_up[i], num_channels_up[i+1], kernel_size, 1, pad=pad))

                if i != len(num_channels_up)-1:        
                    module_name = 'drelu' + str(i)
                    # self.decoder.add_module(module_name,nn.LeakyReLU())
                    self.decoder.add_module(module_name,nn.ReLU())
                    module_name = 'dbn' + str(i)
                    self.decoder.add_module(module_name,nn.BatchNorm2d(num_channels_up[i+1], affine=bn_affine))        

                module_name = 'dups' + str(i)
                self.decoder.add_module(module_name,nn.Upsample(scale_factor=2, mode=upsample_mode, align_corners=True))

            module_name = 'dconv' + str(i+1)
            self.decoder.add_module(module_name, conv(num_channels_up[-1], num_output_channels, kernel_size, 1, pad=pad))
        
            if need_sigmoid:
                self.decoder.add_module('sig',nn.Sigmoid())
                
        #encoder
        self.encoder = nn.Sequential()
        module_name = 'uconv'+str(n_scales-1)   
        self.encoder.add_module(module_name,conv(64,num_channels_up[-1], 1, pad=pad))
        
        for i in range(n_scales-2,-1,-1):
            
            if i != len(num_channels_up)-1:  
                module_name = 'urelu' + str(i)
                self.encoder.add_module(module_name,nn.ReLU())
                module_name = 'ubn' + str(i)
                self.encoder.add_module(module_name,nn.BatchNorm2d(num_channels_up[i+1], affine=bn_affine))     
                
            module_name = 'uconv'+str(i)
            self.encoder.add_module(module_name,conv(num_channels_up[i+1], num_channels_up[i],  1, 1, pad=pad))    
            module_name = 'udns'+str(i)
            self.encoder.add_module(module_name,Downsample(scale_factor=0.5, mode=upsample_mode, align_corners=True))

        if decodetype=='transposeconv':
            #convolutional decoder
            self.convdecoder = nn.Sequential()
            
            for i in range(n_scales-1):
                module_name = 'cdconv'+str(i) 
                
                if i==0:
                    self.convdecoder.add_module(module_name,conv(num_channels_up[i], num_channels_up[i+1], 1, 1, pad=pad))
                else:
                    self.convdecoder.add_module(module_name,nn.ConvTranspose2d(num_channels_up[i], num_channels_up[i+1],2,2)) 

                if i != len(num_channels_up)-1:        
                    module_name = 'cdrelu' + str(i)
                    self.convdecoder.add_module(module_name,nn.ReLU())   
                    module_name = 'cdbn' + str(i)
                    self.convdecoder.add_module(module_name,nn.BatchNorm2d(num_channels_up[i+1], affine=bn_affine))        

            module_name = 'cdconv' + str(i+2)
            self.convdecoder.add_module(module_name,nn.ConvTranspose2d(num_channels_up[-1], num_output_channels, 2, 2)) 
            
            if need_sigmoid:
                self.convdecoder.add_module('sig',nn.Sigmoid())
    def forward(self,x):
        if self.decodetype=='upsample':
            x = self.decoder(x)
        elif self.decodetype=='transposeconv':
            x = self.convdecoder(x)
        return x