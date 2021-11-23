import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from unet_parts import *


class ConvBlockBN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlockBN, self).__init__()

        self.conv = nn.Sequential(
            nn.BatchNorm2d(num_features = in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias=False),
            nn.ReLU(inplace=True),

            nn.BatchNorm2d(num_features = out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias=False),
            nn.ReLU(inplace=True),)

    def forward(self, x):
        return self.conv(x)

class Down_Block(nn.Module):
    def init(self):
        super(Down_Block, self).__init__()


class UNet_spoco(nn.Module):
    def __init__(self, in_channels, out_channels, features=[16, 32, 64, 128, 256, 512]):

        super(UNet_spoco, self).__init__()

        self.max_pool_2d = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 1 )

        self.down_1 = ConvBlockBN(in_channels, 16)
        self.down_2 = ConvBlockBN(16, 32)
        self.down_3 = ConvBlockBN(32, 64)
        self.down_4 = ConvBlockBN(64, 128)
        self.down_5 = ConvBlockBN(128, 256)
        self.down_6 = ConvBlockBN(256, 512)

        self.upsample_1 = nn.ConvTranspose2d(512, 256, kernel_size = 2, stride = 2,)
        self.upsample_2 = nn.ConvTranspose2d(256, 128,  kernel_size = 2, stride = 2)
        self.upsample_3 = nn.ConvTranspose2d(128, 64, kernel_size = 2, stride = 2)
        self.upsample_4 = nn.ConvTranspose2d(64, 32, kernel_size = 2, stride = 2)
        self.upsample_5 = nn.ConvTranspose2d(32, 16, kernel_size = 2, stride = 2)
        #new:
        self.upsample_6 = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2)
        self.upsample_7 = nn.ConvTranspose2d(8, 4, kernel_size=2, stride=2)

        self.up_1 = ConvBlockBN(512, 256)
        self.up_2 = ConvBlockBN(256, 128)
        self.up_3 = ConvBlockBN(128, 64)
        self.up_4 = ConvBlockBN(64, 32)
        self.up_5 = ConvBlockBN(32, 16)
        #new:
        self.up_6 = ConvBlockBN(16, 8)
        self.up_7 = ConvBlockBN(8, 4)


    def forward(self, image):

        #encoder
        #print('Step 0: ' + str(image.shape) + '\n')

        #print('Encoder:')
        x1 = self.down_1(image)
        print('Down 1:', x1.shape)
        #print('Step 1: '+ str(x1.shape))
        x2 = self.max_pool_2d(x1)
        print('Max-Pool 1:', x2.shape)
        #print('Step 2: '+ str(x2.shape))

        x3 = self.down_2(x2)
        print('Down 2:', x3.shape)
        #print('Step 3: ' + str(x3.shape))
        x4 = self.max_pool_2d(x3)
        print('Max-Pool 2:', x4.shape)
        #print('Step 4: ' + str(x4.shape))

        x5 = self.down_3(x4)
        print('Down 3:', x5.shape)
        #print('Step 5: ' + str(x5.shape))
        x6 = self.max_pool_2d(x5)
        print('Max-Pool 3:', x6.shape)
        #print('Step 6: ' + str(x6.shape))

        x7 = self.down_4(x6)
        print('Down 4:', x7.shape)
        #print('Step 7: ' + str(x7.shape))
        x8 = self.max_pool_2d(x7)
        print('Max-Pool 4:', x8.shape)
        #print('Step 8: ' + str(x8.shape))

        x9 = self.down_5(x8)
        print('Down 5:', x9.shape)
        #print('Step 9: ' + str(x9.shape))
        x10 = self.max_pool_2d(x9)
        print('Max-Pool 5:', x10.shape)
        #print('Step 10: ' + str(x10.shape))

        x11 = self.down_6(x10)
        print('Down 6', x11.shape)
        #print('Step 11:' + str(x11.shape)+ '\n')

        #decoder part
        #print('Decoder:')

        x12 = self.upsample_1(x11)
        x13 = TF.resize(x12, x9.shape[2:])  #option 1: resize the image from the encoder to the size of the image after the bottleneck
        x14 = torch.cat((x9, x13), dim=1)

        x15 = self.up_1(x14)
        x16 = self.upsample_2(x15)
        x17 = TF.resize(x16, x7.shape[2:])
        x18 = torch.cat((x7, x17), dim = 1)

        x19 = self.up_2(x18)
        x20 = self.upsample_3(x19)

        x21 = TF.resize(x20, x5.shape[2:])
        x22 = torch.cat((x5, x21), dim = 1)


        x23 = self.up_3(x22)
        x24 = self.upsample_4(x23)

        x25 = TF.resize(x24, x3.shape[2:])
        x26 = torch.cat((x3, x25), dim=1)

        x27 = self.up_4(x26)
        x28 = self.upsample_5(x27)

        x29 = TF.resize(x28, x1.shape[2:])
        x30 = torch.cat((x1, x29), dim=1)

        x31 = self.up_5(x30)

        #print(x31.shape)
        return x31


class UNet2(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet2, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

def test():
    x = torch.randn((1, 4, 400, 400))

    model = UNet_spoco(in_channels = 4, out_channels = 16)
    preds = model(x)
    #print(preds.shape)


if __name__ == "__main__":
    test()