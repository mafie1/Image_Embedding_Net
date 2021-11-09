import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


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
        #print('Step 1: '+ str(x1.shape))
        x2 = self.max_pool_2d(x1)
        #print('Step 2: '+ str(x2.shape))

        x3 = self.down_2(x2)
        #print('Step 3: ' + str(x3.shape))
        x4 = self.max_pool_2d(x3)
        #print('Step 4: ' + str(x4.shape))

        x5 = self.down_3(x4)
        #print('Step 5: ' + str(x5.shape))
        x6 = self.max_pool_2d(x5)
        #print('Step 6: ' + str(x6.shape))

        x7 = self.down_4(x6)
        #print('Step 7: ' + str(x7.shape))
        x8 = self.max_pool_2d(x7)
        #print('Step 8: ' + str(x8.shape))

        x9 = self.down_5(x8)
        #print('Step 9: ' + str(x9.shape))
        x10 = self.max_pool_2d(x9)
        #print('Step 10: ' + str(x10.shape))

        x11 = self.down_6(x10)
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



def test():
    x = torch.randn((1, 3, 50, 50))

    model = UNet_spoco(in_channels = 3, out_channels = 16)
    preds = model(x)
    #print(preds.shape)


if __name__ == "__main__":
    test()