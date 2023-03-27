#############################################################################################
# Cristina Almagro-Pérez, ETH Zürich, 2022
#############################################################################################

# Networks: Attention U-Net, and Attention U-Net with deep supervision

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torchvision.transforms as T

###############################################################################
# Modules
###############################################################################

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv3d(ch_in, ch_out, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1, bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
            )

    def forward(self, x):
        x = self.up(x)
        return x

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1, bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch_out, ch_out, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1, bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

###############################################################################
# Network 1: Attention U-Net
###############################################################################

class AttU_Net(nn.Module):
    def __init__(self, img_ch=1, output_ch=2, features=[16, 32, 64, 128, 256]):
        super(AttU_Net, self).__init__()

        self.Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=features[0])  #64
        self.Conv2 = conv_block(ch_in=features[0], ch_out=features[1])
        self.Conv3 = conv_block(ch_in=features[1], ch_out=features[2])
        self.Conv4 = conv_block(ch_in=features[2], ch_out=features[3])
        self.Conv5 = conv_block(ch_in=features[3], ch_out=features[4])

        self.Up5 = up_conv(ch_in=features[4], ch_out=features[3])
        self.Att5 = Attention_block(F_g=features[3], F_l=features[3], F_int=features[2])
        self.Up_conv5 = conv_block(ch_in=features[4], ch_out=features[3])

        self.Up4 = up_conv(ch_in=features[3], ch_out=features[2])
        self.Att4 = Attention_block(F_g=features[2], F_l=features[2], F_int=features[1])
        self.Up_conv4 = conv_block(ch_in=features[3], ch_out=features[2])

        self.Up3 = up_conv(ch_in=features[2], ch_out=features[1])
        self.Att3 = Attention_block(F_g=features[1], F_l=features[1], F_int=features[0])
        self.Up_conv3 = conv_block(ch_in=features[2], ch_out=features[1])

        self.Up2 = up_conv(ch_in=features[1], ch_out=features[0])
        val = int(features[0]/2)
        self.Att2 = Attention_block(F_g=features[0], F_l=features[0], F_int=val)  #int(features[0]/2)
        self.Up_conv2 = conv_block(ch_in=features[1], ch_out=features[0])

        self.Conv_1x1 = nn.Conv3d(features[0], output_ch, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


###########################################################################################
#  Network 2: Attention U-Net with Deep Supervision
###########################################################################################
class AttU_Net_ds(nn.Module):
    def __init__(self, img_ch=1, output_ch=2, features=[16, 32, 64, 128, 256]):
        super(AttU_Net_ds, self).__init__()

        self.Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=features[0])  #64
        self.Conv2 = conv_block(ch_in=features[0], ch_out=features[1])
        self.Conv3 = conv_block(ch_in=features[1], ch_out=features[2])
        self.Conv4 = conv_block(ch_in=features[2], ch_out=features[3])
        self.Conv5 = conv_block(ch_in=features[3], ch_out=features[4])

        self.Up5 = up_conv(ch_in=features[4], ch_out=features[3])
        self.Att5 = Attention_block(F_g=features[3], F_l=features[3], F_int=features[2])
        self.Up_conv5 = conv_block(ch_in=features[4], ch_out=features[3])

        self.Up4 = up_conv(ch_in=features[3], ch_out=features[2])
        self.Att4 = Attention_block(F_g=features[2], F_l=features[2], F_int=features[1])
        self.Up_conv4 = conv_block(ch_in=features[3], ch_out=features[2])

        self.Up3 = up_conv(ch_in=features[2], ch_out=features[1])
        self.Att3 = Attention_block(F_g=features[1], F_l=features[1], F_int=features[0])
        self.Up_conv3 = conv_block(ch_in=features[2], ch_out=features[1])

        self.Up2 = up_conv(ch_in=features[1], ch_out=features[0])
        val = int(features[0]/2)
        self.Att2 = Attention_block(F_g=features[0], F_l=features[0], F_int=val)  #int(features[0]/2)
        self.Up_conv2 = conv_block(ch_in=features[1], ch_out=features[0])

        self.Conv_1x1 = nn.Conv3d(features[0], output_ch, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=0)

        # deep supervision
        self.dsup1_logits = nn.Sequential(nn.Upsample(size=[128, 128, 128], mode='trilinear'), nn.Conv3d(features[3], output_ch, kernel_size=(1, 1, 1))) #128 if
        self.dsup2_logits = nn.Sequential(nn.Upsample(size=[128, 128, 128],mode='trilinear'), nn.Conv3d(features[2], output_ch, kernel_size=(1, 1, 1))) #64
        self.dsup3_logits = nn.Sequential(nn.Upsample(size=[128, 128, 128],mode='trilinear'), nn.Conv3d(features[1], output_ch, kernel_size=(1, 1, 1))) #32


    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        x = self.Up5(x5)
        x4 = self.Att5(g=x, x=x4)
        x = torch.cat((x4, x), dim=1)
        x = self.Up_conv5(x)
        output_up1 = self.dsup1_logits(x)
        #print("Dimensions d5: ", d5.size())

        x= self.Up4(x)
        x3 = self.Att4(g=x, x=x3)
        x = torch.cat((x3, x), dim=1)
        x = self.Up_conv4(x)
        output_up2 = self.dsup2_logits(x)
        #print("Dimensions d4: ", d4.size())

        x = self.Up3(x)
        x2 = self.Att3(g=x, x=x2)
        x = torch.cat((x2, x), dim=1)
        x = self.Up_conv3(x)
        output_up3 = self.dsup3_logits(x)
        #print("Dimensions d3: ", d3.size())

        x = self.Up2(x)
        x1 = self.Att2(g=x, x=x1)
        x = torch.cat((x1, x), dim=1)
        x = self.Up_conv2(x)

        x = self.Conv_1x1(x)

        return x, output_up1, output_up2, output_up3

