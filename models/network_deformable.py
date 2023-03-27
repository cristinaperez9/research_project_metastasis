
#######################################################################################################
# Cristina Almagro-Pérez, ETH Zürich, 2022

# Deformable Attention U-Net
#######################################################################################################
import torch
import torch.nn as nn
from modules_deformable import ConvBlock, UpConv, AttentionBlock, DeformConvBlock, UpDeformConv
from modules_deformable import ThreeOffsetsBlock, UpThreeOffsetsConv
from modules_deformable import ThreeOffsetsBlockShareWeights, UpThreeOffsetsConvShareWeights
from deform_part_3D import deform_up, deform_down, deform_inconv
from unet_parts_3D import *
####################################################################################################
# Network 1: Attention U-Net + some layers deformable convolutions with learnable offsets
####################################################################################################

class DeformAttentionUNet(nn.Module):
    def __init__(self, img_ch=1, output_ch=2, features=[16, 32, 64, 128, 256]):
        super(DeformAttentionUNet, self).__init__()

        self.Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.Conv1 = DeformConvBlock(ch_in=img_ch, ch_out=features[0])  #16
        self.Conv2 = DeformConvBlock(ch_in=features[0], ch_out=features[1])  #32
        self.Conv3 = DeformConvBlock(ch_in=features[1], ch_out=features[2])  #64
        self.Conv4 = ConvBlock(ch_in=features[2], ch_out=features[3])  #128
        self.Conv5 = ConvBlock(ch_in=features[3], ch_out=features[4])  #256

        self.Up5 = UpConv(ch_in=features[4], ch_out=features[3])
        self.Att5 = AttentionBlock(F_g=features[3], F_l=features[3], F_int=features[2])
        self.Up_conv5 = ConvBlock(ch_in=features[4], ch_out=features[3])

        self.Up4 = UpConv(ch_in=features[3], ch_out=features[2])
        self.Att4 = AttentionBlock(F_g=features[2], F_l=features[2], F_int=features[1])
        self.Up_conv4 = ConvBlock(ch_in=features[3], ch_out=features[2])

        # DEFORMABLE
        self.Up3 = UpDeformConv(ch_in=features[2], ch_out=features[1])
        self.Att3 = AttentionBlock(F_g=features[1], F_l=features[1], F_int=features[0])
        self.Up_conv3 = DeformConvBlock(ch_in=features[2], ch_out=features[1])

        # DEFORMABLE
        self.Up2 = UpDeformConv(ch_in=features[1], ch_out=features[0])
        val = int(features[0]/2)
        self.Att2 = AttentionBlock(F_g=features[0], F_l=features[0], F_int=val)  #int(features[0]/2)
        self.Up_conv2 = DeformConvBlock(ch_in=features[1], ch_out=features[0])

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
        del x4, x5

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)
        del x3, d5

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)
        del x2, d4

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        del x1, d3

        d1 = self.Conv_1x1(d2)

        return d1


####################################################################################################
# Network 2: Attention U-Net + some layers deformable convolutions with FIXED offsets:
# Combination of three offsets
####################################################################################################

class ThreeOffsetsAttentionUNet(nn.Module):
    def __init__(self, img_ch=1, output_ch=2, features=[16, 32, 64, 128, 256]):
        super(ThreeOffsetsAttentionUNet, self).__init__()

        self.Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Encoder
        self.Conv1 = ThreeOffsetsBlock(ch_in=img_ch, ch_out=features[0])  #16
        self.Conv2 = ThreeOffsetsBlock(ch_in=features[0], ch_out=features[1])  #32 # DEFORMABLE
        self.Conv3 = ThreeOffsetsBlock(ch_in=features[1], ch_out=features[2])  #64 # DEFORMABLE
        self.Conv4 = ConvBlock(ch_in=features[2], ch_out=features[3])  #128
        self.Conv5 = ConvBlock(ch_in=features[3], ch_out=features[4])  #256

        # Decoder
        self.Up5 = UpConv(ch_in=features[4], ch_out=features[3])
        self.Att5 = AttentionBlock(F_g=features[3], F_l=features[3], F_int=features[2])
        self.Up_conv5 = ConvBlock(ch_in=features[4], ch_out=features[3])

        self.Up4 = UpConv(ch_in=features[3], ch_out=features[2])
        self.Att4 = AttentionBlock(F_g=features[2], F_l=features[2], F_int=features[1])
        self.Up_conv4 = ConvBlock(ch_in=features[3], ch_out=features[2])

        # DEFORMABLE
        self.Up3 = UpThreeOffsetsConv(ch_in=features[2], ch_out=features[1])
        self.Att3 = AttentionBlock(F_g=features[1], F_l=features[1], F_int=features[0])
        self.Up_conv3 = ThreeOffsetsBlock(ch_in=features[2], ch_out=features[1])

        # DEFORMABLE
        self.Up2 = UpThreeOffsetsConv(ch_in=features[1], ch_out=features[0])
        val = int(features[0]/2)
        self.Att2 = AttentionBlock(F_g=features[0], F_l=features[0], F_int=val)  #int(features[0]/2)
        self.Up_conv2 = ThreeOffsetsBlock(ch_in=features[1], ch_out=features[0])

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
        del x4, x5

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)
        del x3, d5

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)
        del x2, d4

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        del x1, d3

        d1 = self.Conv_1x1(d2)

        return d1


#####################################################################################################
# Network 3: DUNetV1V2 3D extension
#####################################################################################################

class DUNetV1V2(nn.Module):
    # downsize_nb_filters_factor=4 compare to DUNetV1V2_MM
    def __init__(self, img_ch, output_ch, downsize_nb_filters_factor=4):
        super(DUNetV1V2, self).__init__()
        self.inc = deform_inconv(img_ch, 64 // downsize_nb_filters_factor)
        #self.inc = inconv(img_ch, 64 // downsize_nb_filters_factor)  #Second term: 16 in the paper
        self.down1 = deform_down(64 // downsize_nb_filters_factor, 128 // downsize_nb_filters_factor)  #(16, 32)
        self.down2 = deform_down(128 // downsize_nb_filters_factor, 256 // downsize_nb_filters_factor) #(32, 64)
        self.down3 = down(256 // downsize_nb_filters_factor, 512 // downsize_nb_filters_factor)
        self.down4 = down(512 // downsize_nb_filters_factor, 512 // downsize_nb_filters_factor)
        self.up1 = up(1024 // downsize_nb_filters_factor, 256 // downsize_nb_filters_factor)
        self.up2 = up(512 // downsize_nb_filters_factor, 128 // downsize_nb_filters_factor)
        self.up3 = deform_up(256 // downsize_nb_filters_factor, 64 // downsize_nb_filters_factor)
        self.up4 = deform_up(128 // downsize_nb_filters_factor, 64 // downsize_nb_filters_factor)
        self.outc = nn.Conv3d(64 // downsize_nb_filters_factor + 1, output_ch, (1, 1, 1))
        self.dropout = nn.Dropout(0.25)
    def forward(self, inp):
        x1 = self.inc(inp)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = torch.cat([inp, x], dim=1)  # It concatenates the input and the extracted features.
        x = self.outc(x)
        return torch.sigmoid(x)