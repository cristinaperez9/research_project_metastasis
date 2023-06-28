
#######################################################################################################
# Cristina Almagro-Pérez, ETH Zürich, 2022

# Network 1: 2D Deformable U-Net
# Adapted from:
#Jin, Q., et al. (2019). DUNet: A deformable network for retinal vessel segmentation.
# The distribution of deformable and conventional convolutional layers were modified with
# respect to their implementation for the task of brain metastasis segmentation

# Network 2: 2D U-Net
#######################################################################################################

import torch
import torch.nn as nn
from modules_deformable_2D import *


class DUNetV1V2(nn.Module):
    def __init__(self, img_ch, output_ch, downsize_nb_filters_factor=4):
        super(DUNetV1V2, self).__init__()
        self.inc = deform_inconv(img_ch, 64 // downsize_nb_filters_factor)
        self.down1 = deform_down(64 // downsize_nb_filters_factor, 128 // downsize_nb_filters_factor)
        self.down2 = deform_down(128 // downsize_nb_filters_factor, 256 // downsize_nb_filters_factor)
        self.down3 = down(256 // downsize_nb_filters_factor, 512 // downsize_nb_filters_factor)
        self.down4 = down(512 // downsize_nb_filters_factor, 512 // downsize_nb_filters_factor)
        self.up1 = up(1024 // downsize_nb_filters_factor, 256 // downsize_nb_filters_factor)
        self.up2 = up(512 // downsize_nb_filters_factor, 128 // downsize_nb_filters_factor)
        self.up3 = deform_up(256 // downsize_nb_filters_factor, 64 // downsize_nb_filters_factor)
        self.up4 = deform_up(128 // downsize_nb_filters_factor, 64 // downsize_nb_filters_factor)
        self.outc = nn.Conv2d(64 // downsize_nb_filters_factor + 1, output_ch, 1)

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


class UNetV1V2(nn.Module):
    def __init__(self, img_ch, output_ch, downsize_nb_filters_factor=4):
        super(UNetV1V2, self).__init__()
        self.inc = inconv(img_ch, 64 // downsize_nb_filters_factor)
        self.down1 = down(64 // downsize_nb_filters_factor, 128 // downsize_nb_filters_factor)
        self.down2 = down(128 // downsize_nb_filters_factor, 256 // downsize_nb_filters_factor)
        self.down3 = down(256 // downsize_nb_filters_factor, 512 // downsize_nb_filters_factor)
        self.down4 = down(512 // downsize_nb_filters_factor, 512 // downsize_nb_filters_factor)
        self.up1 = up(1024 // downsize_nb_filters_factor, 256 // downsize_nb_filters_factor)
        self.up2 = up(512 // downsize_nb_filters_factor, 128 // downsize_nb_filters_factor)
        self.up3 = up(256 // downsize_nb_filters_factor, 64 // downsize_nb_filters_factor)
        self.up4 = up(128 // downsize_nb_filters_factor, 64 // downsize_nb_filters_factor)
        self.outc = nn.Conv2d(64 // downsize_nb_filters_factor + img_ch, output_ch, 1)

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
        x = torch.cat([inp, x], dim=1)
        x = self.outc(x)
        x = torch.sigmoid(x)
        return x