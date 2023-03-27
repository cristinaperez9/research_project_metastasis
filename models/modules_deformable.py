
########################################################################################################
# Cristina Almagro-Pérez, ETH Zürich, 2022

# Modules employed in the main architecture: Deformable Attention U-Net
# and in the ThreeOffsetMethod
########################################################################################################

# Import necessary packages
import torch
import torch.nn as nn
import numpy as np
import sys
#from DeformableBlock import DeformConv3d
from utils_deformable import DeformConv3dPlusOffset  # Pure Deformable Convolution
sys.path.insert(0, '/scratch_net/biwidl311/Cristina_Almagro/big/modulated-deform-conv')
from modulated_deform_conv import *

#########################################################################################################
#  PART 1 : Usual modules also employed in normal U-Net
#########################################################################################################


class ConvBlock(nn.Module):
    """ Double Convolution Module """
    def __init__(self, ch_in, ch_out):
        super(ConvBlock, self).__init__()
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


class UpConv(nn.Module):

    def __init__(self, ch_in, ch_out):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv3d(ch_in, ch_out, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1, bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
            )

    def forward(self, x):
        x = self.up(x)
        return x


#########################################################################################################
#  PART 2 : Attention Gate module
#########################################################################################################

class AttentionBlock(nn.Module):
    """ Attention Gate """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
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

#########################################################################################################
#  PART 3 : Deformable modules (with learnable offsets).
# Used in Deformable Attention U-Net and DUNetV1V2
#########################################################################################################


class DeformConvBlock(nn.Module):
    """ Double Deformable convolution module """
    """ (3D Deformable convolution => Batch normalization => ReLU) * 2 """

    def __init__(self, ch_in, ch_out):
        super(DeformConvBlock, self).__init__()
        self.conv = nn.Sequential(
            DeformConv3dPlusOffset(ch_in, ch_out, kernel_size=3, padding=1, stride=1, bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
            DeformConv3dPlusOffset(ch_out, ch_out, kernel_size=3, padding=1, stride=1, bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UpDeformConv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(UpDeformConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            DeformConv3dPlusOffset(ch_in, ch_out, kernel_size=3, padding=1, stride=1, bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

#########################################################################################################
#  PART 4 : Three Offsets modules (combination of three fixed offsets through 1x1 convolutions)
#########################################################################################################


####################### Three Offset Module Sharing Weights ###################
class ThreeOffsetsBlockShareWeights(nn.Module):
    """ Combine three offsets """
    """ (3D Deformable convolution => Batch normalization => ReLU) * 2 """
    def __init__(self, ch_in, ch_out):
        super(ThreeOffsetsBlockShareWeights, self).__init__()

        self.deform_conv1 = DeformConv3d(in_channels=ch_in, out_channels=ch_out, kernel_size=3, padding=1, stride=1, bias=True)

        self.after_conv1 = nn.Sequential(
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
        )

        self.deform_conv2 = DeformConv3d(in_channels=ch_out, out_channels=ch_out, kernel_size=3, padding=1, stride=1,
                                         bias=True)

        self.after_conv2 = nn.Sequential(
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
        )


        # 1x1 convolution to combine kernel outputs
        self.combine_kernels_1 = nn.Sequential(
            nn.Conv3d(in_channels=3*ch_out, out_channels=ch_out, kernel_size=(1, 1, 1), padding=0, stride=(1, 1, 1), bias=True),
            nn.BatchNorm3d(ch_out))
        self.combine_kernels_2 = nn.Sequential(
            nn.Conv3d(in_channels=3*ch_out, out_channels=ch_out, kernel_size=(1, 1, 1), padding=0, stride=(1, 1, 1), bias=True),
            nn.BatchNorm3d(ch_out))
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):

        batch_size, h, w, d = x.size(0), x.size(2), x.size(3), x.size(4)
        # offset0 is a list with length 81 (for each element in a kernel 27 offsets)
        # (3*3*3 = 27 kernel elements) * (3 offsets per element: y_offset, x_offset, z_offset)  = 81

        # Each row in the offset0 variable is an element of the 3x3x3 kernel
        # See the enclosed slides for the order of the elements in the kernel
        offset0 = [1, 1, 1,  #y_offset, x_offset, z_offset
                  1, 1, 0,
                  1, 1, -1,
                  1, 0, 1,
                  1, 0, 0,
                  1, 0, -1,
                  1, -1, 1,
                  1, -1, 0,
                  1, -1, -1,

                  0, 1, 1,
                  0, 1, 0,
                  0, 1, -1,
                  0, 0, 1,
                  0, 0, 0,
                  0, 0, -1,
                  0, -1, 1,
                  0, -1, 0,
                  0, -1, -1,

                  -1, 1, 1,
                  -1, 1, 0,
                  -1, 1, -1,
                  -1, 0, 1,
                  -1, 0, 0,
                  -1, 0, -1,
                  -1, -1, 1,
                  -1, -1, 0,
                  -1, -1, -1,
                  ]

        for i in range(0, 2):
            ###########################################################################################
            # First convolutional kernel: normal convolution --> no offset, alpha=0
            ###########################################################################################

            alpha = 0
            offset = [alpha * x for x in offset0]
            offset = torch.tensor(offset).cuda().float()
            offset = offset.repeat_interleave(h * w * d)
            offset = offset.reshape((1, 81, h, w, d))
            offset = torch.cat(batch_size * [offset])

            if i == 0:
                out1 = self.deform_conv1(x, offset)
                out1 = self.after_conv1(out1)
            else:
                out1 = self.deform_conv2(x, offset)
                out1 = self.after_conv2(out1)


            ###########################################################################################
            # Second convolutional kernel
            ###########################################################################################

            alpha = 0.4
            offset = [alpha * x for x in offset0]
            offset = torch.tensor(offset).cuda().float()
            offset = offset.repeat_interleave(h * w * d)
            offset = offset.reshape((1, 81, h, w, d))
            offset = torch.cat(batch_size * [offset])
            if i == 0:
                out2 = self.deform_conv1(x, offset)
                out2 = self.after_conv1(out2)
            else:
                out2 = self.deform_conv2(x, offset)
                out2 = self.after_conv2(out2)


            ###########################################################################################
            # Third convolutional kernel
            ###########################################################################################

            alpha = 0.7
            offset = [alpha * x for x in offset0]
            offset = torch.tensor(offset).cuda().float()
            offset = offset.repeat_interleave(h * w * d)
            offset = offset.reshape((1, 81, h, w, d))
            offset = torch.cat(batch_size * [offset])
            if i == 0:
                out3 = self.deform_conv1(x, offset)
                out3 = self.after_conv1(out3)
            else:
                out3 = self.deform_conv2(x, offset)
                out3 = self.after_conv2(out3)

            x = torch.cat((out1, out2, out3), 1)  # filter concatenation as Inception
            if i == 0:
                x = self.combine_kernels_1(x)  # 1x1 convolution plus batch norm
            else:
                x = self.combine_kernels_2(x)  # 1x1 convolution plus batch norm
            x = self.relu(x)  # relu after 1x1 convolution as Inception module

        return x


class UpThreeOffsetsConvShareWeights(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(UpThreeOffsetsConvShareWeights, self).__init__()

        self.before_conv = nn.Upsample(scale_factor=2)
        self.deform_conv1 = DeformConv3d(in_channels=ch_in, out_channels=ch_out, kernel_size=3, padding=1, stride=1, bias=True)
        self.after_conv1 = nn.Sequential(
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
        )

        # New definitions following Inception paper
        # 1x1 convolution to combine kernel outputs
        self.combine_kernels_1 = nn.Sequential(
            nn.Conv3d(in_channels=3 * ch_out, out_channels=ch_out, kernel_size=(1, 1, 1), padding=0, stride=(1, 1, 1),
                      bias=True),
            nn.BatchNorm3d(ch_out))
        self.relu = nn.ReLU(inplace=True)



    def forward(self, x):

        x = self.before_conv(x)
        batch_size, h, w, d = x.size(0), x.size(2), x.size(3), x.size(4)
        offset0 = [1, 1, 1,
                   1, 1, 0,
                   1, 1, -1,
                   1, 0, 1,
                   1, 0, 0,
                   1, 0, -1,
                   1, -1, 1,
                   1, -1, 0,
                   1, -1, -1,

                   0, 1, 1,
                   0, 1, 0,
                   0, 1, -1,
                   0, 0, 1,
                   0, 0, 0,
                   0, 0, -1,
                   0, -1, 1,
                   0, -1, 0,
                   0, -1, -1,

                   -1, 1, 1,
                   -1, 1, 0,
                   -1, 1, -1,
                   -1, 0, 1,
                   -1, 0, 0,
                   -1, 0, -1,
                   -1, -1, 1,
                   -1, -1, 0,
                   -1, -1, -1,
                   ]

        ###########################################################################################
        # First convolutional kernel: normal convolution --> no offset
        ###########################################################################################
        alpha = 0
        offset = [alpha * x for x in offset0]
        offset = torch.tensor(offset).cuda().float()
        offset = offset.repeat_interleave(h * w * d)
        offset = offset.reshape((1, 81, h, w, d))
        offset = torch.cat(batch_size * [offset])

        out1 = self.deform_conv1(x, offset)
        out1 = self.after_conv1(out1)

        ###########################################################################################
        # Second convolutional kernel
        ###########################################################################################

        alpha = 0.4
        offset = [alpha * x for x in offset0]
        offset = torch.tensor(offset).cuda().float()
        offset = offset.repeat_interleave(h * w * d)
        offset = offset.reshape((1, 81, h, w, d))
        offset = torch.cat(batch_size * [offset])

        out2 = self.deform_conv1(x, offset)
        out2 = self.after_conv1(out2)

        ###########################################################################################
        # Third convolutional kernel
        ###########################################################################################

        alpha = 0.7
        offset = [alpha * x for x in offset0]
        offset = torch.tensor(offset).cuda().float()
        offset = offset.repeat_interleave(h * w * d)
        offset = offset.reshape((1, 81, h, w, d))
        offset = torch.cat(batch_size * [offset])

        out3 = self.deform_conv1(x, offset)
        out3 = self.after_conv1(out3)

        x = torch.cat((out1, out2, out3), 1)  # filter concatenation as Inception
        x = self.combine_kernels_1(x)  # 1x1 convolution plus batch norm
        x = self.relu(x)  # relu after 1x1 convolution as Inception module

        return x


class ThreeOffsetsBlock(nn.Module):
    """ Combine three offsets """
    """ (3D Deformable convolution => Batch normalization => ReLU) * 2 """
    def __init__(self, ch_in, ch_out):
        super(ThreeOffsetsBlock, self).__init__()
        self.deform1 = DeformConv3d(in_channels=ch_in, out_channels=ch_out, kernel_size=3, padding=1, stride=1, bias=True)
        self.deform2 = DeformConv3d(in_channels=ch_out, out_channels=ch_out, kernel_size=3, padding=1, stride=1, bias=True)
        self.conv1 = nn.Conv3d(in_channels=ch_in, out_channels=ch_out, kernel_size=(3, 3, 3), padding=1, stride=(1, 1, 1), bias=True)
        self.conv2 = nn.Conv3d(in_channels=ch_out, out_channels=ch_out, kernel_size=(3, 3, 3), padding=1, stride=(1, 1, 1), bias=True)
        self.after_conv1 = nn.Sequential(
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
        )

        self.after_conv2 = nn.Sequential(
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
        )

        # 1x1 convolution to combine kernel outputs
        self.combine_kernels_1 = nn.Sequential(
            nn.Conv3d(in_channels=3*ch_out, out_channels=ch_out, kernel_size=(1, 1, 1), padding=0, stride=(1, 1, 1), bias=True),
            nn.BatchNorm3d(ch_out))
        self.combine_kernels_2 = nn.Sequential(
            nn.Conv3d(in_channels=3*ch_out, out_channels=ch_out, kernel_size=(1, 1, 1), padding=0, stride=(1, 1, 1), bias=True),
            nn.BatchNorm3d(ch_out))
        self.relu = nn.ReLU(inplace=True)

        # New definitions following Inception paper

        ########################################### no offset ###########################################
        self.conv_block_1 = nn.Sequential(
            nn.Conv3d(in_channels=ch_in, out_channels=ch_out, kernel_size=(3, 3, 3), padding=1, stride=(1, 1, 1), bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv3d(in_channels=ch_out, out_channels=ch_out, kernel_size=(3, 3, 3), padding=1, stride=(1, 1, 1), bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
        )
        ########################################### offset 1 ###########################################

        self.after_conv_1_offset_1 = nn.Sequential(
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
        )

        self.after_conv_2_offset_1 = nn.Sequential(
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
        )

        self.deform_1_offset_1 = DeformConv3d(in_channels=ch_in, out_channels=ch_out, kernel_size=3, padding=1, stride=1, bias=True)
        self.deform_2_offset_1 = DeformConv3d(in_channels=ch_out, out_channels=ch_out, kernel_size=3, padding=1, stride=1, bias=True)
        ########################################### offset 2 ############################################

        self.after_conv_1_offset_2 = nn.Sequential(
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
        )

        self.after_conv_2_offset_2 = nn.Sequential(
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
        )

        self.deform_1_offset_2 = DeformConv3d(in_channels=ch_in, out_channels=ch_out, kernel_size=3, padding=1, stride=1, bias=True)
        self.deform_2_offset_2 = DeformConv3d(in_channels=ch_out, out_channels=ch_out, kernel_size=3, padding=1, stride=1, bias=True)

    def forward(self, x):

        batch_size, h, w, d = x.size(0), x.size(2), x.size(3), x.size(4)

        for i in range(0, 2):
            ###########################################################################################
            # First convolutional kernel: normal convolution --> no offset
            ###########################################################################################

            if i == 0:
                out1 = self.conv_block_1(x)  # conv_block_1 # change later for conv1
            else:
                out1 = self.conv_block_2(x)  # conv_block_2 # change later for conv2


            ###########################################################################################
            # Second convolutional kernel
            ###########################################################################################

            alpha = 0.4
            offset = [alpha, alpha, alpha,
                      alpha, alpha, 0,
                      alpha, alpha, -alpha,
                      alpha, 0, alpha,
                      alpha, 0, 0,
                      alpha, 0, -alpha,
                      alpha, -alpha, alpha,
                      alpha, -alpha, 0,
                      alpha, -alpha, -alpha,

                      0, alpha, alpha,
                      0, alpha, 0,
                      0, alpha, -alpha,
                      0, 0, alpha,
                      0, 0, 0,
                      0, 0, -alpha,
                      0, -alpha, alpha,
                      0, -alpha, 0,
                      0, -alpha, -alpha,

                      -alpha, alpha, alpha,
                      -alpha, alpha, 0,
                      -alpha, alpha, -alpha,
                      -alpha, 0, alpha,
                      -alpha, 0, 0,
                      -alpha, 0, -alpha,
                      -alpha, -alpha, alpha,
                      -alpha, -alpha, 0,
                      -alpha, -alpha, -alpha,
                      ]
            offset = torch.tensor(offset).cuda()
            offset = offset.repeat_interleave(h * w * d)
            offset = offset.reshape((1, 81, h, w, d))
            offset = torch.cat(batch_size * [offset])
            if i == 0:
                out2 = self.deform_1_offset_1.forward(x, offset)
                out2 = self.after_conv_1_offset_1(out2)
            else:
                out2 = self.deform_2_offset_1.forward(x, offset)
                out2 = self.after_conv_2_offset_1(out2)


            ###########################################################################################
            # Third convolutional kernel
            ###########################################################################################

            alpha = 0.7
            offset = [alpha, alpha, alpha,
                      alpha, alpha, 0,
                      alpha, alpha, -alpha,
                      alpha, 0, alpha,
                      alpha, 0, 0,
                      alpha, 0, -alpha,
                      alpha, -alpha, alpha,
                      alpha, -alpha, 0,
                      alpha, -alpha, -alpha,

                      0, alpha, alpha,
                      0, alpha, 0,
                      0, alpha, -alpha,
                      0, 0, alpha,
                      0, 0, 0,
                      0, 0, -alpha,
                      0, -alpha, alpha,
                      0, -alpha, 0,
                      0, -alpha, -alpha,

                      -alpha, alpha, alpha,
                      -alpha, alpha, 0,
                      -alpha, alpha, -alpha,
                      -alpha, 0, alpha,
                      -alpha, 0, 0,
                      -alpha, 0, -alpha,
                      -alpha, -alpha, alpha,
                      -alpha, -alpha, 0,
                      -alpha, -alpha, -alpha,
                      ]
            offset = torch.tensor(offset).cuda()
            offset = offset.repeat_interleave(h * w * d)
            offset = offset.reshape((1, 81, h, w, d))
            offset = torch.cat(batch_size * [offset])
            if i == 0:
                out3 = self.deform_1_offset_2.forward(x, offset)
                out3 = self.after_conv_1_offset_2(out3)
            else:
                out3 = self.deform_2_offset_2.forward(x, offset)
                out3 = self.after_conv_2_offset_2(out3)

            x = torch.cat((out1, out2, out3), 1)  # filter concatenation as Inception
            if i == 0:
                x = self.combine_kernels_1(x)  # 1x1 convolution plus batch norm
            else:
                x = self.combine_kernels_2(x)  # 1x1 convolution plus batch norm
            x = self.relu(x)  # relu after 1x1 convolution as Inception module

        return x


class UpThreeOffsetsConv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(UpThreeOffsetsConv, self).__init__()

        self.deform1 = DeformConv3d(in_channels=ch_in, out_channels=ch_out, kernel_size=3, padding=1, stride=1, bias=True)
        self.after_conv = nn.Sequential(
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
        )
        self.before_conv = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Conv3d(in_channels=ch_in, out_channels=ch_out, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1))


        # New definitions following Inception paper
        # 1x1 convolution to combine kernel outputs
        self.combine_kernels_1 = nn.Sequential(
            nn.Conv3d(in_channels=3 * ch_out, out_channels=ch_out, kernel_size=(1, 1, 1), padding=0, stride=(1, 1, 1),
                      bias=True),
            nn.BatchNorm3d(ch_out))
        self.relu = nn.ReLU(inplace=True)

        ########################################### no offset ###########################################
        self.conv_block_1 = nn.Sequential(
            nn.Conv3d(in_channels=ch_in, out_channels=ch_out, kernel_size=(3, 3, 3), padding=1, stride=(1, 1, 1),
                      bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
        )

        ########################################### offset 1 ###########################################

        self.after_conv_1_offset_1 = nn.Sequential(
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
        )

        self.deform_1_offset_1 = DeformConv3d(in_channels=ch_in, out_channels=ch_out, kernel_size=3, padding=1,stride=1, bias=True)

        ########################################### offset 2 ############################################

        self.after_conv_1_offset_2 = nn.Sequential(
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
        )

        self.deform_1_offset_2 = DeformConv3d(in_channels=ch_in, out_channels=ch_out, kernel_size=3, padding=1, stride=1, bias=True)


    def forward(self, x):

        x = self.before_conv(x)
        batch_size, h, w, d = x.size(0), x.size(2), x.size(3), x.size(4)

        ###########################################################################################
        # First convolutional kernel: normal convolution --> no offset
        ###########################################################################################

        out1 = self.conv_block_1(x)
        #out1 = self.conv1(x)

        ###########################################################################################
        # Second convolutional kernel
        ###########################################################################################

        alpha = 0.4
        offset = [alpha, alpha, alpha,
                  alpha, alpha, 0,
                  alpha, alpha, -alpha,
                  alpha, 0, alpha,
                  alpha, 0, 0,
                  alpha, 0, -alpha,
                  alpha, -alpha, alpha,
                  alpha, -alpha, 0,
                  alpha, -alpha, -alpha,

                  0, alpha, alpha,
                  0, alpha, 0,
                  0, alpha, -alpha,
                  0, 0, alpha,
                  0, 0, 0,
                  0, 0, -alpha,
                  0, -alpha, alpha,
                  0, -alpha, 0,
                  0, -alpha, -alpha,

                  -alpha, alpha, alpha,
                  -alpha, alpha, 0,
                  -alpha, alpha, -alpha,
                  -alpha, 0, alpha,
                  -alpha, 0, 0,
                  -alpha, 0, -alpha,
                  -alpha, -alpha, alpha,
                  -alpha, -alpha, 0,
                  -alpha, -alpha, -alpha,
                  ]
        offset = torch.tensor(offset).cuda()
        offset = offset.repeat_interleave(h * w * d)
        offset = offset.reshape((1, 81, h, w, d))
        offset = torch.cat(batch_size * [offset])
        out2 = self.deform_1_offset_1.forward(x, offset)
        out2 = self.after_conv_1_offset_1(out2)


        ###########################################################################################
        # Third convolutional kernel
        ###########################################################################################

        alpha = 0.7
        offset = [alpha, alpha, alpha,
                  alpha, alpha, 0,
                  alpha, alpha, -alpha,
                  alpha, 0, alpha,
                  alpha, 0, 0,
                  alpha, 0, -alpha,
                  alpha, -alpha, alpha,
                  alpha, -alpha, 0,
                  alpha, -alpha, -alpha,

                  0, alpha, alpha,
                  0, alpha, 0,
                  0, alpha, -alpha,
                  0, 0, alpha,
                  0, 0, 0,
                  0, 0, -alpha,
                  0, -alpha, alpha,
                  0, -alpha, 0,
                  0, -alpha, -alpha,

                  -alpha, alpha, alpha,
                  -alpha, alpha, 0,
                  -alpha, alpha, -alpha,
                  -alpha, 0, alpha,
                  -alpha, 0, 0,
                  -alpha, 0, -alpha,
                  -alpha, -alpha, alpha,
                  -alpha, -alpha, 0,
                  -alpha, -alpha, -alpha,
                  ]
        offset = torch.tensor(offset).cuda()
        offset = offset.repeat_interleave(h * w * d)
        offset = offset.reshape((1, 81, h, w, d))
        offset = torch.cat(batch_size * [offset])
        out3 = self.deform_1_offset_2(x, offset)
        out3 = self.after_conv_1_offset_2(out3)

        x = torch.cat((out1, out2, out3), 1)  # filter concatenation as Inception
        x = self.combine_kernels_1(x)  # 1x1 convolution plus batch norm
        x = self.relu(x)  # relu after 1x1 convolution as Inception module

        return x
