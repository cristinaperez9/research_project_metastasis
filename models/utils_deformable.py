
#####################################################################################################
# Cristina Almagro-Pérez, ETH Zürich, 2022

# Deformable 3D convolution employed in Deformable Attention U-Net #
## Offsets are learnt through an additional convolutional layer ##
#####################################################################################################

# Import necessary packages
import torch
from torch import nn
import sys
sys.path.insert(0, '/scratch_net/biwidl311/Cristina_Almagro/big/modulated-deform-conv')
#sys.path.insert(0, '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/metastases_project/big/modulated-deform-conv')
from modulated_deform_conv import *
import torch.nn.functional as F
import ipdb
import numpy as np
import time


class DeformConv3dPlusOffset(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3, padding=1, stride=1, bias=None):

        super(DeformConv3dPlusOffset, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.deform = DeformConv3d(in_channels=ch_in, out_channels=ch_out, kernel_size=3, padding=1, stride=1, bias=bias)

        self.p_conv = nn.Conv3d(ch_in, 3 * kernel_size * kernel_size * kernel_size, kernel_size=(3, 3, 3), padding=1, stride=(stride, stride, stride))
        nn.init.constant_(self.p_conv.weight, 0)  # Initialize all p_conv.weight with 0 value
        self.p_conv.register_full_backward_hook(self._set_lr)


    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        offset = self.p_conv(x)
        # if self.padding:
        #     pad_val = self.padding  #amount of padding in each direction
        #     x = F.pad(x, (pad_val, pad_val, pad_val, pad_val, pad_val, pad_val), mode='constant', value=0)
        out = self.deform.forward(x, offset)

        return out
