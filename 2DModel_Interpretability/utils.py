
import os
import numpy as np


def threshold_mask(mask, thr = 0):
    # Scale mask from 0 to 255 to 0 - 1
    new_mask = mask / max(mask[np.nonzero(mask)])

    # Threshold to have a 0/1 labelization
    new_mask[new_mask > thr] = 1
    new_mask[new_mask <= thr] = 0
    return new_mask

def update_best_epochs(best_metrics,current_epoch, current_metric):
    """ Select the best 5 epochs in terms of DSC value"""
    expanded_list = list(best_metrics[0, :])  # first row
    expanded_list.append(current_metric)
    # Sort from high to low
    expanded_list.sort(reverse=True)
    best_metrics_updated = expanded_list[0:5]

    # Apply same sort operation to the epochs
    best_epochs = list(best_metrics[1, :])
    best_epochs.append(current_epoch)
    best_dice = list(best_metrics[0, :])
    best_dice.append(current_metric)
    best_epochs_updated = [best_epochs for _, best_epochs in sorted(zip(best_dice, best_epochs), reverse=True)]
    best_epochs_updated = best_epochs_updated[0:5]
    output_metrics = np.zeros((2, 5))
    output_metrics[0, :] = best_metrics_updated  # Best 5 DSC
    output_metrics[1, :] = best_epochs_updated  # Best 5 epochs number (according to DSC)

    return output_metrics

#######################################################################################################
# Local weighted loss: 3D extension to the loss described in [1]
# [1] Deep slice-crossed network with local weighted loss for brain metastases segmentation. Shu et al.
# Code of the above reference not available. Cristina Almagro-Pérez implementation.
#######################################################################################################

import warnings
from typing import Callable, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from monai.losses.focal_loss import FocalLoss
from monai.losses.spatial_mask import MaskedLoss
from monai.networks import one_hot
from monai.utils import DiceCEReduction, LossReduction, Weight, look_up_option, pytorch_after


class LocalWeightedLoss(_Loss):
    """
    Loss function described in Deep slice-crossed network with local weighted loss
    for brain metastases segmentation. Paper by Shu et al. Implementation here by Cristina Almagro-Pérez.

    """

    def __init__(
            self,
            include_background: bool = True,
            to_onehot_y: bool = False,
            sigmoid: bool = False,
            softmax: bool = False,
            reduction: Union[LossReduction, str] = LossReduction.MEAN,
            smooth_nr: float = 1e-5,
            smooth_dr: float = 1e-5,
            batch: bool = False,
    ) -> None:
        """
        For documentation of the above arguments check DSC MONAI implementation:
        https://docs.monai.io/en/stable/_modules/monai/losses/dice.html#DiceLoss

        """
        super().__init__(reduction=LossReduction(reduction).value)
        if int(sigmoid) + int(softmax) > 1:
            raise ValueError("Incompatible values: more than 1 of [sigmoid=True, softmax=True, other_act is not None].")
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)
        self.batch = batch

    def forward(self, input: torch.Tensor, target: torch.Tensor, weight_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD], where N is the number of classes.
            target: the shape should be BNH[WD] or B1H[WD], where N is the number of classes.
            weight_matrix: the shape should be BNH[WD] or B1H[WD], where N is the number of classes.
        """
        if self.sigmoid:
            input = torch.sigmoid(input)
        n_pred_ch = input.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(input, 1)
        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("Single channel prediction, `to_onehot_y=True` ignored.")
            else:
                target = one_hot(target, num_classes=n_pred_ch)

        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                # if skipping background, removing first channel
                # print(target.shape)
                # print(input.shape)
                # print(weight_matrix.shape)

                target = target[:, 1:]
                input = input[:, 1:]
                #weight_matrix = weight_matrix[:, 1:]
        if target.shape != input.shape:
            raise AssertionError(f"Ground truth has different shape ({target.shape}) from input ({input.shape})")

        if target.shape != weight_matrix.shape:
            raise AssertionError(f"Ground truth has different shape ({target.shape}) from weight matrix ({weight_matrix.shape})")

        # reducing only spatial dimensions (not batch nor channels)
        reduce_axis: List[int] = torch.arange(2, len(input.shape)).tolist()
        if self.batch:
            # reducing spatial dimensions and batch
            reduce_axis = [0] + reduce_axis

        # print("The axis to reduce are . . .")
        # print(reduce_axis)

        # Numerator
        #intersection = torch.sum(target * input, dim=reduce_axis)  #Normal DSC
        intersection = torch.sum(target * input * weight_matrix, dim=reduce_axis)
        numer = torch.mul(intersection, 2) + self.smooth_nr
        # print("The type of numer is . . . ", type(numer))
        # print("The dimensions of numer is . . .", numer.shape)

        # Denominator
        tensor_ones = torch.ones(input.size()).cuda()
        #part1 = torch.tensor((torch.mul(input * target, 2) + torch.sub(tensor_ones, input) * target) * weight_matrix)
        #part2 = torch.tensor(input * torch.sub(tensor_ones, target))

        part1 = (torch.mul(input * target, 2) + torch.sub(tensor_ones, input) * target) * weight_matrix
        part2 = input * torch.sub(tensor_ones, target)
        part1 = torch.sum(part1, dim=reduce_axis)
        part2 = torch.sum(part2, dim=reduce_axis)

        all_parts = part1 + part2
        # print("The type of all_parts is . . . ", type(all_parts))
        # print("The dimensions of numer is . . .", all_parts.shape)
        final_reduce_dim = 0 if self.batch else 1
        denom = all_parts.sum(final_reduce_dim, keepdim=True) + self.smooth_dr
        # print("The type of denom is . . . ", type(denom))
        # print("The dimensions of denom is . . .", denom.shape)

        f: torch.Tensor = 1.0 - (numer / denom)

        if self.reduction == LossReduction.MEAN.value:
            f = torch.mean(f)  # the batch and channel average
        elif self.reduction == LossReduction.SUM.value:
            f = torch.sum(f)  # sum over the batch and channel dims
        elif self.reduction == LossReduction.NONE.value:
            # If we are not computing voxelwise loss components at least
            # make sure a none reduction maintains a broadcastable shape
            broadcast_shape = list(f.shape[0:2]) + [1] * (len(input.shape) - 2)
            f = f.view(broadcast_shape)
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

        return f


 # Denominator
        # ground_o = torch.sum(target, dim=reduce_axis)
        # pred_o = torch.sum(input, dim=reduce_axis)
        #
        # denominator = ground_o + pred_o
        # print("The type of denominator is . . . ", type(denominator))
        # print("The dimensions of denominator is . . .", denominator.shape)

        #f: torch.Tensor = 1.0 - (2.0 * intersection + self.smooth_nr) / (denominator + self.smooth_dr)
################################################################################################################
# Compute weighted_mask for a given patient.
# 3D extension of the weighting described in [1] Deep slice-crossed network with local weighted loss for brain metastases segmentation. Shu et al.
# # Code of the above reference not available. Cristina Almagro-Pérez implementation.
#################################################################################################################


def obtain_weighted_mask(binary_mask, rad):
    """
    Weighted mask described in 'Deep slice-crossed network with local weighted loss
    for brain metastases segmentation'. Paper by Shu et al. Implementation here by Cristina Almagro-Pérez.
    """
    import numpy as np
    #print(binary_mask.shape)
    # Initialize weighted mask
    weighted_mask = np.zeros(binary_mask.shape)

    if len(binary_mask.shape) == 2:
        [row, col] = np.nonzero(binary_mask)
    elif len(binary_mask.shape) == 3:
        [row, col, dep] = np.nonzero(binary_mask)
    else:
        raise Warning("Dimension of the GT mask different than 2D or 3D")
    num_voxels = len(row)
    # Loop through all voxels containing a metastasis in the GT
    for kk in range(0, num_voxels):

        # Limits first summation
        low_lim_w = int(row[kk] - (rad-1)/2)
        high_lim_w = int(row[kk] + (rad-1)/2)

        # Limits second summation
        low_lim_h = int(col[kk] - (rad-1)/2)
        high_lim_h = int(col[kk] + (rad - 1) / 2)

        # Limits third summation
        if len(binary_mask.shape) == 3:
            low_lim_d = int(dep[kk] - (rad - 1) / 2)
            high_lim_d = int(dep[kk] + (rad - 1) / 2)

        t_value = 0
        for row_pos in range(low_lim_w, high_lim_w + 1):
            for col_pos in range(low_lim_h, high_lim_h + 1):
                if len(binary_mask.shape) == 2:
                    t_value = t_value + binary_mask[row_pos, col_pos]
                elif len(binary_mask.shape) == 3:
                    for dep_pos in range(low_lim_d, high_lim_d + 1):
                        t_value = t_value + binary_mask[row_pos, col_pos, dep_pos]

        if len(binary_mask.shape) == 2:
            weighted_mask[row[kk], col[kk]] = 1 + binary_mask[row[kk], col[kk]] - (1/rad**2) * t_value
        elif len(binary_mask.shape) == 3:
            weighted_mask[row[kk], col[kk], dep[kk]] = 1 + binary_mask[row[kk], col[kk], dep[kk]] - (1/rad**3) * t_value
    return weighted_mask





