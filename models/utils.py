#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from statistics import mean


def update_best_epochs(best_metrics,current_epoch, current_metric):
    """ Select the best 5 epochs in terms of DSC value in the validation set"""
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
