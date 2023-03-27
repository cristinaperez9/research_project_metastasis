print("Script opened in cluster")
import nibabel
import numpy as np
import os

from skimage import measure
from statistics import mean
import warnings
from typing import Callable, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from monai.losses.focal_loss import FocalLoss
from monai.losses.spatial_mask import MaskedLoss
from monai.networks import one_hot
from monai.utils import DiceCEReduction, LossReduction, Weight, look_up_option, pytorch_after


from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    Spacingd,
    EnsureTyped,
    EnsureType,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,    
    NormalizeIntensityd,
    ScaleIntensityd,
    RandRotated,
    RandZoomd,
    RandGaussianNoised,
    RandGaussianSharpend,
    RandAdjustContrastd,
    Rand3DElasticd,
)
#from monai.handlers.utils import from_engine
from monai.networks.nets import UNETR, ViT, SegResNet, AttentionUnet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss, DiceCELoss, DiceFocalLoss, FocalLoss, GeneralizedDiceLoss
from monai.inferers import sliding_window_inference
from monai.data import (CacheDataset, 
                DataLoader, Dataset, decollate_batch, load_decathlon_datalist)
from monai.config import print_config
from monai.visualize import plot_2d_or_3d_image
print("Monai modules loaded")
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cudnn.benchmark = True
from datetime import datetime
import os
import logging
import sys
from init import Options
from utils import update_best_epochs
from networks import *
from network_deformable import DeformAttentionUNet, ThreeOffsetsAttentionUNet, DUNetV1V2
print("Modules loaded")
print_config()

#####################################################################################
def poly_lr(iter, num_iter, initial_lr, exponent=0.9):
    return initial_lr * (1 - iter / num_iter)**exponent
#####################################################################################

def main():
    opt = Options().parser()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    ########################################################################################
    # Load configuration settings from parser
    ########################################################################################

    root_dir = opt.model_folder
    # Create model folder if it does not exist
    if not os.path.isdir(root_dir):
        os.makedirs(root_dir)
    print("Model folder is: ", root_dir)

    datasets = opt.dataset_folder
    save_name = opt.network
    if opt.gpus != '-1':
        num_gpus = len(opt.gpus.split(','))
    else:
        num_gpus = 0
    print('Number of GPU :', num_gpus)
    print('Number of workers :', opt.workers)
    print('Batch size :', opt.batch_size)
    train_files = load_decathlon_datalist(datasets, True, "training")
    val_files = load_decathlon_datalist(datasets, True, "validation")


    print("Number of training samples : ", len(train_files),\
        "\nNumber of validation samples : ", len(val_files))

    ###################################################################################################
    # Set initial learning rate depending if poly learning rate scheduler is used or not
    if opt.update_lr:
        opt.lr = 0.01
        print("Poly learning rate scheduler employed. Initial learning rate: ", opt.lr)
    else:
        print("Fixed learning rate: ", opt.lr)

    ###################################################################################################
    # Data Loaders
    ###################################################################################################
    num_samples = opt.num_samples
    prob_met = opt.prob_met
    print("Probability a random patch is cropped having as center a metastasis pixel: ", prob_met)
    print("Number of patches extracted per patient: ", num_samples)
    print("The patch size for training the network is: ", opt.patch_size)

    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            CropForegroundd(keys=["image", "label"], source_key="image"), #commented for BrainMetShare
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",  # It is used to find the foreground
                spatial_size=opt.patch_size,
                pos=opt.prob_met,
                neg=1 - opt.prob_met,
                num_samples=num_samples,
                image_key="image",
                image_threshold=0,
            ),
            #Data augmentation
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0), #original
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1), #original
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2), #original
            Rand3DElasticd(keys=["image", "label"], prob=0.5, sigma_range=(5, 7), magnitude_range=(300, 300), mode=['bilinear', 'nearest']),
            RandScaleIntensityd(keys="image", factors=0.1, prob=1.0), #original #nonew
            RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0), #original #nonew
            EnsureTyped(keys=["image", "label"]), #original
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            CropForegroundd(keys=["image", "label"], source_key="image"), #commented for BrainMetShare
            EnsureTyped(keys=["image", "label"]),
        ]
    )

    #Setup Loaders
    ##Cache rate parameters are used to store data in the cache (depends on the memory)
    train_ds = CacheDataset(
        data=train_files, transform=train_transforms,
        cache_rate=0.0, num_workers=opt.workers)
    train_loader = DataLoader(train_ds, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)
    val_ds = CacheDataset(
        data=val_files, transform=val_transforms,
        cache_rate=0.0, num_workers=opt.workers)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=opt.workers)

    device = torch.device("cuda:0")  #Change if multiple gpu training

    ###################################################################################################
    # Model: use "Attention" or "UNet" as baselines
    ###################################################################################################

    if opt.network == "Attention":
        print("Features used in the model: ", opt.features)

        if not opt.deep_supervision:
            model = AttU_Net(img_ch=opt.in_channels, output_ch=opt.out_channels, features=opt.features)
        else:
            model = AttU_Net_ds(img_ch=opt.in_channels, output_ch=opt.out_channels, features=opt.features)

    if opt.network == "ThreeOffsetsAttentionUNet":
        print("Features used in the model: ", opt.features)
        model = ThreeOffsetsAttentionUNet(img_ch=opt.in_channels, output_ch=opt.out_channels, features=opt.features)

    # 3D extension of "DUNet: A deformable network for retinal vessel segmentation"
    if opt.network == "DUNetV1V2":
        model = DUNetV1V2(img_ch=opt.in_channels, output_ch=opt.out_channels)

    # Attention U-Net with deformable convolutions in some convolutional blocks
    if opt.network == "DeformAttention":
        print("Features used in the model: ", opt.features)
        model = DeformAttentionUNet(img_ch=opt.in_channels, output_ch=opt.out_channels, features=opt.features)

    if opt.network == "UNet":
        if not opt.deep_supervision:
            model = UNet(opt.in_channels, opt.out_channels)
        else:
            model = UNet_ds(opt.in_channels, opt.out_channels, features=opt.features)

    ####################################################################################################

    ###################################################################################################
    # GPU setting, and loading pretrained variables (if pretraining = True)
    ###################################################################################################

    if num_gpus > 0:
        if num_gpus > 1:
            model.cuda()
            model = torch.nn.DataParallel(model)
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Model {opt.network} set on multiple gpus.")
        else:
            model.to(device)
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Model {opt.network} set on GPU")
    else:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Model {opt.network} set on CPU")

    pth_model = opt.model_folder
    if not os.path.exists(pth_model):
        print("Creating model path...")
        os.makedirs(pth_model)


    if opt.pretrain is not None:

        model.load_state_dict(torch.load(opt.pretrain))
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Model pretrained loaded")
        print("Pretrained model loaded with name: ", opt.pretrain)

        ## Load last epoch, value of the loss at each epoch
        datafile_loss = os.path.join(pth_model, 'loss.npy')
        epoch_loss_values = list(np.load(datafile_loss))
        current_epoch = len(epoch_loss_values)
        ## Load the best DSC and epoch of the best DSC
        datafile_best_epoch = os.path.join(pth_model, 'best_epoch.npy')
        values_best_epoch = np.load(datafile_best_epoch)
        ## Load list of metrics (DSC)
        datafile_dice = os.path.join(pth_model, 'dice.npy')
        metric_values = list(np.load(datafile_dice))
        ## Load validation loss
        datafile_loss_val = os.path.join(pth_model, 'loss_val.npy')
        val_loss_values = list(np.load(datafile_loss_val))

    ###################################################################################################

    ###################################################################################################
    # Set loss, optimizer and metric
    ###################################################################################################

    print("The loss function used is: ", opt.loss)
    if opt.loss == 'DiceCELoss':
        loss_function = DiceCELoss(to_onehot_y=True, softmax=True, include_background=False) #softmax=true to_onehoy_y=true
        loss_function_val = DiceCELoss(to_onehot_y=True, softmax=True, include_background=False) #softmax=true to_onehoy_y=true
    elif opt.loss == 'DiceLoss':
        loss_function = DiceLoss(to_onehot_y=True, softmax=True, include_background=False)
    elif opt.loss == 'DiceFocalLoss':
        loss_function = DiceFocalLoss(to_onehot_y=True, softmax=True, include_background=False)
    elif opt.loss == 'GeneralizedDiceLoss':
        loss_function = GeneralizedDiceLoss(include_background=False, to_onehot_y=True, softmax=True)
    elif opt.loss == 'FocalLoss':
        loss_function = FocalLoss(include_background=False, to_onehot_y=True)


    optimizer = torch.optim.AdamW(model.parameters(), opt.lr, weight_decay=1e-5)
    if opt.update_lr:
        # Use the same optimizer and learning rate scheduler than nnUNet
        optimizer = torch.optim.SGD(model.parameters(), opt.lr, nesterov=True, momentum=0.99)
        num_iters = opt.epochs * (len(train_files) / opt.batch_size)
        num_steps_epoch = (len(train_files) / opt.batch_size)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    ###################################################################################################

    ###################################################################################################
    # Start (or continue) a typical pytorch training/validation workflow
    ###################################################################################################

    max_epochs = opt.epochs
    val_interval = 2
    print("Validation performed every ", val_interval, " epochs")
    if opt.pretrain is None:
        values_best_epoch = np.multiply(np.ones((2, 5)), -1)
        epoch_loss_values = []
        metric_values = []
        current_epoch = 0
        val_loss_values = []  # validation loss

    post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([EnsureType(), AsDiscrete(to_onehot=2)])

    ###################################### TRAINING ######################################################
    if opt.deep_supervision:
        print("Training with deep supervision")
        print("Training weights: ", opt.ds_loss_weights)


    for epoch in range(current_epoch, max_epochs):
        print("-" * 40)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] epoch {epoch +1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0

        for batch_data in train_loader:
            step += 1
            ### Obtain training image and corresponding label ###
            inputs, labels = (batch_data['image'], batch_data['label'])

            ### Send data to the GPU ###
            if num_gpus > 0:
                inputs = inputs.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()

            if opt.deep_supervision:
                outputs, output_up1, output_up2, output_up3 = model(inputs)
                dsw = opt.ds_loss_weights
                loss = dsw[0]*loss_function(outputs, labels) + dsw[3]*loss_function(output_up1, labels) + \
                       dsw[2]*loss_function(output_up2, labels) + dsw[1]*loss_function(output_up3, labels)

            else:
                outputs = model(inputs)
                loss = loss_function(outputs, labels)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // opt.batch_size

            # Update learning rate after each iteration (every time backpropagation occurs)
            if opt.update_lr:
                new_lr = poly_lr((epoch*num_steps_epoch)+step, num_iters, opt.lr)
                optimizer.param_groups[0]['lr'] = new_lr

            if (step -1)%10 == 0:
                print(
                    f"{step}/{epoch_len}, "
                    f"train_loss: {loss.item():.4f}")
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)


        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        loss_numpy = np.array(epoch_loss_values)

        datafile_loss = pth_model + "loss.npy"
        np.save(datafile_loss, loss_numpy)
        del inputs, outputs, labels


        ###################################### VALIDATION ######################################################

        if (epoch+1)% val_interval == 0:

            # Save model after training phase (every 2 epochs):
            # interesting when pretraining is required (network trained for several days)

            # SAVE CURRENT MODEL:
            torch.save(model.state_dict(), os.path.join(
                root_dir, save_name + str(epoch + 1) + ".pth"
            ))

            print("Saving model with name: ", os.path.join(
                root_dir, save_name + str(epoch + 1) + ".pth"))

            model.eval()
            val_loss = 0
            step_val = 0
            with torch.no_grad():
                
                for val_data in val_loader:
                    step_val = step_val + 1
                    val_inputs, val_labels = (val_data['image'], val_data['label'])

                    if opt.gpus != '-1':
                        val_inputs = val_inputs.cuda()
                        val_labels = val_labels.cuda()

                    roi_size = opt.patch_size
                    sw_batch_size = num_samples  #4 # set to 1 for patches 128 x 128 x 128
                    if opt.deep_supervision:
                        val_outputs, val_output_up1, val_output_up2, val_output_up3 = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model, progress=True, overlap=0.5, mode="gaussian")
                        dsw = opt.ds_loss_weights
                        val_loss0 = dsw[0] * loss_function(val_outputs, val_labels) + dsw[3] * loss_function(val_output_up1, val_labels) + \
                                    dsw[2] * loss_function(val_output_up2, val_labels) + dsw[1] * loss_function(val_output_up3, val_labels)
                    else:
                        val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model, progress=True, overlap=0.5, mode="gaussian")
                        val_loss0 = loss_function_val(val_outputs, val_labels)

                    val_loss += val_loss0.item()

                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                    # compute metric for current iteration
                    dice_metric(y_pred=val_outputs, y=val_labels)

                val_loss /= step_val
                val_loss_values.append(val_loss)
                loss_val_numpy = np.array(val_loss_values)
                datafile_loss_val = pth_model + "loss_val.npy"
                np.save(datafile_loss_val, loss_val_numpy)

                # Clear variables to not saturate the system
                del val_outputs, val_labels, val_inputs

                metric = dice_metric.aggregate().item()
                dice_metric.reset()

                metric_values.append(metric)

                ## SAVE LIST OF DICE VALUES FOR LATER PLOTTING ##

                datafile_metric = pth_model + "dice.npy"
                np.save(datafile_metric, metric_values)

                if metric > values_best_epoch[0, 4]:
                    best_metric_epoch = epoch + 1
                    values_best_epoch = update_best_epochs(values_best_epoch, best_metric_epoch, metric)
                    ## SAVE THE BEST DSC UNTIL NOW AND BEST EPOCH

                    datafile_best_epoch = pth_model + 'best_epoch.npy'
                    np.save(datafile_best_epoch, values_best_epoch)

                    # SAVE CURRENT MODEL:
                    torch.save(model.state_dict(), os.path.join(
                        root_dir, save_name + str(best_metric_epoch) + ".pth"
                    ))

                    print("Saving model with name: ", os.path.join(
                        root_dir, save_name + str(best_metric_epoch) + ".pth"))

                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] New best metric model was saved ")
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"
                    f" current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                    f"\nbest mean dice: {values_best_epoch[0,0]:.4f} "
                    f"at epoch: {values_best_epoch[1,0]}"
                )


if __name__ == "__main__":
    main()
