print("Script opened in cluster")
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

)
#from monai.handlers.utils import from_engine
from monai.metrics import DiceMetric
from monai.losses import DiceLoss, DiceCELoss, DiceFocalLoss, FocalLoss, GeneralizedDiceLoss
from monai.inferers import sliding_window_inference, SimpleInferer, SliceInferer
from monai.data import (CacheDataset, 
                DataLoader, Dataset, decollate_batch, load_decathlon_datalist)
from monai.config import print_config
#from skimage import exposure
import cv2

print("Monai modules loaded")
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cudnn.benchmark = True
import numpy as np
from datetime import datetime
import os
import logging
import random
import sys
from init_2D import Options
from utils import update_best_epochs
from utils import LocalWeightedLoss
from network_deformable_2D import UNetV1V2, DUNetV1V2
import torchvision
from torch.utils.tensorboard import SummaryWriter
print("Modules loaded")
print_config()


def main():

    #######################################################################################
    # Reproducibility
    #######################################################################################
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    #######################################################################################

    opt = Options().parser()
    print(opt.model_folder)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    root_dir = opt.model_folder
    # Create model folder if it does not exist
    if not os.path.isdir(root_dir):
        os.makedirs(root_dir)
    print("Model folder is: ", root_dir)

    datasets = opt.dataset_folder
    print(datasets)
    save_name = opt.network
    if opt.gpus != '-1':
        num_gpus = len(opt.gpus.split(','))
    else:
        num_gpus = 0
    print('Number of GPU :', num_gpus)
    print('Number of workers :', opt.workers)
    print('Batch size :', opt.batch_size)
    val_type = opt.validation_type
    if val_type == '2D':
        val_files = load_decathlon_datalist(datasets, True, "validation")
    elif val_type == '3D':
        if not opt.dataset == 'BrainMetShare':
            val_files = load_decathlon_datalist(datasets, True, "validation_3D")
        else:
            val_files = load_decathlon_datalist(datasets, True, "validation")

    train_files = load_decathlon_datalist(datasets, True, "training")


    print("Number of training samples : ", len(train_files), \
          "\nNumber of validation samples : ", len(val_files))

    ###################################################################################################
    # Data Loaders
    ###################################################################################################

    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),

            #Data augmentation transformation
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            #RandRotated(keys=["image", "label"], prob=0.2, range_x=(-30, 30)),
            #RandZoomd(keys=["image", "label"], prob=0.2, min_zoom=0.7, max_zoom=1.4),
            #RandGaussianNoised(keys=["image"], prob=0.15),
            #RandGaussianSharpend(keys=["image"], prob=0.1, alpha=[0, 0.00001], sigma1_x=(0.5, 1.5), sigma1_y=(0.5, 1.5)),
            #RandScaleIntensityd(keys="image", factors=0.3, prob=0.15),
            #RandAdjustContrastd(keys="image", prob=0.15, gamma=(0.65, 1.5)),

            #RandScaleIntensityd(keys="image", factors=0.1, prob=1.0), not for BM
            #RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0), not for BM
            EnsureTyped(keys=["image", "label"]),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            EnsureTyped(keys=["image", "label"]),
        ]
    )

    # Setup Loaders
    train_ds = Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)
    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=opt.workers)

    device = torch.device("cuda:0")  # Change if multiple gpu training

    ###################################################################################################
    # Model
    ###################################################################################################

    if opt.network == "DUNetV1V2":
        model = DUNetV1V2(img_ch=opt.in_channels, output_ch=opt.out_channels)

    if opt.network == "UNetV1V2":
        model = UNetV1V2(img_ch=opt.in_channels, output_ch=opt.out_channels)


    ####################################################################################################

    ###################################################################################################
    # GPU setting, and loading pretrained variables (if pretraining = True)
    ###################################################################################################

    if num_gpus > 0:
        if num_gpus > 1:
            model.cuda()
            model= torch.nn.DataParallel(model)
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
        datafile_loss = pth_model + '/loss.npy'
        epoch_loss_values = list(np.load(datafile_loss))
        current_epoch = len(epoch_loss_values)
        ## Load the best DSC and epoch of the best DSC
        datafile_best_epoch = pth_model + '/best_epoch.npy'
        values_best_epoch = np.load(datafile_best_epoch)
        ## Load list of metrics (DSC)
        datafile_dice = pth_model + '/dice.npy'
        metric_values = list(np.load(datafile_dice))
        ## Load validation loss
        datafile_loss_val = pth_model + '/loss_val.npy'
        val_loss_values = list(np.load(datafile_loss_val))

    ###################################################################################################

    ###################################################################################################
    # Set loss, optimizer and metric
    ###################################################################################################

    print("The loss function used is: ", opt.loss)
    if opt.loss == 'DiceCELoss':
        loss_function = DiceCELoss(to_onehot_y=True, softmax=True, include_background=False)
    elif opt.loss == 'DiceFocalLoss':
        loss_function = DiceFocalLoss(to_onehot_y=True, softmax=True, include_background=False)
    elif opt.loss == 'GeneralizedDiceLoss':
        loss_function = GeneralizedDiceLoss(include_background=False, to_onehot_y=True, softmax=True)
    elif opt.loss == 'FocalLoss':
        loss_function = FocalLoss(include_background=False, to_onehot_y=True)
    elif opt.loss == 'LocalWeightedLoss':
        loss_function = LocalWeightedLoss(include_background=False, to_onehot_y=True, softmax=True)

    loss_function_val = loss_function
    optimizer = torch.optim.AdamW(model.parameters(), opt.lr, weight_decay=1e-5)
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

    # Create a summary writer for tensorboard
    datafileTB = os.path.join(pth_model, 'TB')
    if not os.path.isdir(datafileTB):
        os.makedirs(datafileTB)

    writer = SummaryWriter(datafileTB)

    for epoch in range(current_epoch, max_epochs):
        print("-" * 40)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] epoch {epoch +1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0

        for batch_data in train_loader:
            step += 1
            inputs, labels = (batch_data['image'], batch_data['label'])

            if int(torch.max(labels)) == 255:
                labels = torch.divide(labels, 255)

            if int(torch.max(labels)) > 1:
                print("Label tensor is not binary!")
                labels = torch.multiply(labels > 0, 1)

            if num_gpus > 0:
                inputs = inputs.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_function(outputs, labels)

            # Visualize inputs and outputs in Tensorboard
            if step == 1:
                image_grid = torchvision.utils.make_grid(inputs, nrow=6, padding=10)
                writer.add_image('Input training data (MRI)', image_grid, global_step=epoch)

                image_grid = torchvision.utils.make_grid(labels, nrow=6, padding=10)
                writer.add_image('Ground truth labels', image_grid, global_step=epoch)

                # Apply the softmax function to the predictions
                m = torch.nn.Softmax(dim=1)
                outputsTB = m(outputs)
                outputsTB1 = outputsTB[:, 1, :, :]  # background first channel (0), predictions second (1)
                outputsTB1 = torch.unsqueeze(outputsTB1, dim=1)
                image_grid = torchvision.utils.make_grid(outputsTB1, nrow=6, padding=10)
                writer.add_image('Predictions', image_grid, global_step=epoch)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // opt.batch_size

            # DSC Metric to Tensorboard for the training set
            m = torch.nn.Softmax(dim=1)
            outputsDSC = m(outputs)
            outputsDSC = outputsDSC[:, 1, :, :]  # background first channel (0), predictions second (1)
            outputsDSC = torch.unsqueeze(outputsDSC, dim=1)
            dice_metric(y_pred=outputsDSC, y=labels)
            
            if (step -1)%10 == 0:
                print(
                    f"{step}/{epoch_len}, "
                    f"train_loss: {loss.item():.4f}")
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        loss_numpy = np.array(epoch_loss_values)

        # Log the losses to TensorBoard
        writer.add_scalar('Training loss', epoch_loss, global_step=epoch)

        metric = dice_metric.aggregate().item()
        dice_metric.reset()
        writer.add_scalar('DSC for training set', metric, global_step=epoch)

        datafile_loss = pth_model + "loss.npy"
        np.save(datafile_loss, loss_numpy)

        ###################################### VALIDATION ######################################################
        im_dim = [272, 347, 299]

        if (epoch + 1) % val_interval == 0:

            # Save model after training phase (every 2 epochs):
            # interesting when pretraining is required (network trained several days)

            # SAVE CURRENT MODEL:
            torch.save(model.state_dict(), os.path.join(
                root_dir, save_name + str(epoch + 1) + ".pth"
            ))

            print("Saving model with name: ", os.path.join(
                root_dir, save_name + str(epoch + 1) + ".pth"))

            val_loss = 0
            step_val = 0
            model.eval()
            with torch.no_grad():

                for val_data in val_loader:

                    step_val = step_val + 1
                    print("Step val:", step_val)
                    val_inputs, val_labels = (val_data['image'], val_data['label'])

                    if opt.gpus != '-1':
                        val_inputs = val_inputs.cuda()
                        val_labels = val_labels.cuda()

                    if val_type == '2D':
                        inferer = SimpleInferer()
                    elif val_type == '3D':
                        inferer = SliceInferer(spatial_dim=2, sw_batch_size=opt.batch_size, roi_size=(im_dim[0], im_dim[1]),
                                               progress=True)
                    val_outputs = inferer(val_inputs, model)

                    # Validation loss
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

                ## SAVE LIST OF DICE VALUES FOR LATER PLOTTING
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
