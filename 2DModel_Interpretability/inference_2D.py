
from init_2D import Options
import numpy as np
import torch
import os
torch.multiprocessing.set_sharing_strategy('file_system')
from datetime import datetime

from monai.inferers import sliding_window_inference, SimpleInferer, SliceInferer
from monai.data import (CacheDataset, DataLoader, decollate_batch, load_decathlon_datalist, Dataset)
from monai.metrics import DiceMetric, HausdorffDistanceMetric

from monai.transforms import (
    AsDiscrete,
    Activations,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    Spacingd,
    EnsureTyped,
    EnsureType,   
    NormalizeIntensityd,
    ScaleIntensityd,)

from network_deformable_2D import UNetV1V2, DUNetV1V2
opt = Options().parser()

##########################################################################################
# Auxiliary functions
##########################################################################################

def compute_dice_patient(gt, pred):

    n = 2 * np.sum(np.multiply(gt, pred).flatten())
    d = np.sum(gt.flatten()) + np.sum(pred.flatten())
    dice_value = n / d

    return dice_value


def main():
    datasets = opt.dataset_folder_test
    val_files = load_decathlon_datalist(datasets, True, "testing")

    ###################################################################################################
    # Data Loaders
    ###################################################################################################

    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            EnsureTyped(keys=["image", "label"]),
        ]
    )

    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=opt.workers)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Dataset {opt.dataset_folder_test[-12:]} on model {opt.network}")
    device = torch.device("cuda:0")

    #########################################################################################################
    # Model
    #########################################################################################################

    if opt.network == "DUNetV1V2":
        model = DUNetV1V2(img_ch=opt.in_channels, output_ch=opt.out_channels)

    if opt.network == "UNetV1V2":
        model = UNetV1V2(img_ch=opt.in_channels, output_ch=opt.out_channels)

    ##########################################################################################################
    # Set model to GPU
    if opt.gpus != '-1':
        num_gpus = len(opt.gpus.split(','))
    else:
        num_gpus = 0

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

    #########################################################################################################
    # Inference in five best epochs according to the validation set
    #########################################################################################################

    # Load name of the five best epochs
    pth_model = opt.model_folder
    datafile_best_epoch = pth_model + '/best_epoch.npy'
    values_best_epoch = np.load(datafile_best_epoch)
    values_best_epoch = values_best_epoch[1, ].astype('int32')


    outpth0 = opt.outpth_inference
    date = opt.date

    cropping = False
    im_dim = [272, 347, 299]

    for count, iepoch in enumerate(values_best_epoch):
        print("Inference in best epoch:" + str(count+1) + '/5' + '. Epoch name: ' + str(iepoch))

        # Define and load the pretrained model
        name_model = opt.network + str(iepoch) + '.pth'
        test_pretrain = os.path.join(pth_model, name_model)
        print(" The datafile of the model is:" + test_pretrain)
        model.load_state_dict(torch.load(test_pretrain))
        model.eval()
        model.to(device)

        #Define outpth to save the results
        output_name = date + '_epoch' + str(iepoch)
        outpth = os.path.join(outpth0, output_name)
        if not os.path.isdir(outpth):
            os.makedirs(outpth)
        print("Outpth is: ", outpth)

        post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=2)])
        post_label = Compose([EnsureType(), AsDiscrete(to_onehot=2)])
        dice_metric = DiceMetric(include_background=False, reduction="mean")

        case_num = 0
        dice_values = []

        # Load patient names
        import json
        datafile = opt.dataset_folder_test
        f = open(datafile)
        data = json.load(f)
        data = data['testing']

        with torch.no_grad():
            for val_data in val_loader:

                # Print patient name
                nm = data[case_num]['image']
                nm = nm.split("/")[-2]
                patient_name = nm + '.nii.gz'
                print("Performing inference on patient:" + nm)
                case_num += 1

                val_inputs, val_labels = (val_data['image'], val_data['label'])
                if opt.gpus != '-1':
                    val_inputs = val_inputs.cuda()
                    val_labels = val_labels.cuda()

                print("The size of val inputs is...", val_inputs.size())
                # Inference for each of the axial slices
                inferer = SliceInferer(spatial_dim=2, sw_batch_size=1, roi_size=(im_dim[0], im_dim[1]), progress=True)
                val_outputs = inferer(val_inputs, model)

                val_outputs_ = [post_pred(i) for i in decollate_batch(val_outputs)]
                val_labels_ = [post_label(i) for i in decollate_batch(val_labels)]

                ##################################################################
                ### Extract forward cropping information after preprocessing ###
                if cropping:
                    mia = val_outputs_[0]
                    # a = mia[1]  ## prediction tensor
                    b = mia.applied_operations[1]  ## dictionary
                    b = b['extra_info']
                    b = b['cropped']

                ### Obtain predictions ###
                pred_mask = torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, :]
                if cropping:
                    final_mask = np.zeros((im_dim[0], im_dim[1], im_dim[2]))
                    final_mask[b[0]:-b[1], b[2]:-b[3], b[4]:-b[5]] = pred_mask
                else:
                    final_mask = pred_mask

                ### Calculate DICE ###
                dice_metric(y_pred=val_outputs_, y=val_labels_)
                dice = dice_metric.aggregate().item()
                print("Patient DSC: ", dice)
                dice_metric.reset()
                dice_values.append(dice)

                ## Save prediction mask ##
                output_datafile = os.path.join(outpth, patient_name[:-7])
                np.save(output_datafile, final_mask)

            print("The average DSC is...", np.mean(dice_values))
            print("The standard deviation of DSC is...", np.std(dice_values))

if __name__ == "__main__":
    main()
