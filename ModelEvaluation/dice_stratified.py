
########################################################################################
# Cristina Almagro Pérez, ETH Zürich, 2022
########################################################################################

# Save a variable with the volume of the GT metastasis, the volume of the predicted metastasis
# and DSC between the two.
# The generated variable can be used to analyze the performance of the model
# based on the lesion size.

import glob
import os
import numpy as np
from skimage import measure
import scipy.io as sio
import nibabel as nib
#######################################################################################
# Auxiliary functions
#######################################################################################


def compute_dice_patient(gt, pred):

    n = 2 * np.sum(np.multiply(gt, pred).flatten())
    d = np.sum(gt.flatten()) + np.sum(pred.flatten())
    dice_value = n / d

    return dice_value


def compute_iou_patient(gt, pred):

    n = np.sum(np.multiply(gt, pred).flatten())
    d = gt + pred
    d = np.sum(np.multiply(d > 0, 1))
    iou_value = n / d

    return iou_value

def remove_small_objects(pred00, min_size):

    pred_output = np.zeros(pred00.shape)
    pred00_labelled = measure.label(pred00)
    num_pred = np.max(pred00_labelled)
    for z in range(1, num_pred+1):
        my_pred_met = np.multiply(pred00_labelled == z, 1)
        vol_met = np.sum(my_pred_met)
        if not vol_met < min_size:
            pred_output = pred_output + my_pred_met
    return pred_output

############################################################################################
dsc_stratified = True
volume_GT = False
#############################################################################################


small_objects = False
factor = 0.6 * 0.6 * 0.6
################################################################################################
# Two ways of measuring the metastases sizes:
option2 = False  # project in the craniocaudal direction and measure the largest diameter
option3 = True  # project in the craniocaudal direction and measure the largest cross-sectional dimension (Used in MetNet paper)
##################################################################################################
met_size = []

pth_pred = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/research_project_results/attention/mymodel1/ensemble_02_12_2022/'
pth_gt = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/attention_unet/test/masks/'
format_pred = '.npy'
format_gt = '.nii.gz'
format_gtk = '*' + format_gt
rotation_required = True
myfiles = glob.glob(pth_gt + format_gtk)
# small_objects = False

if volume_GT:
    count_met = 0
    volume_data_GT_attention = []
    metsize_data_GT_attention = []

    n_patients = 53
    DSC = []
    count_TP = 0
    count_FN = 0
    count_FP = 0
    FP_patient = []

    for count, ipatient in enumerate(myfiles):

        print("Evaluating patient: " + ipatient.split('/')[-1] + " ### " + str(count) + '/' + str(len(myfiles)))


        # Load GT
        patient_name = ipatient.split('/')[-1]
        datafile_gt = os.path.join(pth_gt, patient_name)
        if format_gt == '.nii.gz' or format_gt == '.nii':
            gt = nib.load(datafile_gt).get_fdata()
        else:
            gt = np.load(datafile_gt)
        gt = np.squeeze(gt)
        if rotation_required:
            gt = np.rot90(gt, 2)

        # Load Prediction
        if patient_name[-4::] == '.npy' or patient_name[-4::] == '.nii':
            patient_id = patient_name[:-4]
        else:
            patient_id = patient_name[:-7]

        patient_name_pred = patient_id + format_pred
        datafile_pred = os.path.join(pth_pred, patient_name_pred)
        if format_pred == '.nii.gz' or format_pred == '.nii':
            pred = nib.load(datafile_pred).get_fdata()
        else:
            pred = np.load(datafile_pred)
        pred = np.squeeze(pred)

        if small_objects:
            size_small = 42  # 42, 63.28
            print(" Removing objects smaller than:", size_small)
            pred = remove_small_objects(pred, size_small)
        else:
            print("No removing any objects")


        ##### Compute detection metrics #####
        gt_labelled = measure.label(gt)
        num_gt_met = np.max(gt_labelled)
        print('The number of GT metastases is :', num_gt_met)

        pred_labelled = measure.label(pred)
        num_pred_met = np.max(pred_labelled)
        print('The number of predicted metastases is :', num_pred_met)

        # Extra mask with labels: 1 - TP, 2 - FN, 3 - FP
        mask_gt_det_labels = np.zeros(np.shape(pred))
        mask_pred_det_labels = np.zeros(np.shape(pred))
        mask_pred_det_labels0 = np.zeros(np.shape(pred))

        # Find TP and FN metastases
        TP = 0
        FN = 0
        for x in range(1, num_gt_met + 1):  # Loop through each GT metastases
            gt_met = gt_labelled == x
            count_met = count_met + 1
            vol_met = np.sum(gt_met) * factor  # volume in mm3

            # Options 2 and 3
            # Common part to both
            met_proj = np.sum(gt_met, axis=2)
            met_proj = met_proj > 0
            met_proj = np.multiply(met_proj, 1)

            # Calculate size
            if option2:
                props = measure.regionprops(met_proj)
                maj_ax_le = props[0].major_axis_length
            elif option3:  # option 3
                loc = np.nonzero(met_proj)
                xmin = np.min(loc[0])
                xmax = np.max(loc[0])
                ymin = np.min(loc[1])
                ymax = np.max(loc[1])
                width1 = xmax - xmin
                width2 = ymax - ymin
                if width1 > width2:
                    maj_ax_le = width1
                else:
                    maj_ax_le = width2
            met_size = maj_ax_le * 0.6


            gt_bb = np.zeros(gt.shape)
            gt_bb[np.min(np.nonzero(gt_met)[0]): np.max(np.nonzero(gt_met)[0]),
            np.min(np.nonzero(gt_met)[1]): np.max(np.nonzero(gt_met)[1]),
            np.min(np.nonzero(gt_met)[2]): np.max(np.nonzero(gt_met)[2])] = 1
            iou_temp_max = 0
            label_tp = 0

            for y in range(1, num_pred_met + 1):
                pred_met = pred_labelled == y
                pred_bb = np.zeros(pred.shape)
                pred_bb[np.min(np.nonzero(pred_met)[0]): np.max(np.nonzero(pred_met)[0]),
                np.min(np.nonzero(pred_met)[1]): np.max(np.nonzero(pred_met)[1]),
                np.min(np.nonzero(pred_met)[2]): np.max(np.nonzero(pred_met)[2])] = 1

                iou_temp = compute_iou_patient(gt_bb, pred_bb)
                # print(iou_temp)
                if iou_temp > iou_temp_max:
                    iou_temp_max = iou_temp
                    label_tp = y
            if iou_temp_max > 0.1:
                TP = TP + 1
                var = [vol_met, 1]
                volume_data_GT_attention.append(var)
                var_size = [met_size, 1]
                metsize_data_GT_attention.append(var_size)


            else:
                FN = FN + 1
                var = [vol_met, 2]
                volume_data_GT_attention.append(var)
                var_size = [met_size, 2]
                metsize_data_GT_attention.append(var_size)

    # Save as mat and npy file
    # Save variable volume
    output_datafile = '/scratch_net/biwidl311/Cristina_Almagro/big/mymodel/' + 'mymodel1_volume_data_GT_attention'
    mat_datafile = output_datafile +'.mat'
    sio.savemat(mat_datafile, {'volume_data_GT_attention': volume_data_GT_attention})
    np.save(output_datafile + '.npy', volume_data_GT_attention)

    # Save variable size
    output_datafile = '/scratch_net/biwidl311/Cristina_Almagro/big/mymodel/' + 'mymodel1_metsize_data_GT_attention'
    mat_datafile = output_datafile +'.mat'
    sio.savemat(mat_datafile, {'metsize_data_GT_attention': metsize_data_GT_attention})
    np.save(output_datafile + '.npy', metsize_data_GT_attention)




if dsc_stratified:

    pth_pred = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/metastases_project/Meva-share/predTs-newnnUNet/'
    pth_gt = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/attention_unet/test/masks/'
    dsc_stratified_Attention = []  # First column: GT met. volume. Second col.: pred. met volume. The third the dice coefficient
    format_pred = '.nii.gz'
    format_gt = '.nii.gz'
    format_gtk = '*' + format_gt
    rotation_required = False
    myfiles = glob.glob(pth_gt + format_gtk)

    # Print configurations #
    print("pth_pred: ", pth_pred)
    print("pth_gt: ", pth_gt)
    print("format_pred: ", format_pred)
    print("format_gt: ", format_gt)

    for count, ipatient in enumerate(myfiles):

        print("Evaluating patient: " + ipatient.split('/')[-1] + " ### " + str(count + 1) + '/' + str(len(myfiles)))

        # Load GT
        patient_name = ipatient.split('/')[-1]
        datafile_gt = os.path.join(pth_gt, patient_name)
        if format_gt == '.nii.gz' or format_gt == '.nii':
            gt = nib.load(datafile_gt).get_fdata()
        else:
            gt = np.load(datafile_gt)
        gt = np.squeeze(gt)
        if rotation_required:
            gt = np.rot90(gt, 2)


        # Load Prediction
        if patient_name[-4::] == '.npy' or patient_name[-4::] == '.nii':
            patient_id = patient_name[:-4]
        else:
            patient_id = patient_name[:-7]

        patient_name_pred = patient_id + format_pred
        datafile_pred = os.path.join(pth_pred, patient_name_pred)
        if format_pred == '.nii.gz' or format_pred == '.nii':
            pred = nib.load(datafile_pred).get_fdata()
        else:
            pred = np.load(datafile_pred)
        pred = np.squeeze(pred)

        ##### Compute detection metrics #####
        #print(np.unique(gt))
        gt_labelled = measure.label(gt)
        num_gt_met = np.max(gt_labelled)
        print('The number of GT metastases is :', num_gt_met)

        pred_labelled = measure.label(pred)
        num_pred_met = np.max(pred_labelled)
        print('The number of predicted metastases is :', num_pred_met)

        y_taken = []
        for x in range(1, num_gt_met+1):  # Loop through each GT metastases

            gt_met = gt_labelled == x

            # Options 2 and 3
            # Common part to both
            met_proj = np.sum(gt_met, axis=2)
            met_proj = met_proj > 0
            met_proj = np.multiply(met_proj, 1)

            # Calculate size GT metastasis
            if option2:
                props = measure.regionprops(met_proj)
                maj_ax_le = props[0].major_axis_length
            elif option3:  # option 3
                loc = np.nonzero(met_proj)
                xmin = np.min(loc[0])
                xmax = np.max(loc[0])
                ymin = np.min(loc[1])
                ymax = np.max(loc[1])
                width1 = xmax - xmin
                width2 = ymax - ymin
                if width1 > width2:
                    maj_ax_le = width1
                else:
                    maj_ax_le = width2
            met_size = maj_ax_le * 0.6


            gt_bb = np.zeros(gt.shape)
            gt_bb[np.min(np.nonzero(gt_met)[0]): np.max(np.nonzero(gt_met)[0]), np.min(np.nonzero(gt_met)[1]): np.max(np.nonzero(gt_met)[1]), np.min(np.nonzero(gt_met)[2]): np.max(np.nonzero(gt_met)[2])] = 1
            iou_temp_max = 0
            label_tp = 0

            for y in range(1, num_pred_met+1):
                pred_met = pred_labelled == y
                pred_bb = np.zeros(pred.shape)
                pred_bb[np.min(np.nonzero(pred_met)[0]): np.max(np.nonzero(pred_met)[0]), np.min(np.nonzero(pred_met)[1]): np.max(np.nonzero(pred_met)[1]), np.min(np.nonzero(pred_met)[2]): np.max(np.nonzero(pred_met)[2])] = 1

                iou_temp = compute_iou_patient(gt_bb, pred_bb)
                #print(iou_temp)
                if iou_temp > iou_temp_max:
                    iou_temp_max = iou_temp
                    label_tp = y


            if iou_temp_max > 0.05:  #0.1
                result = y_taken.count(label_tp)
                if result > 0:
                    raise Warning("Predicted label already used")
                else:
                    y_taken.append(label_tp)
                dsc_temp_max = compute_dice_patient(gt_met, np.multiply(pred_labelled == label_tp, 1))
            else:
                dsc_temp_max = 0

            # Calculate proportion of occupied volume
            d = np.sum(gt_met)
            n = np.sum(np.multiply(gt_met, pred))
            value_prop = (n / d) * 100  # proportion of occupied volume for this met

            var = [met_size, dsc_temp_max, value_prop]
            print(var)
            dsc_stratified_Attention.append(var)

    # Save variable size
    output_datafile = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/metastases_project/big/' + 'dsc_stratified_TwoKernelUNet_28_02_23_exp1'
    mat_datafile = output_datafile + '.mat'
    sio.savemat(mat_datafile, {'dsc_stratified_TwoKernelUNet_28_02_23_exp1': dsc_stratified_Attention})
    np.save(output_datafile + '.npy', dsc_stratified_Attention)

