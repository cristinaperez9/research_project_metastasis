
###################################################################################################
# Cristina Almagro Perez, 2023, ETH Zürich
###################################################################################################
# Evaluation of 2D models
###################################################################################################
import glob
import numpy as np
import os
import nibabel as nib
import json
from statistics import mean, stdev
from skimage import measure
import scipy.io as sio
import math
from scipy import ndimage
from init_2D import Options
opt = Options().parser()


DSC2D = True
DSCStratified = True
DetectionMetrics3D = True


#################################################################################################
# Auxiliary functions
#################################################################################################
def compute_dice(gt, pred,dim):
    if dim == 2:
        gamma = 0.0000000000000000000000000000000000001
    else:  #dim=3
        gamma = 0

    n = 2 * np.sum(np.multiply(gt, pred).flatten()) # Numerator
    d = np.sum(gt.flatten()) + np.sum(pred.flatten()) # Discriminator
    dice_value = (n + gamma) / (d + gamma)

    return dice_value

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

def compute_iou_patient(gt, pred):

    n = np.sum(np.multiply(gt, pred).flatten())
    d = gt + pred
    d = np.sum(np.multiply(d > 0, 1))
    iou_value = n / d

    return iou_value
###################################################################################################
# Load test masks (tight)
datafile = "/scratch_net/biwidl311/Cristina_Almagro/research_project/dataset/Cristina_dataset_test_preprocessed_tight.json"
f = open(datafile)
data = json.load(f)
test = data["testing"]

# Predictions directory
# 2D DUNetV1V2
#pth_pred = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/research_project_results/2Dmodels/DUNetV1V2_08_01_23/ensemble_08_01_2023/'
# 2D UNetV1V2
#pth_pred = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/metastases_project/Cristina_Almagro/results/ResearchProject/model_2D_DUNetV1V2_03_04_23_exp1/ensemble_03_04_2023/'

pth_pred = os.path.join(opt.outpth_inference, 'ensemble_' + opt.date)
print("Prediction pth:", pth_pred)
# Data type parameters
format_pred = '.npy'
format_gt = '.nii.gz'
format_predk = '*' + format_pred
rotation_required = False
myfiles = glob.glob(pth_pred + format_predk)
# Define output directory
output_path = os.path.join(pth_pred, 'Segmentation_Detection_metrics/')
if not os.path.exists(output_path):
    os.makedirs(output_path)

##########################################################################################
# Calculate 2D DSC as in "2.5D and 3D segmentation of brain metastases with deep learning
# on multinational MRI data"
##########################################################################################

if DSC2D:
    DSCcohort=[]
    for count, ipatient in enumerate(myfiles):
        print("Evaluating patient: " + ipatient.split('/')[-1] + " ### " + str(count + 1) + '/' + str(len(myfiles)))
        ########################### Load mask & prediction #############################

        patient_name = ipatient.split('/')[-1]
        if patient_name[-4::] == '.npy' or patient_name[-4::] == '.nii':
            patient_id = patient_name[:-4]
        else:
            patient_id = patient_name[:-7]

        # Load prediction
        datafile_pred = os.path.join(pth_pred, patient_name)
        if format_pred == '.nii.gz' or format_pred == '.nii':
            pred = nib.load(datafile_pred).get_fdata()
        else:
            pred = np.load(datafile_pred)
            patient_id = patient_name[0:-4]
        pred = np.squeeze(pred)

        # Find GT directory
        found = 0
        count = 0
        while found == 0:
            ppth_gt = test[count]["label"]
            if patient_id in ppth_gt:
                found = 1
                datafile_gt = ppth_gt
            else:
                count = count + 1

        # Load ground truth
        if format_gt == '.nii.gz' or format_gt == '.nii':
            gt = nib.load(datafile_gt).get_fdata()
        else:
            gt = np.load(datafile_gt)
        gt = np.squeeze(gt)
        if rotation_required:
            gt = np.rot90(gt, 2)

        # Loop through all slices and compute DSC
        num_slices = gt.shape[2]
        DSCPatient = []
        for kk in range(0, num_slices):
            sliceGT = gt[:, :, kk]
            slicePRED = pred[:, :, kk]
            sliceDSC = compute_dice(sliceGT, slicePRED, dim=2)
            #print("The dice value of this slice is...", sliceDSC)
            DSCPatient.append(sliceDSC)
        DSCcohort.append(mean(DSCPatient))
        #print("The dice value of this patient is...", mean(DSCPatient))

    # 2D DSC
    average_dice = mean(DSCcohort)
    std_dice = stdev(DSCcohort)
    print("The average 2D DSC is...", average_dice)
    print("The standard deviation of 2D DSC is...", std_dice)

###########################################################################################
# For each GT metastases, store the GT volume, GT size (mm), predicted volume
# and DSC
###########################################################################################
factor = 0.6 * 0.6 * 0.6
option2 = False  # project in the craniocaudal direction and measure the largest diameter
option3 = True  # project in the craniocaudal direction and measure the largest cross-sectional dimension (Used in MetNet paper)

if DSCStratified:
    dsc_stratified = []
    for count_patient, ipatient in enumerate(myfiles):
        print("Evaluating patient: " + ipatient.split('/')[-1] + " ### " + str(count_patient + 1) + '/' + str(len(myfiles)))
        ########################### Load mask & prediction #############################

        patient_name = ipatient.split('/')[-1]
        if patient_name[-4::] == '.npy' or patient_name[-4::] == '.nii':
            patient_id = patient_name[:-4]
        else:
            patient_id = patient_name[:-7]

        # Load prediction
        datafile_pred = os.path.join(pth_pred, patient_name)
        if format_pred == '.nii.gz' or format_pred == '.nii':
            pred = nib.load(datafile_pred).get_fdata()
        else:
            pred = np.load(datafile_pred)
            patient_id = patient_name[0:-4]
        pred = np.squeeze(pred)

        # Find GT directory
        found = 0
        count = 0
        while found == 0:
            ppth_gt = test[count]["label"]
            if patient_id in ppth_gt:
                found = 1
                datafile_gt = ppth_gt
            else:
                count = count + 1

        # Load ground truth
        if format_gt == '.nii.gz' or format_gt == '.nii':
            gt = nib.load(datafile_gt).get_fdata()
        else:
            gt = np.load(datafile_gt)
        gt = np.squeeze(gt)
        if rotation_required:
            gt = np.rot90(gt, 2)

        gt_labelled = measure.label(gt)
        num_gt_met = np.max(gt_labelled)
        print('The number of GT metastases is :', num_gt_met)

        for x in range(1, num_gt_met + 1):  # Loop through each GT metastases
            gt_met = gt_labelled == x
            vol_met = np.sum(gt_met) * factor


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

            # Calculate bounding box of the ground truth metastasis
            extra=25

            croppedGT = gt_met[np.min(np.nonzero(gt_met)[0])-extra: np.max(np.nonzero(gt_met)[0])+extra, np.min(np.nonzero(gt_met)[1])-extra: np.max(np.nonzero(gt_met)[1])+extra, np.min(np.nonzero(gt_met)[2])-extra: np.max(np.nonzero(gt_met)[2])+extra]
            croppedPREDICTION = pred[np.min(np.nonzero(gt_met)[0])-extra: np.max(np.nonzero(gt_met)[0])+extra, np.min(np.nonzero(gt_met)[1])-extra: np.max(np.nonzero(gt_met)[1])+extra, np.min(np.nonzero(gt_met)[2])-extra: np.max(np.nonzero(gt_met)[2])+extra]
            # Metastasis DSC
            MetDSC = compute_dice(croppedGT, croppedPREDICTION, dim=3)
            if math.isnan(MetDSC):
                MetDSC = 0
            # Volume of the predicted metastasis
            vol_pred = np.sum(croppedPREDICTION) * factor
            # Variable to save
            var = [vol_met, vol_pred, met_size, MetDSC, count_patient]
            print(var)
            dsc_stratified.append(var)
    # Save variable size
    output_datafile = os.path.join(output_path, 'dsc_stratified_2D_')
    mat_datafile = output_datafile + '.mat'
    sio.savemat(mat_datafile, {'dsc_stratified_2D_': dsc_stratified})
    np.save(output_datafile + '.npy', dsc_stratified)
###################################################################################################

if DetectionMetrics3D:
    print("Calculating Detection Metrics")
    DSC = []
    dsc_tp_all = []
    count_TP = 0
    count_FN = 0
    count_FP = 0
    FP_patient = []
    sensitivity_patient = []
    vol_label_tp_fn = []  # First column volume of the metastasis, second label: 1 TP, 2FN
    fp_list = []
    dsc_tp_online = True
    small_objects = True
    for count_patient, ipatient in enumerate(myfiles):
        print("Evaluating patient: " + ipatient.split('/')[-1] + " ### " + str(count_patient + 1) + '/' + str(len(myfiles)))
        ########################### Load mask & prediction #############################

        patient_name = ipatient.split('/')[-1]
        if patient_name[-4::] == '.npy' or patient_name[-4::] == '.nii':
            patient_id = patient_name[:-4]
        else:
            patient_id = patient_name[:-7]

        # Load prediction
        datafile_pred = os.path.join(pth_pred, patient_name)
        if format_pred == '.nii.gz' or format_pred == '.nii':
            pred = nib.load(datafile_pred).get_fdata()
        else:
            pred = np.load(datafile_pred)
            patient_id = patient_name[0:-4]
        pred = np.squeeze(pred)

        from skimage.morphology import ball
        print("### Morphological closing ###")
        pred = ndimage.binary_closing(pred, ball(5))

        # Find GT directory
        found = 0
        count = 0
        while found == 0:
            ppth_gt = test[count]["label"]
            if patient_id in ppth_gt:
                found = 1
                datafile_gt = ppth_gt
            else:
                count = count + 1

        # Load ground truth
        if format_gt == '.nii.gz' or format_gt == '.nii':
            gt = nib.load(datafile_gt).get_fdata()
        else:
            gt = np.load(datafile_gt)
        gt = np.squeeze(gt)
        if rotation_required:
            gt = np.rot90(gt, 2)

        if small_objects:
            size_small = 42  # 42, 63.28
            print(" Removing objects smaller than:", size_small)
            pred = remove_small_objects(pred, size_small)
        else:
            print("No removing any objects")

        image_tp = np.zeros(gt.shape)

        ###### Compute DSC and store in vector #####
        dice_patient = compute_dice(gt, pred, dim=3)
        print("The dice value of this patient is...", dice_patient)
        DSC.append(dice_patient)

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
            extra = 25
            croppedGT = gt_met[np.min(np.nonzero(gt_met)[0]) - extra: np.max(np.nonzero(gt_met)[0]) + extra,
                        np.min(np.nonzero(gt_met)[1]) - extra: np.max(np.nonzero(gt_met)[1]) + extra,
                        np.min(np.nonzero(gt_met)[2]) - extra: np.max(np.nonzero(gt_met)[2]) + extra]
            croppedPREDICTION = pred[np.min(np.nonzero(gt_met)[0]) - extra: np.max(np.nonzero(gt_met)[0]) + extra,
                                np.min(np.nonzero(gt_met)[1]) - extra: np.max(np.nonzero(gt_met)[1]) + extra,
                                np.min(np.nonzero(gt_met)[2]) - extra: np.max(np.nonzero(gt_met)[2]) + extra]
            # Metastasis DSC
            MetDSC = compute_dice(croppedGT, croppedPREDICTION, dim=3)
            print("MetDSC:", MetDSC)



            # Calculate size of the metastasis
            met_proj = np.sum(gt_met, axis=2)
            met_proj = met_proj > 0
            met_proj = np.multiply(met_proj, 1)
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

            #vol_met = np.sum(gt_met) * factor  # volume in mm3

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
                if iou_temp > iou_temp_max:
                    iou_temp_max = iou_temp
                    label_tp = y
            if iou_temp_max > 0.1:
                image_tp = image_tp + np.multiply(pred_labelled == label_tp, 1)
                TP = TP + 1
                print("Met is TP")
                var = [met_size, 1, count_patient, x]
                print(var)
                vol_label_tp_fn.append(var)
                mask_gt_det_labels = mask_gt_det_labels + np.multiply(gt_met, 1)
                mask_pred_det_labels = mask_pred_det_labels + np.multiply(pred_labelled == label_tp, 1)
                mask_pred_det_labels0 = mask_pred_det_labels0 + np.multiply(pred_labelled == label_tp, 1)
            else:
                FN = FN + 1
                print("Met is FN")
                mask_gt_det_labels = mask_gt_det_labels + np.multiply(gt_met, 2)
                mask_pred_det_labels = mask_pred_det_labels + np.multiply(gt_met, 2)
                var = [met_size, 2, count_patient, x]
                print(var)
                vol_label_tp_fn.append(var)

        print("The total number of TP in this patient is...", TP)
        print("The total number of FN in this patient is...", FN)

        # Calculate sensitivity per patient
        sp = TP / (TP + FN)
        sensitivity_patient.append(sp)

        # Calculate the number of FP
        mask_fp = pred - mask_pred_det_labels0  # mask_pred_det_labels contains the TP detected metastases
        FP = np.max(measure.label(mask_fp))
        mask_pred_det_labels = mask_pred_det_labels + np.multiply(mask_fp, 3)
        FP_patient.append(FP)
        # Find false positive indices
        mask_fp_labelled = measure.label(mask_fp)
        pred_labelled = measure.label(pred)
        for iii in range(1, np.max(mask_fp_labelled)+1):
            # calculate the size of the FP
            fp_met = mask_fp_labelled == iii
            fp_met_proj = np.sum(fp_met, axis=2)
            fp_met_proj = fp_met_proj > 0
            fp_met_proj = np.multiply(fp_met_proj, 1)
            loc = np.nonzero(fp_met_proj)
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
            fp_met_size = maj_ax_le * 0.6

            # calculate the index in the prediction
            index_pred = np.multiply(np.multiply(fp_met > 0, 1), pred_labelled)
            print(np.unique(index_pred))
            index_pred = np.max(index_pred)

            # create variable for saving
            print("FP met")
            var_fp = [fp_met_size, count_patient, index_pred] #size, patient, index in prediction
            print(var_fp)
            fp_list.append(var_fp)


        print("The total number of FP in this patient is...", FP)
        count_TP += TP
        count_FP += FP
        count_FN += FN

        if dsc_tp_online:
            dice_patient = compute_dice(gt, image_tp,dim=3)
            print("The dice value (tp) of this patient is...", dice_patient)
            if dice_patient > 0:
                dsc_tp_all.append(dice_patient)

    if dsc_tp_online:
        average_dice = mean(dsc_tp_all)
        std_dice = stdev(dsc_tp_all)
        print("The average DSC (tp) is...", average_dice)
        print("The standard deviation of DSC (tp) is...", std_dice)

    # Report TP, FP, FN
    print("The total number of TP is:", count_TP)
    print("The total number of FN is:", count_FN)
    print("The total number of FP is:", count_FP)

    # Calculate precision and recall
    recall = count_TP / (count_TP + count_FN)
    precision = count_TP / (count_TP + count_FP)
    print("The recall is...", recall)
    print("The precision is...", precision)

    # Calculate F1
    F1 = 2 * precision * recall / (precision + recall)
    print("The F1 score is...", F1)

    # Calculate FP per patient
    average_FP = mean(FP_patient)
    std_FP = stdev(FP_patient)
    print("The average FP/patient is...", average_FP)
    print("The standard deviation of FP is...", std_FP)

    # Calculate sensitivity per patient
    average_sensitivity = mean(sensitivity_patient)
    std_sensitivity = stdev(sensitivity_patient)
    print("The average sensitivity per patient is...", average_sensitivity)
    print("The standard deviation of sensitivity per patient is...", std_sensitivity)

    # Save variable FP
    print("Saving variable with metastasis volume and label: 1 (FP)")
    output_datafile = os.path.join(output_path, 'FP_Indices')
    mat_datafile = output_datafile + '.mat'
    sio.savemat(mat_datafile, {'FP_Indices': fp_list})
    np.save(output_datafile + '.npy',  fp_list)

    # Calculate DSC
    average_dice = mean(DSC)
    std_dice = stdev(DSC)
    print("The average DSC is...", average_dice)
    print("The standard deviation of DSC is...", std_dice)

    # Save variable for TP/FN plot
    print("Saving variable with metastasis volume and label: 1 (TP), 2(FN)")
    output_datafile = os.path.join(output_path, 'TP_FN_Indices')
    mat_datafile = output_datafile + '.mat'
    sio.savemat(mat_datafile, {'TP_FN_indices': vol_label_tp_fn})
    np.save(output_datafile + '.npy',  vol_label_tp_fn)