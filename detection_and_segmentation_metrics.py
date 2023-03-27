
#################################################################################################
# Cristina Almagro Pérez, 08/09/2022, ETH University

# Calculate Precision, Recall, F1 score and average FP/per patient
#################################################################################################


# Import necessary packages
import numpy as np
import nibabel as nib
from skimage import measure
import os
from statistics import mean, stdev
import glob
import json
import scipy.io as sio


######################################################################################
# Auxiliary functions
######################################################################################


# Auxiliary functions
def compute_iou_patient(gt, pred):

    n = np.sum(np.multiply(gt, pred).flatten())
    d = gt + pred
    d = np.sum(np.multiply(d > 0, 1))
    iou_value = n / d

    return iou_value

def compute_dice_patient(gt, pred):

    n = 2 * np.sum(np.multiply(gt, pred).flatten())
    d = np.sum(gt.flatten()) + np.sum(pred.flatten())
    dice_value = n / d

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


# Create and save images with 3 labels: 1 (true positive mets), 2(false negative mets)
# 3 (false positive mets)
create_labeled_predictions = False

#######################################################################################
# Algorithm evaluation
#######################################################################################
# Please specify the following:
# pth_pred: directory containing the output segmentation masks (0-1 values).
# format_pred: format of the predictions, either '.nii.gz' (nifti) or '.npy' (numpy).
# pth_gt: directory containing the ground truth segmentation masks (0-1 values).
# format_gt: format of the ground truth masks, either '.nii.gz' (nifti) or '.npy' (numpy).

# Parameters that do not require modification
dsc_tp_online = True   # (also output separately the DSC of the metastases correctly detected).
small_objects = False  # Preprocess the ouput segmentation masks removing small objects in the prediction.
rotation_required = True  # In case the predictions and ground truth masks are rotated with respect to one another.

# Example_ Attention U-Net
pth_pred = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/research_project_results/attention/mymodel1/ensemble_02_12_2022/'
pth_gt = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/attention_unet/test/masks/'
format_pred = '.npy'
format_gt = '.nii.gz'
format_gtk = '*' + format_gt
myfiles = glob.glob(pth_gt + format_gtk)
outpth_labeled_pred = ''


format_gtk = '*' + format_gt
if dsc_tp_online:
    dsc_tp_all = []


# Print configurations #
print("pth_pred: ", pth_pred)
print("pth_gt: ", pth_gt)
print("format_pred: ", format_pred)
print("format_gt: ", format_gt)


n_patients = 53
DSC = []
count_TP = 0
count_FN = 0
count_FP = 0
FP_patient = []

for count, ipatient in enumerate(myfiles):

    print("Evaluating patient: " + ipatient.split('/')[-1] + " ### " + str(count+1) + '/' + str(len(myfiles)))

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

    print(gt.shape)
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
    print(pred.shape)
    if small_objects:
        size_small = 42  #42, 63.28
        print(" Removing objects smaller than:", size_small)
        pred = remove_small_objects(pred, size_small)
    else:
        print("No removing any objects")

    image_tp = np.zeros(gt.shape)

    ###### Compute DSC and store in vector #####
    dice_patient = compute_dice_patient(gt, pred)
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
    for x in range(1, num_gt_met+1):  # Loop through each GT metastases
        gt_met = gt_labelled == x


        gt_bb = np.zeros(gt.shape)
        gt_bb[np.min(np.nonzero(gt_met)[0]): np.max(np.nonzero(gt_met)[0]), np.min(np.nonzero(gt_met)[1]): np.max(np.nonzero(gt_met)[1]), np.min(np.nonzero(gt_met)[2]): np.max(np.nonzero(gt_met)[2])] = 1
        iou_temp_max = 0
        label_tp = 0

        for y in range(1, num_pred_met+1):
            pred_met = pred_labelled == y
            pred_bb = np.zeros(pred.shape)
            pred_bb[np.min(np.nonzero(pred_met)[0]): np.max(np.nonzero(pred_met)[0]), np.min(np.nonzero(pred_met)[1]): np.max(np.nonzero(pred_met)[1]), np.min(np.nonzero(pred_met)[2]): np.max(np.nonzero(pred_met)[2])] = 1

            iou_temp = compute_iou_patient(gt_bb, pred_bb)
            if iou_temp > iou_temp_max:
                iou_temp_max = iou_temp
                label_tp = y
        if iou_temp_max > 0.1:
            image_tp = image_tp + np.multiply(pred_labelled == label_tp, 1)
            TP = TP + 1
            mask_gt_det_labels = mask_gt_det_labels + np.multiply(gt_met, 1)
            mask_pred_det_labels = mask_pred_det_labels + np.multiply(pred_labelled == label_tp, 1)
            mask_pred_det_labels0 = mask_pred_det_labels0 + np.multiply(pred_labelled == label_tp, 1)
        else:
            FN = FN + 1
            mask_gt_det_labels = mask_gt_det_labels + np.multiply(gt_met, 2)
            mask_pred_det_labels = mask_pred_det_labels + np.multiply(gt_met, 2)

    print("The total number of TP in this patient is...", TP)
    print("The total number of FN in this patient is...", FN)

    # Calculate the number of FP
    mask_fp = pred - mask_pred_det_labels0  # mask_pred_det_labels contains the TP detected metastases
    FP = np.max(measure.label(mask_fp))
    mask_pred_det_labels = mask_pred_det_labels + np.multiply(mask_fp, 3)
    FP_patient.append(FP)

    print("The total number of FP in this patient is...", FP)
    count_TP += TP
    count_FP += FP
    count_FN += FN

    if create_labeled_predictions:
        datafile_pred_labelled = os.path.join(outpth_labeled_pred, patient_name)
        np.save(datafile_pred_labelled, mask_pred_det_labels)
        print("Printing unique labels...")
        print(np.unique(mask_pred_det_labels))

    if dsc_tp_online:
        dice_patient = compute_dice_patient(gt, image_tp)
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

#Calculate precision and recall
recall = count_TP/(count_TP + count_FN)
precision = count_TP/(count_TP + count_FP)
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

#Calculate DSC
average_dice = mean(DSC)
std_dice = stdev(DSC)
print("The average DSC is...", average_dice)
print("The standard deviation of DSC is...", std_dice)

# Save list of DSC and patients
mydata = {}
mydata['Dice'] = DSC
mydata['patients'] = myfiles
output_datafile = os.path.join(pth_pred, 'dice_and_patients_UNet_28_02_23.json')
with open(output_datafile, "w") as write_file:
    json.dump(mydata, write_file)
