
# ---------------------------------------------------------------------------------------
# Given a patient index and a metastasis index --> find and save the slice
# ---------------------------------------------------------------------------------------
# Cristina Almagro Perez, ETH Zurich

# Once inference has been performed, we seek to analyse the offsets in false positive
# metastases. This script does the following:

# Given a patient index and a prediction of metastasis index (within that patient) --> find and save the slice
# ---------------------------------------------------------------------------------------
import numpy as np
import nibabel as nib
import glob
import os
import matplotlib.pyplot as plt
import json
from skimage import measure

########################################################################################
# Auxiliary functions
########################################################################################

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


def get_coloured_mask(mask,gt=False):
    """
    random_colour_masks
      parameters:
        - image - predicted masks
      method:
        - the masks of each predicted object is given random colour for visualization
    """
    colours = [[0, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 255], [255, 255, 0], [255, 0, 255], [80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    #r[mask == 1], g[mask == 1], b[mask == 1] = colours[random.randrange(0,10)]
    r[mask == 1], g[mask == 1], b[mask == 1] = colours[3]  #display mask in blue
    if gt:
        r[mask == 1], g[mask == 1], b[mask == 1] = colours[0]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask


# Load the list of patients
datafile = "/scratch_net/biwidl311/Cristina_Almagro/research_project/dataset/Cristina_dataset_test_preprocessed_tight.json"
f = open(datafile)
data = json.load(f)
test = data["testing"]
# 2D DUNetV1V2
pth_pred = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/research_project_results/2Dmodels/DUNetV1V2_08_01_23/ensemble_08_01_2023/'
# 2D UNetV1V2
pth_pred2 = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/metastases_project/Cristina_Almagro/results/ResearchProject/model_2D_DUNetV1V2_03_04_23_exp1/ensemble_03_04_2023/'
# Data type parameters
format_pred = '.npy'
format_gt = '.nii.gz'
format_predk = '*' + format_pred
myfiles = glob.glob(pth_pred + format_predk)
rotation_required = False #False

##########################################################
# Image  FP 1 - patient 52, met 12 (in prediction volume)
# Image  FP 2 - patient 4, met 6 (in prediction volume)
# Image  FP 3 - patient 51, met 1 (in prediction volume)
##########################################################
# FP NEW
# Image  FP 1 - patient 0, met 2 (in prediction volume)
# Image  FP 1 - patient 6, met 5 (in prediction volume)
# Image  FP 1 - patient 2, met 4 (in prediction volume)
##########################################################
patient = 6
met = 5
##########################################################

# Load prediction
ipatient = myfiles[patient]
patient_name = ipatient.split('/')[-1]
if patient_name[-4::] == '.npy' or patient_name[-4::] == '.nii':
    patient_id = patient_name[:-4]
else:
    patient_id = patient_name[:-7]
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
# Load image
im = nib.load(test[count]["image"]).get_fdata()

# Load prediction from UNetV1V2
datafile_pred2 = os.path.join(pth_pred2, patient_name)
pred2 = np.load(datafile_pred2)


from skimage.morphology import ball
print("### Morphological closing ###")
from scipy import ndimage
pred = ndimage.binary_closing(pred, ball(5))
# Remove small objects in the prediction
size_small = 42
print(" Removing objects smaller than:", size_small)
pred = remove_small_objects(pred, size_small)


pred_labelled = measure.label(pred)
pred_met = pred_labelled == met

loc = np.sum(np.sum(pred_met, 0), 0)
loc = np.nonzero(loc)

section = int(np.round(np.mean(loc)))
print("Section is: ", section)

# Calculate the centroid in a given section
pred_slice = pred_met[:, :, section]
indices = np.argwhere(pred_slice == 1)

# Calculate the centroid coordinates
centroid = np.mean(indices, axis=0)
centroid = np.round(centroid)
print("The centroid is", centroid)

# Save image
im_section = im[:, :, section]
im_datafile='/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/attention_unet/test/images_tight1/'
nm_output = patient_name[0:-4] + '_section_' + str(section) + '_FP.npy'
im_datafile = os.path.join(im_datafile, nm_output)
print(im_datafile)
np.save(im_datafile, im_section)


# Plot GT
plt.imshow(im[:, :, int(np.round(np.mean(loc)))], cmap="gray")
gt_coloured = get_coloured_mask(gt[:, :, section], gt=True)
plt.imshow(gt_coloured, alpha=0.3)
plt.title("GT")
plt.show()

# Plot prediction DUNEtV1V2
# Adjust vmax for proper visualization
plt.imshow(im[:, :, int(np.round(np.mean(loc)))], cmap="gray", vmin=0, vmax=0.7213082313537598)
pred_coloured = get_coloured_mask(pred[:, :, section], gt=False)
plt.imshow(pred_coloured, alpha=0.3)
plt.title("Prediction DUNetV1V2")
plt.show()

# Plot prediction UNEtV1V2
plt.imshow(im[:, :, int(np.round(np.mean(loc)))], cmap="gray")
pred_coloured2 = get_coloured_mask(pred2[:, :, section], gt=False)
plt.imshow(pred_coloured2, alpha=0.3)
plt.title("Prediction UNetV1V2")
plt.show()

# Plot image
plt.imshow(im[:, :, int(np.round(np.mean(loc)))], cmap="gray")
gt_coloured[gt_coloured > 0] = 0
plt.imshow(gt_coloured, alpha=0.3)
plt.title("Image")
plt.show()


extra = 25
croppedIM = im[np.min(np.nonzero(pred_met)[0]) - extra: np.max(np.nonzero(pred_met)[0]) + extra,
            np.min(np.nonzero(pred_met)[1]) - extra: np.max(np.nonzero(pred_met)[1]) + extra,
            :]

croppedGT = gt[np.min(np.nonzero(pred_met)[0]) - extra: np.max(np.nonzero(pred_met)[0]) + extra,
            np.min(np.nonzero(pred_met)[1]) - extra: np.max(np.nonzero(pred_met)[1]) + extra,
            :]

croppedPRED = pred[np.min(np.nonzero(pred_met)[0]) - extra: np.max(np.nonzero(pred_met)[0]) + extra,
            np.min(np.nonzero(pred_met)[1]) - extra: np.max(np.nonzero(pred_met)[1]) + extra,
            :]

croppedPRED2 = pred2[np.min(np.nonzero(pred_met)[0]) - extra: np.max(np.nonzero(pred_met)[0]) + extra,
              np.min(np.nonzero(pred_met)[1]) - extra: np.max(np.nonzero(pred_met)[1]) + extra,
              :]

# Zoom in GT
plt.imshow(croppedIM[:, :, int(np.round(np.mean(loc)))], cmap="gray")
gt_coloured = get_coloured_mask(croppedGT[:, :, section], gt=True)
plt.imshow(gt_coloured, alpha=0.3)
plt.title("Cropped GT")
plt.show()

# Zoom in PRED
plt.imshow(croppedIM[:, :, int(np.round(np.mean(loc)))], cmap="gray")
gt_coloured = get_coloured_mask(croppedPRED[:, :, section], gt=False)
plt.imshow(gt_coloured, alpha=0.3)
plt.title("Cropped PRED")
plt.show()

# Zoom in PRED 2
plt.imshow(croppedIM[:, :, int(np.round(np.mean(loc)))], cmap="gray")
gt_coloured = get_coloured_mask(croppedPRED2[:, :, section], gt=False)
plt.imshow(gt_coloured, alpha=0.3)
plt.title("Cropped PRED2")
plt.show()

# Zoom in IMAGE
plt.imshow(croppedIM[:, :, int(np.round(np.mean(loc)))], cmap="gray")
gt_coloured[gt_coloured > 0] = 0
plt.imshow(gt_coloured, alpha=0.3)
plt.title("Cropped Image")
plt.show()

