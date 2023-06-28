
# Combine results from the 5 best epochs
# Inputs to this algorithm are the binary segmentation masks

#######################################################################################
# Cristina Almagro Perez, 2022, ETH Zurich
#######################################################################################
import os
import numpy as np
import nibabel as nib
from init_2D import Options
opt = Options().parser()

def compute_dice_patient(gt_vol, pred):

    n = 2 * np.sum(np.multiply(gt_vol, pred).flatten())
    d = np.sum(gt_vol.flatten()) + np.sum(pred.flatten())
    dice_value = n / d

    return dice_value


# Define the date:
pth_model = opt.model_folder
date = opt.date
print("Path model is: ", pth_model)
datafile_best_epoch = pth_model + '/best_epoch.npy'
values_best_epoch = np.load(datafile_best_epoch)
best_epochs = values_best_epoch[1, ].astype('int32')

# Define the root path with the predictions
pth0 = opt.outpth_inference

# GT AFTER CORRECT PREPROCESSING
pth_gt = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/attention_unet/test/masks_tight/'
#########################################################################################
# Original
#########################################################################################

pth1 = os.path.join(pth0, date + '_epoch' + str(best_epochs[0]))
pth2 = os.path.join(pth0, date +'_epoch' + str(best_epochs[1]))
pth3 = os.path.join(pth0, date + '_epoch' + str(best_epochs[2]))
pth4 = os.path.join(pth0, date + '_epoch' + str(best_epochs[3]))
pth5 = os.path.join(pth0, date + '_epoch' + str(best_epochs[4]))

outpth = os.path.join(pth0,'ensemble_' + date)
if not os.path.isdir(outpth):
    os.makedirs(outpth)
print(outpth)
myfiles = os.listdir(pth1)
print(myfiles)
my_dice = []
for ipatient in myfiles:
    #patient_name = str(ipatient) + '.npy'
    patient_name = ipatient
    print("Evaluating patient... ", ipatient)
    im1 = np.load(pth1 + '/' + patient_name)
    im2 = np.load(pth2 + '/' + patient_name)
    im3 = np.load(pth3 + '/' + patient_name)
    im4 = np.load(pth4 + '/' + patient_name)
    im5 = np.load(pth5 + '/' + patient_name)
    im_all = (im1 + im2 + im3 + im4 + im5)/5
    im = np.multiply(im_all > 0.5, 1)


    # Calculate DSC
    # Load gt label
    datafile_gt = os.path.join(pth_gt, patient_name[:-4]+'.nii') #.nii.gz
    gt = nib.load(datafile_gt).get_fdata()
    #gt = np.rot90(gt, 2)

    dice_patient = compute_dice_patient(gt, im)
    print("The dice value of this patient is...", dice_patient)
    my_dice.append(dice_patient)

    ## Save without considering ROI ##
    output_datafile = os.path.join(outpth, patient_name[:-4])
    np.save(output_datafile, im)

print("The average my DSC is...", np.mean(my_dice))
print("The standard deviation of my DSC is...", np.std(my_dice))




