## Code for bias correction of MR images: N4 algorithm used ##

#-------------------------------------------------------------------------
# Code written by Cristina Almagro-Pérez,2022, ETH University (Zurich).
#--------------------------------------------------------------------------


# Import necessary packages
import matplotlib.pyplot as plt
import os
import numpy as np
from nibabel.testing import data_path
import nibabel as nib
import SimpleITK as sitk
import sys
import os
from pytictoc import TicToc

import SimpleITK as sitk
t = TicToc()

def myN4(inputImage, pth, output_name, save_mask=True):
    #print("N4 bias correction runs.")
    maskImage = sitk.OtsuThreshold(inputImage, 0, 1, 200)
    # Code to save the mask image
    filename = os.path.join(pth, 'brain_mask_for_bias_correction1.nii.gz')
    if save_mask:
        sitk.WriteImage(maskImage, filename)

    image = sitk.Shrink(inputImage, [4, 4, 4])  #4
    maskImage = sitk.Shrink(maskImage, [4, 4, 4])
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    t.tic()  #Start timer
    #corrector.SetMaximumNumberOfIterations(int(32))  # 4 fitting levels *
    corrected_image = corrector.Execute(image, maskImage)
    t.toc()  #Time elapsed since t.tic()
    log_bias_field = corrector.GetLogBiasFieldAsImage(inputImage)
    corrected_image_full_resolution = inputImage / sitk.Exp(log_bias_field)
    filename = os.path.join(pth, output_name)
    sitk.WriteImage(corrected_image_full_resolution, filename)
    #print("Finished N4 Bias Field Correction.....")
    return corrected_image_full_resolution


#####################################################################################
# SPECIFY THE FOLLOWING:
one_image = False  # Test bias-correction code for one image
visualize_one_image = False  # Visualize example image before and after bias correction
all_images = True  # Performs bias-corrections field for all images
remove_old_files = False
#####################################################################################

# Bias correction for one image
if one_image:
    pth00 = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/metastases_project/preprocessed_CAP/Melanoma_nii/000019-27'
    #filename = os.path.join(pth00, 'img_resized_resample_bias_corrected.nii.gz')
    filename = os.path.join(pth00, 'image_resampled.nii.gz')
    inputImage = sitk.ReadImage(filename, sitk.sitkFloat32)
    pth = pth00
    output_name = 'img_resized_resample_bias_corrected4.nii.gz'
    im1 = myN4(inputImage, pth, output_name, save_mask=False)  #It will write the bias corrected image
    #im_final = myN4(im1, pth, save_mask=False)
    #filename = os.path.join(pth, 'img_resized_resample_bias_corrected3.nii.gz')
    #sitk.WriteImage(im_final, filename)

# 000001-10

if visualize_one_image:
    #Visualize bias corrected images
    pth00 = r'/usr/bmicnas01/data-biwi-01/bmicdatasets-originals/Originals/USZ/metastases_segmentation/Processed/Registered/Breast_nii/000000-11'
    filename = os.path.join(pth00, 'reg_img_cropped.nii.gz')
    pth = r'/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/Breast_nii/000000-11'
    filename_corrected = os.path.join(pth, 'reg_img_cropped_bias_correc.nii.gz')
    img1 = nib.load(filename).get_fdata()
    img1_corrected = nib.load(filename_corrected).get_fdata()
    # Plot images before and after bias field correction
    test = img1[:,:,100]
    plt.savefig("test.png")
    plt.imshow(test,cmap='gray')
    plt.show()
    test_corrected = img1_corrected[:,:,100]
    plt.savefig("test_corrected.png")
    plt.imshow(test_corrected,cmap='gray')
    plt.show()
    import cv2
    original = img1
    duplicate = img1_corrected
    difference = cv2.subtract(original, duplicate)
    test_difference = difference[:,:,100]
    plt.imshow(test_difference,cmap='gray')
    plt.show()

if all_images:
    print("Performing bias correction first time")
    pth00 = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/metastases_project/preprocessed_CAP/'
    outpth00 = pth00
    myfiles = os.listdir(pth00)
    for imyfiles in myfiles:
        if 'DS_Store' in imyfiles:
            continue
        pth0 = os.path.join(pth00, imyfiles)  #4 folders with the 4 types of cancers
        outpth0 = os.path.join(outpth00, imyfiles)
        patients = os.listdir(pth0)
        for ipatients in patients:
            if '.' in ipatients:
                continue
            print(ipatients)
            pth = os.path.join(pth0,ipatients)
            outpth = os.path.join(outpth0, ipatients)
            filename = os.path.join(pth, 'image_resampled.nii.gz')
            DS = pth.find("DS_Store")
            output_datafile = os.path.join(pth, 'img_resampled_bias_corrected1.nii.gz')
            if os.path.isfile(output_datafile):
                continue
            if DS == -1:
                inputImage = sitk.ReadImage(filename, sitk.sitkFloat32)
                myN4(inputImage, outpth, output_name='img_resampled_bias_corrected1.nii.gz', save_mask=True)  # It will write the bias corrected image and the mask brain-background used


if remove_old_files:
    pth00 = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/metastases_project/preprocessed_CAP/'
    outpth00 = pth00
    myfiles = os.listdir(pth00)
    for imyfiles in myfiles:
        if 'DS_Store' in imyfiles:
            continue
        pth0 = os.path.join(pth00, imyfiles)  # 4 folders with the 4 types of cancers
        outpth0 = os.path.join(outpth00, imyfiles)
        patients = os.listdir(pth0)
        for ipatients in patients:
            if '.' in ipatients:
                continue
            print(ipatients)
            pth = os.path.join(pth0, ipatients)
            outpth = os.path.join(outpth0, ipatients)
            filename1 = os.path.join(pth, 'img_resized_resample_bias_corrected.nii.gz')
            filename2 = os.path.join(pth, 'brain_mask_for_bias_correction.nii.gz')
            if os.path.isfile(filename1):
                os.remove(filename1)
            if os.path.isfile(filename2):
                os.remove(filename2)
            print("Removing: ", filename1)
            print("Removing: ", filename2)
