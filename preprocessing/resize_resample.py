
########################################################################################
# Cristina Almagro-Pérez, November 2022, ETH Zürich
########################################################################################
# Import necessary packages
from scipy import ndimage, misc
import numpy as np
import nibabel as nib
import os
from utils import rescale_affine
import matplotlib.pyplot as plt


step_1 = True  # Match voxel size (resample)
step_2 = False # Crop or pad if necessary to make the image sizes the same.
step_3 = False # Match voxel size and resize for masks (step_1 and step_2 together for masks)
step_4 = False # change headers (correct error)

########################################################################################
# Auxiliary functions
########################################################################################
def match_voxel_size(img_datafile, ref_voxel_size):
    im0 = nib.load(img_datafile)
    im = im0.get_fdata()
    print("Original image size: ", im.shape)
    pixel_size = im0.header['pixdim'][1:4]
    factor = np.divide(pixel_size, ref_voxel_size)
    print("Original pixel size: ", pixel_size)
    imout = ndimage.zoom(im, factor)
    empty_header = nib.Nifti1Header()
    new_affine = rescale_affine(im0.affine, ref_voxel_size)
    imout_nifti = nib.Nifti1Image(imout, new_affine, empty_header)  # Header created automatically
    print("Resampled image size: ", imout.shape)
    return imout_nifti

def affine_after_resizing(old_affine, original_shape, ref_shape):
    print("Original shape:", original_shape)
    #print("Old affine", old_affine)
    new_affine = old_affine
    var = np.array(ref_shape) - np.array(original_shape)
    new_affine[0, 3] = old_affine[0, 3] + var[0]/2
    new_affine[1, 3] = old_affine[1, 3] + var[1]/2
    new_affine[2, 3] = old_affine[2, 3] - var[2]/2
    return new_affine


########################################################################################
# Match voxel size
########################################################################################
if step_1:
    ref_voxel_size = [0.6, 0.6, 0.6]
    pth0 = '/usr/bmicnas01/data-biwi-01/bmicdatasets-originals/Originals/USZ/metastases_segmentation/Processed/Processed/'
    outpth0 ='/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/metastases_project/preprocessed_CAP/'
    files = os.listdir(pth0)
    for ifiles in files:
        if 'DS_Store' in ifiles:
            continue
        pth = os.path.join(pth0, ifiles)
        patient_files = os.listdir(pth)
        for ipatient in patient_files:
            if '.' in ipatient:
                continue
            print("Resampling patient: ", ipatient)
            pth_patient = os.path.join(pth, ipatient)
            # Define output datafile
            outpth = os.path.join(outpth0, ifiles)
            if not os.path.isdir(outpth):
                os.makedirs(outpth)
            outpth_patient = os.path.join(outpth, ipatient)
            if not os.path.isdir(outpth_patient):
                os.makedirs(outpth_patient)
            output_datafile = os.path.join(outpth_patient, 'image_resampled.nii.gz')
            if os.path.isfile(output_datafile):  # skip process if the image has already been saved
                continue
            datafile = os.path.join(pth_patient, 'image.nii.gz')  # Original image
            if not os.path.isfile(datafile):
                datafile = os.path.join(pth_patient, 'image.nii')  # Original image
            im_resampled_nifti = match_voxel_size(datafile, ref_voxel_size)
            nib.save(im_resampled_nifti, output_datafile)


########################################################################################
# Images same size
########################################################################################
check_sizes = False
load_sizes = False
resize_images = True
count = 0
sizes = []
if step_2:
    if check_sizes:
        outpth0 = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/metastases_project/preprocessed_CAP/'
        files = os.listdir(outpth0)
        for ifiles in files:
            if 'DS_Store' in ifiles:
                continue
            pth = os.path.join(outpth0, ifiles)
            patient_files = os.listdir(pth)
            for ipatient in patient_files:
                if '.' in ipatient:
                    continue
                print("Loading patient: ", ipatient)
                outpth = os.path.join(outpth0, ifiles)
                outpth_patient = os.path.join(outpth, ipatient)
                output_datafile = os.path.join(outpth_patient, 'image_resampled.nii.gz')
                im = nib.load(output_datafile).get_fdata()
                count = count + 1
                sizes.append(im.shape)
        output_datafile = '/scratch_net/biwidl311/Cristina_Almagro/big/resampled_image_sizes.npy'
        np.save(output_datafile, sizes)
    if load_sizes:
        sizes = np.load('/scratch_net/biwidl311/Cristina_Almagro/big/resampled_image_sizes.npy')
        print("Printing min values")
        print("Min x value", min(sizes[:, 0]))
        print("Min y value", min(sizes[:, 1]))
        print("Printing max values")
        print("Max x value", max(sizes[:, 0]))
        print("Max y value", max(sizes[:, 1]))
        print(np.unique(sizes[:, 2]))
        y = sizes[:, 2]
        plt.hist(y, bins=len(np.unique(y)))
        plt.ylabel('Number of MR volumes (Frequency)', fontsize=15)
        plt.show()
        values, counts = np.unique(y, return_counts=True)
        print("Printing values")
        print(values)
        print("Printing counts")
        print(counts)
        ind = np.argmax(counts)
        print(values[ind])  # prints the most frequent element

    if resize_images:
        outpth0 = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/metastases_project/preprocessed_CAP/'
        files = os.listdir(outpth0)
        for ifiles in files:
            if 'DS_Store' in ifiles:
                continue
            pth = os.path.join(outpth0, ifiles)
            patient_files = os.listdir(pth)
            for ipatient in patient_files:
                if '.' in ipatient:
                    continue
                print("Resizing patient: ", ipatient)
                outpth = os.path.join(outpth0, ifiles)
                outpth_patient = os.path.join(outpth, ipatient)
                input_datafile = os.path.join(outpth_patient, 'img_resampled_bias_corrected1.nii.gz')  #image_resampled
                im0 = nib.load(input_datafile)
                im = im0.get_fdata()
                im_size = im.shape
                ref_size = (400, 400, 400)

                # Padding (if needed)
                pad_all = np.array(ref_size) - np.array(im_size)
                #print("Printing pad_all", pad_all)
                pad_width = [[0, 0], [0, 0], [0, 0]]
                for kk in range(0, 3):
                    val = pad_all[kk]
                    if val >= 0:
                        if (val % 2) == 0:
                            before = val/2
                            after = before
                        else:
                            before = (val-1) / 2
                            after = before + 1
                        if not int(after+before) == int(val):
                            print(after+before)
                            print(val)
                            raise Warning("Padding is not correct")
                        pad_width[kk] = (int(before), int(after))
                pad_width = tuple(pad_width)
                #print("Printing pad_width", pad_width)
                # Calculate padding value as the average of the 4 corners of the central slice
                #pad_value = (im[0, 0, 200] + im[0, -1, 200] + im[-1, 0, 200] + im[-1, -1, 200])/4
                pad_value = np.min(im)
                print("Pad value is: ", pad_value)
                im_resized = np.pad(im, pad_width, constant_values=(pad_value, pad_value))
                im_resized = (im_resized - np.min(im_resized)) / (np.max(im_resized) - np.min(im_resized))
                current_size = im_resized.shape

                # Cropping (if needed)
                crop_all = np.array(im_resized.shape) - np.array(ref_size)
                fc = [[0, 0], [0, 0], [0, 0]] # final crop
                for kk in range(0, 3):
                    val = crop_all[kk]
                    if val > 0:
                        if (val % 2) == 0:
                            before = val / 2
                            after = before
                        else:
                            before = (val - 1) / 2
                            after = before + 1
                        if not int(after + before) == int(val):
                            print(after + before)
                            print(val)
                            raise Warning("Padding is not correct")
                        fc[kk] = [int(before), int(after)]
                im_resized = im_resized[fc[0][0]:current_size[0]-fc[0][1], fc[1][0]:current_size[1]-fc[1][1], fc[2][0]:current_size[2]-fc[2][1]]
                if not im_resized.shape == (400, 400, 400):
                    raise Warning("Not correct size")

                # Update headers and save Nifti Files
                #output_datafile = os.path.join(outpth_patient, 'img_resampled_bias_corrected1_resized2.nii.gz')
                # empty_header = nib.Nifti1Header()
                # new_affine = affine_after_resizing(im0.affine, im.shape, [400, 400, 400])
                # imout_nifti = nib.Nifti1Image(im_resized, new_affine, empty_header)  # Header created automatically
                #nib.save(imout_nifti, output_datafile)

                datafile1 = os.path.join(outpth_patient, 'img_resized_resample_bias_corrected2.nii.gz')
                im1 = nib.load(datafile1)


                # Save image with header changed
                output_datafile = os.path.join(outpth_patient, 'img_resampled_bias_corrected1_resized3.nii.gz')
                imout_nifti = nib.Nifti1Image(im_resized, im1.affine, im1.header)
                nib.save(imout_nifti, output_datafile)



##########################################################################################
# Resample and resize masks
##########################################################################################
if step_3:
    print("Resample and resize masks")
    ref_voxel_size = [0.6, 0.6, 0.6]
    pth0 = '/usr/bmicnas01/data-biwi-01/bmicdatasets-originals/Originals/USZ/metastases_segmentation/Processed/Processed/'
    outpth0 ='/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/metastases_project/preprocessed_CAP/'
    files = os.listdir(pth0)
    for ifiles in files:
        if 'DS_Store' in ifiles:
            continue
        pth = os.path.join(pth0, ifiles)
        patient_files = os.listdir(pth)
        for ipatient in patient_files:
            if '.' in ipatient:
                continue
            print("Resampling patient: ", ipatient)
            pth_patient = os.path.join(pth, ipatient)
            # Define output datafile
            outpth = os.path.join(outpth0, ifiles)
            if not os.path.isdir(outpth):
                os.makedirs(outpth)
            outpth_patient = os.path.join(outpth, ipatient)
            if not os.path.isdir(outpth_patient):
                os.makedirs(outpth_patient)
            output_datafile = os.path.join(outpth_patient, 'mask_resized.nii.gz')
            if os.path.isfile(output_datafile):  # skip process if the image has already been saved
                continue
            datafile = os.path.join(pth_patient, 'seg_mask_fullsize.nii.gz')  # Original image

            if not os.path.isfile(datafile):
                datafile = os.path.join(pth_patient, 'seg_mask_fullsize.nii')  # Original image

            #### Match voxel size ### - Step 1
            datafile_mask = datafile
            datafile_img = os.path.join(pth_patient, 'image.nii.gz')
            if not os.path.isfile(datafile_img):
                datafile_img = os.path.join(pth_patient, 'image.nii')

            mask = nib.load(datafile_mask).get_fdata()
            print("Mask values:", np.unique(mask))
            im0 = nib.load(datafile_img)
            print("Original mask size: ", mask.shape)
            pixel_size = im0.header['pixdim'][1:4]
            factor = np.divide(pixel_size, ref_voxel_size)
            print("Original pixel size: ", pixel_size)
            mask_out = ndimage.zoom(mask, factor, order=0)
            print(np.unique(mask_out))
            empty_header = nib.Nifti1Header()
            new_affine = rescale_affine(im0.affine, ref_voxel_size)

            ### Resize mask ### -Step 2
            im_size = mask_out.shape
            ref_size = (400, 400, 400)

            # Padding (if needed)
            pad_all = np.array(ref_size) - np.array(im_size)
            # print("Printing pad_all", pad_all)
            pad_width = [[0, 0], [0, 0], [0, 0]]
            for kk in range(0, 3):
                val = pad_all[kk]
                if val >= 0:
                    if (val % 2) == 0:
                        before = val / 2
                        after = before
                    else:
                        before = (val - 1) / 2
                        after = before + 1
                    if not int(after + before) == int(val):
                        print(after + before)
                        print(val)
                        raise Warning("Padding is not correct")
                    pad_width[kk] = (int(before), int(after))
            pad_width = tuple(pad_width)
            # print("Printing pad_width", pad_width)
            mask_resized = np.pad(mask_out, pad_width, constant_values=((0, 0)))
            current_size = mask_resized.shape

            # Cropping (if needed)
            crop_all = np.array(mask_resized.shape) - np.array(ref_size)
            fc = [[0, 0], [0, 0], [0, 0]]  # final crop
            for kk in range(0, 3):
                val = crop_all[kk]
                if val > 0:
                    if (val % 2) == 0:
                        before = val / 2
                        after = before
                    else:
                        before = (val - 1) / 2
                        after = before + 1
                    if not int(after + before) == int(val):
                        print(after + before)
                        print(val)
                        raise Warning("Padding is not correct")
                    fc[kk] = [int(before), int(after)]
            mask_resized = mask_resized[fc[0][0]:current_size[0] - fc[0][1], fc[1][0]:current_size[1] - fc[1][1],
                         fc[2][0]:current_size[2] - fc[2][1]]
            if not mask_resized.shape == (400, 400, 400):
                raise Warning("Not correct size")

            # Update headers and save Nifti Files
            datafile_image_resized = os.path.join(outpth_patient, 'image_resized.nii.gz')
            im_ref = nib.load(datafile_image_resized)
            imout_nifti = nib.Nifti1Image(mask_resized, im_ref.affine, im_ref.header)  # Header created automatically
            nib.save(imout_nifti, output_datafile)





            # imout_nifti = nib.Nifti1Image(imout, new_affine, empty_header)  # Header created automatically
            # print("Resampled image size: ", imout.shape)
            #
            #
            # im_resampled_nifti = match_voxel_size(datafile, ref_voxel_size)
            #nib.save(im_resampled_nifti, output_datafile)



########################################################################################
# Match voxel size one image
########################################################################################

# if step_1:
#     ref_voxel_size = [0.6, 0.6, 0.6]
#     # Image
#     datafile = '/usr/bmicnas01/data-biwi-01/bmicdatasets-originals/Originals/USZ/metastases_segmentation/Processed/Processed/Breast_nii/000002-13/image.nii.gz'
#     im0 = nib.load(datafile)
#     im = im0.get_fdata()
#     print("Original image size: ", im.shape)
#     pixel_size = im0.header['pixdim'][1:4]
#     print(pixel_size)
#     factor = np.divide(pixel_size, ref_voxel_size)
#     print(factor)
#     imout = ndimage.zoom(im, factor)
#
#     # New nifti file
#     empty_header = nib.Nifti1Header()
#     new_affine = rescale_affine(im0.affine, ref_voxel_size)
#     another_img = nib.Nifti1Image(imout, new_affine, empty_header)  # Header created automatically

if step_4:
    print("Changing headers")
    pth0 = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/metastases_project/preprocessed_CAP/'
    files = os.listdir(pth0)
    for ifiles in files:
        if 'DS_Store' in ifiles:
            continue
        pth = os.path.join(pth0, ifiles)
        patient_files = os.listdir(pth)
        for ipatient in patient_files:
            if '.' in ipatient:
                continue
            print("Correcting patient: ", ipatient)
            pth_patient = os.path.join(pth, ipatient)
            datafile1 = os.path.join(pth_patient, 'img_resampled_bias_corrected1_resized3.nii.gz')  #img_resized_resample_bias_corrected2.nii.gz
            im1 = nib.load(datafile1)
            datafile2 = os.path.join(pth_patient, 'mask_resized.nii.gz') #img_resampled_bias_corrected1_resized.nii.gz
            im2 = nib.load(datafile2).get_fdata()

            # Save image with header changed
            output_datafile = os.path.join(pth_patient, 'mask_resized1.nii.gz') #img_resampled_bias_corrected1_resized1.nii.gz
            imout_nifti = nib.Nifti1Image(im2, im1.affine, im1.header)
            nib.save(imout_nifti, output_datafile)







