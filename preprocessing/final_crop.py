
#######################################################################################
# Cristina Almagro-Pérez, ETH Zürich, 2022
########################################################################################
# Crop images in x, y, and z dimensions after skull stripping to reduce image dimensions
import glob
import nibabel as nib
import numpy as np
import os

def affine_after_resizing(old_affine, original_shape, ref_shape):
    #print("Original shape:", original_shape)
    #print("Old affine", old_affine)
    new_affine = old_affine
    var = np.array(ref_shape) - np.array(original_shape)
    new_affine[0, 3] = old_affine[0, 3] + var[0]/2
    new_affine[1, 3] = old_affine[1, 3] + var[1]/2
    new_affine[2, 3] = old_affine[2, 3] - var[2]/2
    return new_affine

find_tissue_area = False  # Find the x, y and z location smallest and largest occupy by the brains in all patients
crop_images = False
crop_images_to_bb = True
if find_tissue_area:

    pth0 = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/metastases_project/preprocessed_CAP_all_patients/skull_removed/'
    files = glob.glob(pth0 + '*mask.*')
    #files = list(set(glob.glob(pth0 + '*')) - set(glob.glob(pth0 + '*mask.*')))
    x_min_all = []
    y_min_all = []
    z_min_all = []
    x_max_all = []
    y_max_all = []
    z_max_all = []

    for ifiles in files:
        #ifiles = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/metastases_project/preprocessed_CAP_all_patients/skull_removed/000000-1_mask.nii.gz'
        mask = nib.load(ifiles).get_fdata()
        # Find x_min and x_max
        loc = np.sum(np.sum(mask, 2), 1)
        loc = np.nonzero(loc)
        x_min_all.append(np.min(loc))
        x_max_all.append(np.max(loc))

        # Find y_min and y_max
        loc = np.sum(np.sum(mask, 2), 0)
        loc = np.nonzero(loc)
        y_min_all.append(np.min(loc))
        y_max_all.append(np.max(loc))

        # Find z_min and z_max
        loc = np.sum(np.sum(mask, 0), 0)
        loc = np.nonzero(loc)
        z_min_all.append(np.min(loc))
        z_max_all.append(np.max(loc))

    print("x_min is: ", np.min(x_min_all))
    print("x_max is: ", np.max(x_max_all))
    print("y_min is: ", np.min(y_min_all))
    print("y_max is: ", np.max(y_max_all))
    print("z_min is: ", np.min(z_min_all))
    print("z_max is: ", np.max(z_max_all))

if crop_images:
    pth0 = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/metastases_project/preprocessed_CAP/'
    pth1 = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/metastases_project/preprocessed_CAP_all_patients/'
    outpth0 = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/metastases_project/preprocessed_CAP_final/'
    files = os.listdir(pth0)
    for ifiles in files:
        if 'DS_Store' in ifiles:
            continue
        pth = os.path.join(pth0, ifiles)
        print("Cancer type: ", ifiles)
        patient_files = os.listdir(pth)
        for ipatient in patient_files:
            if '.' in ipatient:
                continue
            pth_patient = os.path.join(pth, ipatient)
            print("Saving patient: ", ipatient)

            pth1_skull = os.path.join(pth1, 'no_skull_removed')
            pth1_no_skull = os.path.join(pth1, 'skull_removed')

            # Before skull stripping
            datafile_skull = os.path.join(pth1_skull, ipatient + '.nii.gz')
            im_skull0 = nib.load(datafile_skull)
            im_skull = im_skull0.get_fdata()

            # After skull stripping
            datafile_no_skull = os.path.join(pth1_no_skull, ipatient + '.nii.gz')
            im_no_skull0 = nib.load(datafile_no_skull)
            im_no_skull = im_no_skull0.get_fdata()

            # Mask
            datafile_mask = os.path.join(pth_patient, 'mask_reg.nii.gz')
            mask0 = nib.load(datafile_mask)
            mask = mask0.get_fdata()
            mask[mask == 255] = 1
            if not (np.unique(mask) == [0., 1.]).all:
                raise Warning("Additional values present in segmentation mask")

            # Crop the three images
            im_no_skull = im_no_skull[45:395, 21:371, 35:385]
            im_skull = im_skull[45:395, 21:371, 35:385]
            mask = mask[45:395, 21:371,  35:385]

            # NOTE: The above is what I used, however I believe rows should be changed by columns.

            # Save images in corresponding folder

            # Create output folder
            outpth = os.path.join(outpth0, ifiles)
            if not os.path.isdir(outpth):
                os.makedirs(outpth)
            outpth_patient = os.path.join(outpth, ipatient)
            if not os.path.isdir(outpth_patient):
                os.makedirs(outpth_patient)

            # Adapt header (header the same for the three images)
            empty_header = nib.Nifti1Header()
            new_affine = affine_after_resizing(im_skull0.affine, [400, 400, 400], [350, 350, 350])

            # Save mask
            output_datafile_mask = os.path.join(outpth_patient, 'mask_reg_cropped.nii.gz')
            imout_nifti = nib.Nifti1Image(mask, new_affine, empty_header)  # Header created automatically
            nib.save(imout_nifti, output_datafile_mask)

            # Save cropped image before skull stripping
            output_datafile_skull = os.path.join(outpth_patient, 'img_reg_cropped.nii.gz')
            imout_nifti = nib.Nifti1Image(im_skull, new_affine, empty_header)  # Header created automatically
            nib.save(imout_nifti, output_datafile_skull)

            # Save cropped image before skull stripping
            output_datafile_no_skull = os.path.join(outpth_patient, 'img_reg_cropped_stripped.nii.gz')
            imout_nifti = nib.Nifti1Image(im_no_skull, new_affine, empty_header)  # Header created automatically
            nib.save(imout_nifti, output_datafile_no_skull)

if crop_images_to_bb:
    pth0 = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/metastases_project/preprocessed_CAP/'
    pth1 = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/metastases_project/preprocessed_CAP_all_patients/'
    outpth0 = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/metastases_project/preprocessed_CAP_final/'
    files = os.listdir(pth0)
    # for ifiles in files:
    #     if 'DS_Store' in ifiles:
    #         continue
    ifiles = 'melanomapreopMRI_nii'
    pth = os.path.join(pth0, ifiles)
    print("Cancer type: ", ifiles)
    patient_files = os.listdir(pth)
    for ipatient in patient_files:
        if '.' in ipatient:
            continue
        pth_patient = os.path.join(pth, ipatient)
        print("Saving patient: ", ipatient)

        pth1_skull = os.path.join(pth1, 'no_skull_removed')
        pth1_no_skull = os.path.join(pth1, 'skull_removed')

        # Before skull stripping
        datafile_skull = os.path.join(pth1_skull, ipatient + '.nii.gz')
        im_skull0 = nib.load(datafile_skull)
        im_skull = im_skull0.get_fdata()

        # After skull stripping
        datafile_no_skull = os.path.join(pth1_no_skull, ipatient + '.nii.gz')
        im_no_skull0 = nib.load(datafile_no_skull)
        im_no_skull = im_no_skull0.get_fdata()

        # Mask
        datafile_mask = os.path.join(pth_patient, 'mask_reg.nii.gz')
        mask0 = nib.load(datafile_mask)
        mask = mask0.get_fdata()
        mask[mask == 255] = 1
        if not (np.unique(mask) == [0., 1.]).all:
            raise Warning("Additional values present in segmentation mask")

        # Crop the three images
        im_no_skull = im_no_skull[65-5:327+5, 52-5:389+5, 65-5:354+5]  # leave 5 pixels margin from the bounding box
        im_skull = im_skull[65-5:327+5, 52-5:389+5, 65-5:354+5]  # leave 5 pixels margin from the bounding box
        mask = mask[65-5:327+5, 52-5:389+5, 65-5:354+5]  # leave 5 pixels margin from the bounding box
        # old: [52-5:389+5, 65-5:327+5, 65-5:354+5]
        # import matplotlib.pyplot as plt
        # plt.imshow(im_no_skull[:,:,100])
        # plt.show()


        # Save images in corresponding folder

        # Create output folder
        outpth = os.path.join(outpth0, ifiles)
        if not os.path.isdir(outpth):
            os.makedirs(outpth)
        outpth_patient = os.path.join(outpth, ipatient)
        if not os.path.isdir(outpth_patient):
            os.makedirs(outpth_patient)

        # Adapt header (header the same for the three images)
        empty_header = nib.Nifti1Header()
        new_affine = affine_after_resizing(im_skull0.affine, [400, 400, 400], im_no_skull.shape)

        # Save mask
        output_datafile_mask = os.path.join(outpth_patient, 'mask_reg_cropped_tight.nii.gz')
        imout_nifti = nib.Nifti1Image(mask, new_affine, empty_header)  # Header created automatically
        nib.save(imout_nifti, output_datafile_mask)

        # Save cropped image before skull stripping
        output_datafile_skull = os.path.join(outpth_patient, 'img_reg_cropped_tight.nii.gz')
        imout_nifti = nib.Nifti1Image(im_skull, new_affine, empty_header)  # Header created automatically
        nib.save(imout_nifti, output_datafile_skull)

        # Save cropped image before skull stripping
        output_datafile_no_skull = os.path.join(outpth_patient, 'img_reg_cropped_stripped_tight.nii.gz')
        imout_nifti = nib.Nifti1Image(im_no_skull, new_affine, empty_header)  # Header created automatically
        nib.save(imout_nifti, output_datafile_no_skull)
