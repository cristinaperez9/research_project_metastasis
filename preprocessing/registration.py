# Co-register MR volumes with ANTS library #
print("hola")
# ######################################################################################
# # Cristina Almagro-Pérez, November 2022, ETH Zürich
# ######################################################################################
#
import ants
import matplotlib.pyplot as plt
import imshowpair
from utils import blend
import os
import nibabel as nib
import numpy as np
import random
import shutil
#
registration_images = False
registration_masks = False
register_both = False
copy_output_images = True
# ################################################################################################
# # Registration images
# ################################################################################################
if registration_images:
    # Fix first I tried:
    # Define fix image 000006-19, 000001-12(use this from breast) # prueba Melanoma_nii/000019-27
    datafile_fix = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/metastases_project/preprocessed_CAP/Breast_nii/000001-12/img_resampled_bias_corrected1_resized3.nii.gz'
    fix = ants.image_read(datafile_fix)
    print("Fix image loaded")
    fix_np = fix.numpy()
    fix_coronal = ants.from_numpy(np.rot90(fix_np[:, 200, :], 2))
    fix_sagittal = ants.from_numpy(np.rot90(fix_np[200, :, :], 2))
    fix_axial = ants.from_numpy(fix_np[:, :, 200])

    pth0 = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/metastases_project/preprocessed_CAP/'
    files = os.listdir(pth0)
    ifiles = 'melanomapreopMRI_nii'
    # for ifiles in files:
    #     if 'DS_Store' in ifiles:
    #         continue
    pth = os.path.join(pth0, ifiles)
    print("Cancer type: ", ifiles)
    patient_files = os.listdir(pth)
    for ipatient in patient_files:
        if '.' in ipatient:
            continue
        pth_patient = os.path.join(pth, ipatient)
        print("Registering patient: ", ipatient)
        datafile_mov = os.path.join(pth_patient, 'img_resampled_bias_corrected1_resized3.nii.gz')

        #datafile_mov = os.path.join(pth_patient,'img_resized_resample_bias_corrected2.nii.gz')
        #datafile_mov = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/metastases_project/preprocessed_CAP/Breast_nii/000011-24/img_resized_resample_bias_corrected1.nii.gz'

        print("Datafile mov is ...", datafile_mov)
        mov = ants.image_read(datafile_mov)
        print("Image loaded")
        mytx = ants.registration(fix, mov, 'Rigid')
        #print(mytx)
        reg_mov = ants.apply_transforms(fixed=fix,
                                        moving=mov,
                                        transformlist=mytx['fwdtransforms'])

        # Convert variables to numpy
        mov_np = mov.numpy()
        reg_mov_np = reg_mov.numpy()

        # Normalize between 0 and 1
        #reg_mov_np = (reg_mov_np - np.min(reg_mov_np)) / (np.max(reg_mov_np) - np.min(reg_mov_np))


        # Change padding - find minimum value different from 0
        # voxel_values = np.unique(reg_mov_np)
        # voxel_values = voxel_values[voxel_values > 0]
        # min_value = np.min(voxel_values)
        # reg_mov_np[reg_mov_np == 0] = min_value

        if not reg_mov_np.shape == (400, 400, 400):
            raise Warning("Output image does not have the desired dimensions")


        # Extract coronal, sagittal and axial slices for plotting
        mov_coronal = ants.from_numpy(np.rot90(mov_np[:, 200, :], 2))  #rotate 180
        reg_mov_coronal = ants.from_numpy(np.rot90(reg_mov_np[:, 200, :], 2))

        mov_sagittal = ants.from_numpy(np.rot90(mov_np[200, :, :], 2))
        reg_mov_sagittal = ants.from_numpy(np.rot90(reg_mov_np[200, :, :], 2))

        mov_axial = ants.from_numpy(mov_np[:, :, 200])
        reg_mov_axial = ants.from_numpy(reg_mov_np[:, :, 200])


        # Create image folder and save coronal, sagittal and axial slices before and after
        outpth_images = os.path.join(pth_patient, 'visualize_registration')
        if not os.path.isdir(outpth_images):
            os.makedirs(outpth_images)
         #ORIENTATION THE IMAGES
        # Coronal
        datafile_coronal = os.path.join(outpth_images, 'coronal_before_reg.jpg')
        datafile_coronal_reg = os.path.join(outpth_images, 'coronal_reg.jpg')
        ants.plot(fix_coronal, mov_coronal, overlay_alpha=0.5, overlay_cmap='BrBG',
                  title='Before registration', filename=datafile_coronal)
        ants.plot(fix_coronal, reg_mov_coronal, overlay_alpha=0.5, overlay_cmap='BrBG',
                  title='After registration', filename=datafile_coronal_reg)

        # Axial
        datafile_axial = os.path.join(outpth_images, 'axial_before_reg.jpg')
        datafile_axial_reg = os.path.join(outpth_images, 'axial_reg.jpg')
        ants.plot(fix_axial, mov_axial, overlay_alpha=0.5, overlay_cmap='BrBG',
                  title='Before registration', filename=datafile_axial)
        ants.plot(fix_axial, reg_mov_axial, overlay_alpha=0.5, overlay_cmap='BrBG',
                  title='After registration', filename=datafile_axial_reg)

        # Sagittal
        datafile_sagittal = os.path.join(outpth_images, 'sagittal_before_reg.jpg')
        datafile_sagittal_reg = os.path.join(outpth_images, 'sagittal_reg.jpg')
        ants.plot(fix_sagittal, mov_sagittal, overlay_alpha=0.5, overlay_cmap='BrBG',
                  title='Before registration', filename=datafile_sagittal)
        ants.plot(fix_sagittal, reg_mov_sagittal, overlay_alpha=0.5, overlay_cmap='BrBG',
                  title='After registration', filename=datafile_sagittal_reg)


        # Save registered image in nifti file
        empty_header = nib.Nifti1Header()
        im0_mov = nib.load(datafile_mov)
        output_datafile = os.path.join(pth_patient, 'img_reg.nii.gz')
        imout_nifti = nib.Nifti1Image(reg_mov_np, im0_mov.affine, im0_mov.header)
        nib.save(imout_nifti, output_datafile)

if registration_masks:
    create_plots = True
    # Fix first I tried:
    # Define fix image 000006-19, 000001-12(use this from breast) # prueba Melanoma_nii/000019-27
    datafile_fix = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/metastases_project/preprocessed_CAP/Breast_nii/000001-12/img_resampled_bias_corrected1_resized3.nii.gz'
    fix = ants.image_read(datafile_fix)

    pth0 = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/metastases_project/preprocessed_CAP/'
    files = os.listdir(pth0)
    ifiles = 'melanomapreopMRI_nii'
    # for ifiles in files:
    #     if 'DS_Store' in ifiles:
    #         continue
    pth = os.path.join(pth0, ifiles)
    print("Cancer type: ", ifiles)
    patient_files = os.listdir(pth)
    for ipatient in patient_files:
        if '.' in ipatient:
            continue
        pth_patient = os.path.join(pth, ipatient)
        print("Registering patient: ", ipatient)
        datafile_mov = os.path.join(pth_patient, 'img_resampled_bias_corrected1_resized3.nii.gz')

        print("Datafile mov is ...", datafile_mov)
        mov = ants.image_read(datafile_mov)
        print("Image loaded")
        mytx = ants.registration(fix, mov, 'Rigid')

        # Load mask
        datafile_mask = os.path.join(pth_patient, 'mask_resized.nii.gz')
        mask = ants.image_read(datafile_mask)

        # Apply transformation to mask
        reg_mask = ants.apply_transforms(fixed=fix,
                                         moving=mask,
                                         transformlist=mytx['fwdtransforms'],
                                         interpolator='nearestNeighbor')
        reg_mov = ants.apply_transforms(fixed=fix,
                                        moving=mov,
                                        transformlist=mytx['fwdtransforms'])
        reg_mov_np = reg_mov.numpy()
        # Convert variables to numpy
        reg_mask_np = reg_mask.numpy()

        if not reg_mask_np.shape == (400, 400, 400):
            raise Warning("Output image does not have the desired dimensions")
        if not (np.unique(reg_mask_np) == [0., 255.]).all:
            print("hola")
            raise Warning("Additional values present in segmentation mask")

        # Load registered image
        datafile_reg_image = os.path.join(pth_patient, 'img_reg.nii.gz')
        reg_img0 = nib.load(datafile_reg_image)
        reg_img = reg_img0.get_fdata()

        # Find two slices with metastases
        loc = np.sum(np.sum(reg_mask_np, 0), 0)
        loc = np.nonzero(loc)
        rand_ind = np.random.choice(np.squeeze(loc), 2)

        outpth_images = os.path.join(pth_patient, 'visualize_registration')
        # Plot first slice together with mask
        rand_ind[0] = 257
        slice_mask_1 = ants.from_numpy(reg_mask_np[:, :, rand_ind[0]])
        reg_mov_1 = ants.from_numpy(reg_mov[:, :, rand_ind[0]])
        datafile_1 = os.path.join(outpth_images, 'registered_mask_slice_1_new.jpg')
        ants.plot(reg_mov_1, slice_mask_1, overlay_alpha=0.5, overlay_cmap='BrBG',
                  title='Slice ' + str(rand_ind[0]) + 'after registration', filename=datafile_1)

        # Plot second slice together with mask
        rand_ind[1] = 275
        slice_mask_2 = ants.from_numpy(reg_mask_np[:, :, rand_ind[1]])
        reg_mov_2 = ants.from_numpy(reg_mov[:, :, rand_ind[1]])
        datafile_2 = os.path.join(outpth_images, 'registered_mask_slice_2_new.jpg')
        ants.plot(reg_mov_2, slice_mask_2, overlay_alpha=0.5, overlay_cmap='BrBG',
                  title='Slice ' + str(rand_ind[1]) + 'after registration', filename=datafile_2)

        # Save registered mask in nifti file
        output_datafile = os.path.join(pth_patient, 'mask_reg.nii.gz')
        imout_nifti = nib.Nifti1Image(reg_mask_np, reg_img0.affine, reg_img0.header)
        nib.save(imout_nifti, output_datafile)

if register_both:
    datafile_fix = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/metastases_project/preprocessed_CAP/Breast_nii/000001-12/img_resampled_bias_corrected1_resized3.nii.gz'
    fix = ants.image_read(datafile_fix)
    print("Fix image loaded")
    fix_np = fix.numpy()
    fix_coronal = ants.from_numpy(np.rot90(fix_np[:, 200, :], 2))
    fix_sagittal = ants.from_numpy(np.rot90(fix_np[200, :, :], 2))
    fix_axial = ants.from_numpy(fix_np[:, :, 200])

    pth0 = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/metastases_project/preprocessed_CAP/'
    files = os.listdir(pth0)
    ifiles = 'NSCLC_nii' #melanomapreopMRI_nii # Breast_nii #Melanoma_nii
    # for ifiles in files:
    #     if 'DS_Store' in ifiles:
    #         continue
    pth = os.path.join(pth0, ifiles)
    print("Cancer type: ", ifiles)
    patient_files = os.listdir(pth)
    for ipatient in patient_files:
        if '.' in ipatient:
            continue
        pth_patient = os.path.join(pth, ipatient)
        print("Registering patient: ", ipatient)
        datafile_mov = os.path.join(pth_patient, 'img_resampled_bias_corrected1_resized3.nii.gz')

        print("Datafile mov is ...", datafile_mov)
        mov = ants.image_read(datafile_mov)
        print("Image loaded")
        mytx = ants.registration(fix, mov, 'Rigid')

        # Register image
        reg_mov = ants.apply_transforms(fixed=fix,
                                        moving=mov,
                                        transformlist=mytx['fwdtransforms'])
        # Load and register mask
        datafile_mask = os.path.join(pth_patient, 'mask_resized1.nii.gz')
        mask = ants.image_read(datafile_mask)
        reg_mask = ants.apply_transforms(fixed=fix,
                                         moving=mask,
                                         transformlist=mytx['fwdtransforms'],
                                         interpolator='nearestNeighbor')

        # Convert variables to numpy
        mov_np = mov.numpy()
        reg_mov_np = reg_mov.numpy()
        reg_mask_np = reg_mask.numpy()

       # Perform checks
        if not reg_mov_np.shape == (400, 400, 400):
            raise Warning("Output image does not have the desired dimensions")
        if not reg_mask_np.shape == (400, 400, 400):
            raise Warning("Output image does not have the desired dimensions")
        if not (np.unique(reg_mask_np) == [0., 255.]).all:
            print("hola")
            raise Warning("Additional values present in segmentation mask")

        # Extract coronal, sagittal and axial slices for plotting
        mov_coronal = ants.from_numpy(np.rot90(mov_np[:, 200, :], 2))  # rotate 180
        reg_mov_coronal = ants.from_numpy(np.rot90(reg_mov_np[:, 200, :], 2))

        mov_sagittal = ants.from_numpy(np.rot90(mov_np[200, :, :], 2))
        reg_mov_sagittal = ants.from_numpy(np.rot90(reg_mov_np[200, :, :], 2))

        mov_axial = ants.from_numpy(mov_np[:, :, 200])
        reg_mov_axial = ants.from_numpy(reg_mov_np[:, :, 200])

        # Create image folder and save coronal, sagittal and axial slices before and after
        outpth_images = os.path.join(pth_patient, 'visualize_registration')
        if not os.path.isdir(outpth_images):
            os.makedirs(outpth_images)

        # Coronal
        datafile_coronal = os.path.join(outpth_images, 'coronal_before_reg.jpg')
        datafile_coronal_reg = os.path.join(outpth_images, 'coronal_reg.jpg')
        ants.plot(fix_coronal, mov_coronal, overlay_alpha=0.5, overlay_cmap='BrBG',
                  title='Before registration', filename=datafile_coronal)
        ants.plot(fix_coronal, reg_mov_coronal, overlay_alpha=0.5, overlay_cmap='BrBG',
                  title='After registration', filename=datafile_coronal_reg)

        # Axial
        datafile_axial = os.path.join(outpth_images, 'axial_before_reg.jpg')
        datafile_axial_reg = os.path.join(outpth_images, 'axial_reg.jpg')
        ants.plot(fix_axial, mov_axial, overlay_alpha=0.5, overlay_cmap='BrBG',
                  title='Before registration', filename=datafile_axial)
        ants.plot(fix_axial, reg_mov_axial, overlay_alpha=0.5, overlay_cmap='BrBG',
                  title='After registration', filename=datafile_axial_reg)

        # Sagittal
        datafile_sagittal = os.path.join(outpth_images, 'sagittal_before_reg.jpg')
        datafile_sagittal_reg = os.path.join(outpth_images, 'sagittal_reg.jpg')
        ants.plot(fix_sagittal, mov_sagittal, overlay_alpha=0.5, overlay_cmap='BrBG',
                  title='Before registration', filename=datafile_sagittal)
        ants.plot(fix_sagittal, reg_mov_sagittal, overlay_alpha=0.5, overlay_cmap='BrBG',
                  title='After registration', filename=datafile_sagittal_reg)


        #### Plots for masks ###
        # Find two slices with metastases
        loc = np.sum(np.sum(reg_mask_np, 0), 0)
        loc = np.nonzero(loc)
        rand_ind = np.random.choice(np.squeeze(loc), 2)

        # Plot first slice together with mask
        slice_mask_1 = ants.from_numpy(reg_mask_np[:, :, rand_ind[0]])
        reg_mov_1 = ants.from_numpy(reg_mov_np[:, :, rand_ind[0]])
        datafile_1 = os.path.join(outpth_images, 'registered_mask_slice_1.jpg')
        ants.plot(reg_mov_1, slice_mask_1, overlay_alpha=0.5, overlay_cmap='BrBG',
                  title='Slice ' + str(rand_ind[0]) + 'after registration', filename=datafile_1)

        # Plot second slice together with mask
        slice_mask_2 = ants.from_numpy(reg_mask_np[:, :, rand_ind[1]])
        reg_mov_2 = ants.from_numpy(reg_mov_np[:, :, rand_ind[1]])
        datafile_2 = os.path.join(outpth_images, 'registered_mask_slice_2.jpg')
        ants.plot(reg_mov_2, slice_mask_2, overlay_alpha=0.5, overlay_cmap='BrBG',
                  title='Slice ' + str(rand_ind[1]) + 'after registration', filename=datafile_2)

        # Save registered image in nifti file
        im0_mov = nib.load(datafile_mov)
        output_datafile = os.path.join(pth_patient, 'img_reg.nii.gz')
        imout_nifti = nib.Nifti1Image(reg_mov_np, im0_mov.affine, im0_mov.header)
        nib.save(imout_nifti, output_datafile)

        # Save registered mask in nifti file
        output_datafile = os.path.join(pth_patient, 'mask_reg.nii.gz')
        imout_nifti = nib.Nifti1Image(reg_mask_np, im0_mov.affine, im0_mov.header)
        nib.save(imout_nifti, output_datafile)
#
if copy_output_images:
    pth0 = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/metastases_project/preprocessed_CAP/'
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
            print("Copying images of patient: ", ipatient)
            pth = os.path.join(pth0, ifiles)
            pth_patient = os.path.join(pth, ipatient)
            src = os.path.join(pth_patient, 'visualize_registration')

            outpth = os.path.join(outpth0, ifiles)
            outpth_patient = os.path.join(outpth, ipatient)
            dst = os.path.join(outpth_patient, 'visualize_registration')
            shutil.copytree(src, dst)
            print("hola")
