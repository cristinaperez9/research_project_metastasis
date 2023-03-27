
################################################################################
# Cristina Almagro-Perez
################################################################################
# Files need to be in the same folder to apply skull removal
import os
import shutil
pth0 = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/metastases_project/preprocessed_CAP/'
pth_target = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/metastases_project/preprocessed_CAP_all_patients/no_skull_removed/'
files = os.listdir(pth0)
#ifiles = 'NSCLC_nii' #melanomapreopMRI_nii # Breast_nii #Melanoma_nii
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
        print("Copying patient: ", ipatient)
        original_datafile = os.path.join(pth_patient, 'img_reg.nii.gz')
        nm_output = ipatient + '.nii.gz'
        target_datafile = os.path.join(pth_target, nm_output)
        shutil.copyfile(original_datafile, target_datafile)

# Skull removal run directly on command window (really easy).
# See this Github page:  https://github.com/MIC-DKFZ/HD-BET
