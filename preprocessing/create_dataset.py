
# Create .json files to test the preprocessing is correct


import json
json_test = False
json_train = False
json_test_new_dataset = False
json_train_new_dataset = True

import os
###############################################################################################
# Create .json files for test set
###############################################################################################
if json_test:

    # Load .json
    datafile_test = "/scratch_net/biwidl311/Cristina_Almagro/research_project/dataset/Cristina_dataset_test_skull_removed.json"
    f = open(datafile_test)
    data = json.load(f)
    print(data)

    pth0 = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/metastases_project/preprocessed_CAP_final/'
    files = os.listdir(pth0)
    old_test = data['testing']
    new_test = []
    for ipatient in old_test:
        patient_name = ipatient['image'].split('/')[-1][:-7]

        # find cancer type
        count = 0
        found = False
        while not found:
            ifiles = files[count]
            count = count + 1

            if 'DS_Store' in ifiles:
                continue
            if 'Preprocessing' in ifiles:
                continue
            pth = os.path.join(pth0, ifiles)
            patient_files = os.listdir(pth)
            count2 = 0
            while not found and count2 < len(patient_files):
                ipatient = patient_files[count2]

                if patient_name == ipatient:
                    found = True
                    cancer_type = ifiles
                count2 = count2 + 1
        if ipatient != '000012-20':
            print("Cancer type: ", cancer_type)
            dict_patient = {}
            output_pth_patient = os.path.join(pth0, cancer_type)
            output_pth_patient = os.path.join(output_pth_patient, ipatient)
            dict_patient['image'] = os.path.join(output_pth_patient, 'img_reg_cropped_stripped.nii.gz')
            dict_patient['label'] = os.path.join(output_pth_patient, 'mask_reg_cropped.nii.gz')
            new_test.append(dict_patient)
        else:
            print('patient found')

    data['testing'] = new_test
    print(new_test)
    print(len(new_test))
    #Save .json file
    datafile_test = "/scratch_net/biwidl311/Cristina_Almagro/research_project/dataset/Cristina_dataset_test_preprocessed1.json"
    json.dump(data, open(datafile_test, 'w'))


###############################################################################################
# Create .json files for training and validtion set
###############################################################################################
if json_train:

    # Load .json
    datafile_test = "/scratch_net/biwidl311/Cristina_Almagro/research_project/dataset/Cristina_dataset_skull_removed.json"
    f = open(datafile_test)
    data = json.load(f)
    #print(data)

    pth0 = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/metastases_project/preprocessed_CAP_final/'
    files = os.listdir(pth0)
    old_train = data['training']
    new_train = []
    for ipatient in old_train:
        patient_name = ipatient['image'].split('/')[-1][:-7]

        # find cancer type
        count = 0
        found = False
        while not found:
            ifiles = files[count]
            count = count + 1

            if 'DS_Store' in ifiles:
                continue
            pth = os.path.join(pth0, ifiles)
            patient_files = os.listdir(pth)
            count2 = 0
            while not found and count2 < len(patient_files):
                ipatient = patient_files[count2]
                if patient_name == ipatient:
                    found = True
                    cancer_type = ifiles
                count2 = count2 + 1
        print("Cancer type: ", cancer_type)
        dict_patient = {}
        output_pth_patient = os.path.join(pth0, cancer_type)
        output_pth_patient = os.path.join(output_pth_patient, ipatient)
        dict_patient['image'] = os.path.join(output_pth_patient, 'img_reg_cropped_stripped.nii.gz')
        dict_patient['label'] = os.path.join(output_pth_patient, 'mask_reg_cropped.nii.gz')
        new_train.append(dict_patient)
    data['training'] = new_train

    old_val = data['validation']
    new_val = []
    for ipatient in old_val:
        patient_name = ipatient['image'].split('/')[-1][:-7]

        # find cancer type
        count = 0
        found = False
        while not found:
            ifiles = files[count]
            count = count + 1

            if 'DS_Store' in ifiles:
                continue
            pth = os.path.join(pth0, ifiles)
            patient_files = os.listdir(pth)
            count2 = 0
            while not found and count2 < len(patient_files):
                ipatient = patient_files[count2]
                if patient_name == ipatient:
                    found = True
                    cancer_type = ifiles
                count2 = count2 + 1
        print("Cancer type: ", cancer_type)
        dict_patient = {}
        output_pth_patient = os.path.join(pth0, cancer_type)
        output_pth_patient = os.path.join(output_pth_patient, ipatient)
        dict_patient['image'] = os.path.join(output_pth_patient, 'img_reg_cropped_stripped.nii.gz')
        dict_patient['label'] = os.path.join(output_pth_patient, 'mask_reg_cropped.nii.gz')
        new_val.append(dict_patient)
    data['validation'] = new_val

    print(data)

    # #Save json file
    datafile_train = "/scratch_net/biwidl311/Cristina_Almagro/research_project/dataset/Cristina_dataset_preprocessed.json"
    json.dump(data, open(datafile_train, 'w'))

###############################################################################################
# Create .json files for test set
###############################################################################################
if json_test_new_dataset:

    # Load .json
    datafile_test = "/scratch_net/biwidl311/Cristina_Almagro/research_project/dataset/Cristina_dataset_test_skull_removed.json"
    f = open(datafile_test)
    data = json.load(f)
    new_data = data
    new_data['numTest'] = 35
    new_test = []

    pth0 = '/usr/bmicnas01/data-biwi-01/bmicdatasets-originals/Originals/USZ/metastases_v2/Data/test/'
    files = os.listdir(pth0)
    for ifiles in files:
        dict_patient = {}
        pth = os.path.join(pth0, ifiles)
        datafile = os.path.join(pth, 'noskull_zscore.nii.gz')
        if os.path.isfile(datafile):
            dict_patient['image'] = datafile
            dict_patient['label'] = os.path.join(pth, 'segment.nii.gz')
            new_test.append(dict_patient)
        else:
            datafile = os.path.join(pth, 'Corrected_bet_zscore.nii.gz')
            if os.path.isfile(datafile):
                dict_patient['image'] = datafile
                dict_patient['label'] = os.path.join(pth, 'segment.nii.gz')
                new_test.append(dict_patient)

    new_data['testing'] = new_test
    print(new_test)
    print(len(new_test))
    #Save .json file
    datafile_test = "/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/metastases_project/dataset_test.json"
    json.dump(data, open(datafile_test, 'w'))


###############################################################################################
# Create .json files for train set
###############################################################################################
if json_train_new_dataset:

    # Load .json
    datafile_test = "/scratch_net/biwidl311/Cristina_Almagro/research_project/dataset/Cristina_dataset_skull_removed.json"
    f = open(datafile_test)
    data = json.load(f)

    new_data = data
    new_test = []

    pth0 = '/usr/bmicnas01/data-biwi-01/bmicdatasets-originals/Originals/USZ/metastases_v2/Data/train/'
    files = os.listdir(pth0)
    for ifiles in files:
        dict_patient = {}
        pth = os.path.join(pth0, ifiles)
        datafile = os.path.join(pth, 'noskull_zscore.nii.gz')
        if os.path.isfile(datafile):
            dict_patient['image'] = datafile
            dict_patient['label'] = os.path.join(pth, 'segment.nii.gz')
            new_test.append(dict_patient)
        else:
            datafile = os.path.join(pth, 'Corrected_bet_zscore.nii.gz')
            if os.path.isfile(datafile):
                dict_patient['image'] = datafile
                dict_patient['label'] = os.path.join(pth, 'segment.nii.gz')
                new_test.append(dict_patient)

    # Split into validation and training
    validation = new_test[0:27]
    train = new_test[27:]

    new_data['training'] = train
    new_data['validation'] = validation
    new_data['numTrain'] = len(train)
    new_data['numVal'] = len(validation)
    print(new_data)

    #Save .json file
    datafile_test = "/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/metastases_project/dataset_train.json"
    json.dump(data, open(datafile_test, 'w'))