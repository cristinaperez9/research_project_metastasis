########################################################################################################################
# Cristina Almagro-Pérez, ETH Zürich, 2023
########################################################################################################################

'init.py', 'train.py' and 'inference.py' are the three main files for specifying the parameters, training and inference respectively.


- File 'networks.py' contains 4 architectures: UNet, UNet with deep supervision, Attention U-Net and Attention U-Net with deep supervision.
- File 'network_deformable.py' contains 3 architectures: ThreeOffsetsAttentionUNet, DUNetV1V2***, DeformAttention.
- File 'modules_deformable.py' contains the Three Offset Block as well as the modules for Deformable Attention Unet. 
- File 'unet_parts_3D.py' contains some modules required in a 3D U-Net architecture; 'deform_part_3D.py' contains
  modules used in a deformable unet (offsets are learned). These modules are used in the DUNetV1V2 network.
  
  Note: for baseline models use only the models within networks.py.

###### Training #####
In 'init.py' file specify the following parameters. The rest of parameters can be left unchanged. 

1. Specify "model_folder". Location where the training loss, validation loss and validation DSC will be saved. The parameters
of the model will be also saved every two epochs in files with name 'ArchitectureXXX.pth' where XXX is the training epoch and
Architecture is the network name. See point 2 below.

2. Specify the "network" . Options:
   - UNet
   - Attention: Attention U-Net.
   - ThreeOffsetsAttentionUNet.
   - DUNetV1V2: Deformable U-Net (***3D Extension of "DUNet: A deformable network for retinal vessel segmentation")
   - DeformAttention: Attention U-Net in which some of the convolutions have been substituted by deformable convolutions
     (with learnable offsets).

###### Inference #####
In addition to the above specifications include the following:
1. specify "outpth0". Output path were the predictions will be saved.The code is set to perform inference in the best 5 epochs
according to the validation set. Inside outpth0, five folders will be created for each of the best epochs.
2. Then you can combine the predicitions using the code 'combine_predictions.py' or just use the predicitions of one of the folders.


###################################################################################################################
# Notes regarding memory constraints
###################################################################################################################
The default configuration uses a batch size of 2 and 4 patches for 128 x 128 x 128 are extracted per patient. 
Hence the 'effective' batch size is 8.
- Attention U-Net can be trained with this configuration using 1 GPU.
- Attention U-Net with deep supervision requires 2 GPU to train with this configuration.
- DUNetV1V2, DeformAttention and ThreeOffsetAttentionUNet require 1 GPU per each element in the batch (each patch 
  of 128 x 128 x 128). I trained this models using batch size = 2, and extracting 2 patches per patient (variable num_samples
in init.py). I used 4 GPUs.

###################################################################################################################
# Notes regarding trained models
###################################################################################################################

## Trained Attention U-Net ##
folder = /usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/attention/my_model1
The 5 best epochs are: 290 (DSC=0.82), 350 (DSC=0.81), 418 (DSC=0.81), 338 (DSC=0.81), 430 (DSC=0.81).
For example, the parameters of the best epoch are stored in the above folder with name 'Attention290.pth'


## Trained Deformable U-Net (DUNetV1V2) ##
folder = /usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/metastases_project/deformable/model_3D_DUNetV1V2_11_01_23_exp1/
The 5 best epochs are: 696 (DSC=0.77), 796 (DSC=0.76), 694 (DSC=0.76), 654 (DSC=0.75), 332 (DSC=0.75)
For example, the parameters of the best epoch are stored in the above folder with name DUNetV1V2696.pth

## Trained ThreeOffsetsAttentionUNet ##
folder = /usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/metastases_project/deformable/model_3D_ThreeOffsetsAttentionUNet_05_02_23_exp1/
The 5 best epochs are: 234 (DSC=0.72), 286 (DSC=0.72), 232 (DSC=0.72), 288 (DSC=0.71), 262 (DSC=0.71)
For example, the parameters of the best epoch are stored in the above folder with name ThreeOffsetsAttentionUNet234.pth 

