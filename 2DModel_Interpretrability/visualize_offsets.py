
################################################################################################
# Cristina Almagro-Pérez, 08-01-23 ; Visualize offsets of deformable convolution
################################################################################################

# https://github.com/developer0hye/PyTorch-Deformable-Convolution-v2/blob/main/offset_visualization.py

# The offset visualization from the original paper is in included in the author's repository:
# https://github.com/msracver/Deformable-ConvNets/tree/master/demo

import sys
import os
# Get the current directory path
current_dir = os.getcwd()
# Get the parent directory path
parent_dir = os.path.dirname(current_dir)
# Add the parent directory path to sys.path
sys.path.append(parent_dir)
import cv2
import torch.nn as nn
import numpy as np
import math
import torch
import statistics
import glob
import matplotlib.pyplot as plt
from network_deformable_2D import UNetV1V2, DUNetV1V2


select_images = False
obtain_offsets = True

################################################################################################
# Auxiliary functions
################################################################################################


def get_coloured_mask(mask, gt=False):
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

def my_plot_offsets(img, save_output,deform_layer,unet, roi_x, roi_y, color_circle):

    input_img_h, input_img_w = img.shape[:2]


    for offsets in save_output.outputs[deform_layer:deform_layer+1]:

        if unet:
            offsets = offsets * 0
        offset_tensor_h, offset_tensor_w = offsets.shape[2:]
        resize_factor_h, resize_factor_w = input_img_h / offset_tensor_h, input_img_w / offset_tensor_w

        offsets_y = offsets[:, ::2]
        offsets_x = offsets[:, 1::2]

        grid_y = np.arange(0, offset_tensor_h)
        grid_x = np.arange(0, offset_tensor_w)

        grid_x, grid_y = np.meshgrid(grid_x, grid_y)

        sampling_y = grid_y + offsets_y.detach().cpu().numpy()
        sampling_x = grid_x + offsets_x.detach().cpu().numpy()

        sampling_y *= resize_factor_h
        sampling_x *= resize_factor_w

        sampling_y = sampling_y[0]  # remove batch axis
        sampling_x = sampling_x[0]  # remove batch axis

        sampling_y = sampling_y.transpose(1, 2, 0)  # c, h, w -> h, w, c
        sampling_x = sampling_x.transpose(1, 2, 0)  # c, h, w -> h, w, c

        sampling_y = np.clip(sampling_y, 0, input_img_h)
        sampling_x = np.clip(sampling_x, 0, input_img_w)

        sampling_y = cv2.resize(sampling_y, dsize=None, fx=resize_factor_w, fy=resize_factor_h)
        sampling_x = cv2.resize(sampling_x, dsize=None, fx=resize_factor_w, fy=resize_factor_h)

        sampling_y = sampling_y[roi_y, roi_x]
        sampling_x = sampling_x[roi_y, roi_x]

        dist_list = []
        for y, x in zip(sampling_y, sampling_x):
            y = round(y)
            x = round(x)
            plt.scatter(x, y, c=color_circle, s=0.5, linewidths=1)

            dist = math.sqrt((x - roi_x) ** 2 + (y - roi_y) ** 2)
            dist_list.append(dist)
        plt.scatter(roi_x, roi_y, c=[0, 1, 0], s=5, linewidths=5)
        mean_dist = statistics.mean(dist_list)
        std_dist = statistics.stdev(dist_list)
        print("The mean is ", mean_dist)
        print("The std is ", std_dist)
    return dist_list


class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []


#######################################################################################################
if obtain_offsets:
    print("Obtaining offsets")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DUNetV1V2(img_ch=1, output_ch=2)
    model_datafile = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/metastases_project/deformable/model_2D_DUNetV1V2_08_01_23_exp1/DUNetV1V2200.pth'
    model.load_state_dict(torch.load(model_datafile, map_location=torch.device('cpu')))
    model = model.to(device)
    save_output = SaveOutput()
    pth_images = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/attention_unet/test/images_tight1/'
    imlist = glob.glob(pth_images +'*')

    imlist = imlist[0]
    # Select an image (image saved with find_slices_TP code)
    image1A_TP = False
    image2A_TP = False
    image3A_TP = False
    image4A_TP = False
    image1B_TP = False
    image2B_TP = False
    image3B_TP = False

    image1A_FN = False
    image2A_FN = False
    image3A_FN = False
    image4A_FN = False
    image1B_FN = False
    image2B_FN = False
    image3B_FN = False

    image1A_FP = False
    image2A_FP = False
    image3A_FP = False
    image4A_FP = False
    image1B_FP = False
    image2B_FP = False
    image3B_FP = False
    image4B_FP = False

    if image1A_TP:
        imlist = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/attention_unet/test/images_tight1/000026-34_section_200.npy'
        roi_y_met, roi_x_met = 200, 132
        roi_y, roi_x = 100, 150
    elif image2A_TP:
        imlist = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/attention_unet/test/images_tight1/000020-122_section_170.npy'
        roi_y_met, roi_x_met = 66, 224
        roi_y, roi_x = 100, 150
    elif image3A_TP:
        imlist = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/attention_unet/test/images_tight1/0000146-81_section_97_TP.npy'
        roi_y_met, roi_x_met = 99, 228
        roi_y, roi_x = 100, 200
    elif image4A_TP:
        imlist = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/attention_unet/test/images_tight1/000019-35_section_157_TP.npy'
        roi_y_met, roi_x_met = 181, 262
        roi_y, roi_x = 100, 150

    elif image1B_TP:
        imlist = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/attention_unet/test/images_tight1/000097-29_section_85_TP.npy'
        roi_y_met, roi_x_met = 172, 266
        roi_y, roi_x = 100, 250
    elif image2B_TP:
        imlist = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/attention_unet/test/images_tight1/000151-87_section_238_TP.npy'
        roi_y_met, roi_x_met = 124, 190
        roi_y, roi_x = 100, 150
    elif image3B_TP:
        imlist = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/attention_unet/test/images_tight1/000154-90_section_60_TP.npy'
        roi_y_met, roi_x_met = 177, 229
        roi_y, roi_x = 100, 250

    #########################################################################

    elif image1A_FN:
        imlist = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/attention_unet/test/images_tight1/000146-81_section_118_FN.npy'
        roi_y_met, roi_x_met = 87, 191
        roi_y, roi_x = 100, 150
    elif image2A_FN:
        imlist = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/attention_unet/test/images_tight1/000146-81_section_109_FN.npy'
        roi_y_met, roi_x_met = 220, 195
        roi_y, roi_x = 80, 150
    elif image3A_FN:
        imlist = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/attention_unet/test/images_tight1/000053-163_section_102_FN.npy'
        roi_y_met, roi_x_met = 224, 234
        roi_y, roi_x = 80, 150
    elif image4A_FN:
        imlist = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/attention_unet/test/images_tight1/000081-195_section_204_FN.npy'
        roi_y_met, roi_x_met = 179, 163
        roi_y, roi_x = 100, 150
    elif image1B_FN:
        imlist = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/attention_unet/test/images_tight1/000067-179_section_123_TP.npy'
        roi_y_met, roi_x_met = 111, 279
        roi_y, roi_x = 100, 250
    elif image2B_FN:
        imlist = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/attention_unet/test/images_tight1/000067-179_section_88_TP.npy'
        roi_y_met, roi_x_met = 135, 235
        roi_y, roi_x = 200, 250
    elif image3B_FN:
        imlist = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/attention_unet/test/images_tight1/000067-179_section_252_TP.npy'
        roi_y_met, roi_x_met = 201, 199
        roi_y, roi_x = 150, 200

    ####################################################################################
    elif image1A_FP:
        imlist = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/attention_unet/test/images_tight1/000020-122_section_184_FP.npy'
        roi_y_met, roi_x_met = 222, 240
        roi_y, roi_x = 100, 150
    elif image2A_FP:
        imlist = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/attention_unet/test/images_tight1/000097-29_section_60_FP.npy'
        roi_y_met, roi_x_met = 149, 273
        roi_y, roi_x = 200, 250
    elif image3A_FP:
        imlist = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/attention_unet/test/images_tight1/000065-177_section_109_FP.npy'
        roi_y_met, roi_x_met = 104, 234
        roi_y, roi_x = 200, 250
    elif image1B_FP:
        imlist = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/attention_unet/test/images_tight1/000097-29_section_88_FP.npy'
        roi_y_met, roi_x_met = 66, 215
        roi_y, roi_x = 100, 250
    elif image2B_FP:
        imlist= '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/attention_unet/test/images_tight1/000132-64_section_138_FP.npy'
        roi_y_met, roi_x_met = 88, 154
        roi_y, roi_x = 100, 250
    elif image3B_FP:
        imlist = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/attention_unet/test/images_tight1/000146-81_section_134_FP.npy'
        roi_y_met, roi_x_met = 145, 292
        roi_y, roi_x = 100, 250

    for name, layer in model.named_modules():
        # p_conv are the layers containing the offsets
        if "p_conv" in name and isinstance(layer, nn.Conv2d):
            layer.register_forward_hook(save_output)

    with torch.no_grad():
        image = np.load(imlist)
        plt.imshow(image, cmap='gray')
        plt.show()
        input_img_h, input_img_w = image.shape

        image_tensor = torch.from_numpy(image)
        image_tensor = image_tensor.view(1, 1, input_img_h, input_img_w)
        image_tensor = image_tensor.float()
        image_tensor = image_tensor.to(device)

        out = model(image_tensor)

        # There are 10 layers that have deformable convolutions:
        # Select the indices for the layers you desire to plot the offsets. Examples:
        #indx = [2, 3, 7, 9]
        #indx = [0, 1, 2, 3,4,5,6, 7,8, 9]
        #indx = [6, 7, 9]
        indx = [2, 7, 9]
        dist_list_met = []
        dist_list_back = []

        for count, deform_layer in enumerate(indx):

            print("Plotting offsets of deformable layer: ", indx)

            if count == 0:
                plt.imshow(image, cmap='gray')

            # Plot offsets in metastasis' centroid
            color_circle = [0, 0, 1]
            dist_list = my_plot_offsets(image, save_output, deform_layer, unet=False, roi_x=roi_x_met, roi_y=roi_y_met, color_circle=color_circle)
            dist_list_met.extend(dist_list)

            # Plot offsets in a random point in the background
            color_circle = [1, 0, 0]
            dist_list = my_plot_offsets(image, save_output, deform_layer, unet=False, roi_x=roi_x, roi_y=roi_y, color_circle=color_circle)
            dist_list_back.extend(dist_list)

            plt.title("Offsets of deformable convolutional layer " + str(deform_layer))


        # Uncomment this section if you want to do zoom-in in the metastasis point
        # xmin = roi_x_met - 15
        # xmax = roi_x_met + 15
        # ymin = roi_y_met - 15
        # ymax = roi_y_met + 15
        # plt.xlim(xmin, xmax)
        # plt.ylim(ymin, ymax)

        # Uncomment this section if you want to do zoom-in in the background point
        # xmin = roi_x - 15
        # xmax = roi_x + 15
        # ymin = roi_y - 15
        # ymax = roi_y + 15
        # plt.xlim(xmin, xmax)
        # plt.ylim(ymin, ymax)
        plt.show()

mean_dist_met = statistics.mean(dist_list_met)
std_dist_met = statistics.stdev(dist_list_met)
print("The  met mean is ", mean_dist_met)
print("The met std is ", std_dist_met)

mean_dist_back = statistics.mean(dist_list_back)
std_dist_back = statistics.stdev(dist_list_back)
print("The back mean is ", mean_dist_back)
print("The back std is ", std_dist_back)

