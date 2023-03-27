%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Cristina Almagro Pérez, 2022, ETH Zürich
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Create a variable with the volume of TP and FN metatasis

% TP: Green colour, label 1
% FN: Yellow colour, label 2
% FP: Red colour, label 3
clear;clc;close all;
addpath('C:\Users\crist\Desktop\Lab_Ender\Research_project\code\analyse_detections\npy-matlab-master\npy-matlab-master\npy-matlab\')

pth_pred_labelled = 'C:\Users\crist\Desktop\Lab_Ender\Research_project\code\analyse_detections\postprocessed1\';

count=0;
volume_data_GT_attention = zeros(count,2);
chr_patients=dir([pth_pred_labelled,'*.npy']);

for kk=1:54
    
    %patient_name=split(chr_patients(index(kk),:),'/');patient_name=patient_name{end};
    patient_name=chr_patients(kk).name;
%     patient_name=split(chr_patients((kk),:),'/');patient_name=patient_name{end};
%     patient_name=split(patient_name,'.');patient_name=patient_name{1};
    
    disp(['Analyzing patient: ', patient_name, ' ### ' num2str(kk), '/', '54']);
    
    % Load images
    %im_pred = squeeze(readNPY([pth_pred_labelled,patient_name, '.npy']));
    im_pred = squeeze(readNPY([pth_pred_labelled,patient_name]));
    im_tp = im_pred == 1;
    im_fn = im_pred == 2;
    
    im_tp_labelled = bwlabeln(im_tp);num_tp = bwconncomp(im_pred==1).NumObjects;
    for i=1:num_tp
        count = count +1;
        tp = im_tp_labelled == i;
        vol = sum(tp(:)) * 0.001; %volume in mL
        volume_data_GT_attention(count,1)=vol;
        volume_data_GT_attention(count,2)=1; 
    end
    
     im_fn_labelled = bwlabeln(im_fn);num_fn = bwconncomp(im_pred==2).NumObjects;
    for i=1:num_fn
        count = count +1;
        fn = im_fn_labelled == i;
        vol = sum(fn(:)) * 0.001; %volume in mL
        if vol>1
            disp(vol);
        end
        volume_data_GT_attention(count,1)=vol;
        volume_data_GT_attention(count,2)=2; 
    end
    

end