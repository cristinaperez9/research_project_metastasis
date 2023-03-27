addpath('C:\Users\crist\Desktop\Lab_Ender\Research_project\code\analyse_detections\npy-matlab-master\npy-matlab-master\npy-matlab\') 
addpath('C:\Users\crist\Desktop\Masters\First year\Second semester\Project\Code\statistical_analysis\');
%clear;clc;close all;
graph_1 = 0; % Predicted metastases' volume vs. GT metastases' volume (it requires metastasis to be TP);
graph_2 = 0;  % BlandAltman plot
graph_3 = 0;  % Histogram of all GT metastases with classification in TP and FN
graph_4 = 0;  % Box-whisker plots.
graph_5= 0; % Stratified DSC (considering TP and FN)
graph_6= 0; % Stratified DSC (considering TP and FN) - Normal scale (not log!)
graph_7 = 0; % Proportion of occupied volume (analyze small metastases)
graph_8 = 1; % DSC stratified new
%% Plot graph 1
if graph_1 == 1
    % Load variable containing the predicted and GT metastases volume
    load('volume_data.mat');%volume_data(101,2) = 48;
    
    volume_data=volume_data*0.001 ; % 
    % Calculate linear regression line
    xvalues= volume_data(:,1);yvalues=volume_data(:,2);
    %Make a line fit
    coefficients=polyfit(xvalues,yvalues,1);
    xFit=linspace(min(xvalues),max(xvalues)+20,1000);
    yFit=polyval(coefficients,xFit);
    yRsquared=polyval(coefficients,xvalues);
    [r,p]=corrcoef(yvalues,yRsquared); % It contains the Pearson correlation coefficient; p is the p value
    rsqu=r(1,2)*r(1,2);
    disp(rsqu)
    
    % Create plot
    figure;
    hold on;
    % Add black line indicating the division of the quadrant (y=x)
    plot(1:90,1:90,'-','color',[0 0 0],'LineWidth',1.5);
    % Add fitting line 
    plot(xFit,yFit,'--','color',[0.00,0.36,0.65],'LineWidth',1.5);
    % Add scatter of points
    scatter(xvalues,yvalues,20,[0.00,0.36,0.65],'filled');
    set(gcf,'color','w');
    xlabel('Ground Truth metastasis volume (ml)','FontSize',15);
    ylabel('Predicted metastasis volume (ml)','FontSize',15);
    box on
    set(gca,'linewidth',1)
    xlim([0 15]); %xlim([0 90]);
    ylim([0 15]); %ylim([0 90]);
    xticks([0, 5, 10, 15]);%xticks([0, 20, 40, 60, 80]);
    yticks([0, 5, 10, 15]);%yticks([0, 20, 40, 60, 80]);
    
    
     
    % Calculate Lin Concordance Correlation coefficient
    CCC = f_CCC(volume_data,0.05);
    CCC1 = concordance_correlation_coefficient(xvalues, yvalues);
end

%% Graph 2
if graph_2 == 1
    % Load variable containing the predicted and GT metastases volume
    %volume_data = readNPY('volume_data.npy');
    load('volume_data.mat');%volume_data(101,2) = 48;
    
    volume_data = volume_data * 0.001;
    tit = 'BlandAltman plot';
    label = {'True volume','Predicted volume','mL'}; % Names of data sets
    colors = [0 0.4470 0.7410];
    colors = [0.00,0.36,0.65];
    [cr, fig, statsStruct] = BlandAltman(volume_data(:,1), volume_data(:,2),label,tit,'colors',colors);
 
% Analyze outliers
volume_data(:,3) = (volume_data(:,1)+ volume_data(:,2))/2; %third column is the average
volume_data(:,4) = (volume_data(:,2)- volume_data(:,1)); %fourth column is the difference
volume_data_sv =volume_data(volume_data(:,3)<20,:); % analyze only average volumes smaller than 20 mL
[cr, fig, statsStruct] = BlandAltman(volume_data_sv(:,1), volume_data_sv(:,2),label,tit,'colors',colors);

end




%% Graph 3
if graph_3 == 1
    % Load variable containing the volume and label of the GT metastases
    % 1: TP ; 2: FN
    load('mymodel1_metsize_data_GT_attention.mat'); volume_data_GT = metsize_data_GT_attention;
    tp = volume_data_GT(volume_data_GT(:,2) == 1);
    fn = volume_data_GT(volume_data_GT(:,2) == 2);
    figure;
    box on
    set(gca,'linewidth',1)
    set(gcf,'color','w');
    h = histogram(tp,0:3:35); %0:5:80
    edges = h.BinEdges ;
    hold on
    p = histogram(fn,edges);
    %xlabel('Lesion volume (mL)');
    xlabel('Lesion diameter (mm)');
    ylabel('Count');
    c_tp = [157,112,204]/255; %purple
    c_tp = [42,113,189]/255; %blue
    c_fn = [217,203,52]/255; %yellow
    c_fn = [207,110,110]/255;%reddish
    h.FaceColor = c_tp;
    p.FaceColor = c_fn; 
end
%% Graph 4   % Box-whisker plots.
if graph_4 == 1
    
   % all metastases
    load('dsc_stratified_Attention.mat');
    var = dsc_stratified_Attention;
    gt=dsc_stratified_Attention(:,1); % extract ground truth volumes
    %[N,edges,bin] = histcounts(log(gt) ,linspace(1, max(log(gt)),10)); %cluster GT volumes
    [N,edges,bin] = histcounts(gt ,0:3:35); %cluster GT volumes
    unique_bins = unique(bin);
    dice_vals = {};
    group = [];
    for kk = 1:max(bin(:))
       dice_vals{kk} = var(bin==kk,2); %3 for volume

       add = ones(size(dice_vals{kk},1)); 
       if size(add,1)> 0
        add = add(:,1);
       else
        add = 1;
       end
       group = [group; kk * add];

    end
% Create figure
figure;
hold on
set(gcf,'color','w');
boxplot([dice_vals{1};dice_vals{2};dice_vals{3};dice_vals{4};dice_vals{5};dice_vals{6};...
    dice_vals{7};dice_vals{8};dice_vals{9};dice_vals{10};dice_vals{11}],group);
%ylim([0.2152,1]);
ylim([0,1]);
ylabel('DSC','fontweight','bold');
xlabel('Metastasis size (mL), log scale','fontweight','bold');
title('Segmentation performance of detected metastastases','fontweight','bold');

middle_volume = zeros(1, max(bin(:)));
for i=1:length(edges)-1
    low_vol = edges(i);
    high_vol = edges(i+1);
    middle_volume(i)=(low_vol + high_vol)/2;
end
middle_volume = round(exp(middle_volume), 1);
%set(gca,'XTickLabel',{'A','B','C'})
xticklabels(middle_volume)
    











%     
%    % option 1: all ground truth metastases
%    load('dice_stratified_Attention.mat');
%    load('volume_data_GT.mat'); fn = volume_data_GT(volume_data_GT(:,2) == 2,1);
%    
%    myvar =zeros(165, 2); %first column volume, second column dice
%    myvar(1:length(dice_stratified),1) = dice_stratified(:,1);
%    myvar(1:length(dice_stratified),2) = dice_stratified(:,3);
%    myvar(length(dice_stratified)+1:end,1) = fn;
%    myvar(length(dice_stratified)+1:end,2) = 0;
%    
%    tp = myvar;
%    %[N,edges,bin] = histcounts(tp(:,1) ,0:5:80); %cluster GT volumes
%    [N,edges,bin] = histcounts(log(tp(:,1)) ,linspace(1, max(log(tp(:,1))),10)); %cluster GT volumes
%    unique_bins = unique(bin);
%    dice_vals = {};
%    group = [];
%    for kk = 1:max(bin(:))
%        dice_vals{kk} = tp(bin==kk,2);
%        
%        add = ones(size(dice_vals{kk},1)); 
%        if size(add,1)> 0
%         add = add(:,1);
%          
%        else
%         add = 1;
%        end
%        group = [group; kk * add];
%       
%    end
% % Create figure
% figure;
% hold on
% set(gcf,'color','w');
% % boxplot([dice_vals{1};dice_vals{2};dice_vals{3};dice_vals{4};dice_vals{5};dice_vals{6};...
% %     dice_vals{7};dice_vals{8};dice_vals{9};dice_vals{10};dice_vals{11};dice_vals{12};...
% %     dice_vals{13};dice_vals{14};dice_vals{15};dice_vals{16}],group);
% boxplot([dice_vals{1};dice_vals{2};dice_vals{3};dice_vals{4};dice_vals{5};dice_vals{6};...
%     dice_vals{7};dice_vals{8};dice_vals{9};],group);
%     
% 
% %set(gca,'XTickLabel',{'A','B','C'})
% 
% % option 2: only TP metastases
%    load('dice_stratified.mat');
%    
%    
%    tp = dice_stratified;
%    %[N,edges,bin] = histcounts(tp(:,1) ,0:5:80); %cluster GT volumes
%    [N,edges,bin] = histcounts(log(tp(:,1)) ,linspace(1, max(log(tp(:,1))),10)); %cluster GT volumes
%    unique_bins = unique(bin);
%    dice_vals = {};
%    group = [];
%    for kk = 1:max(bin(:))
%        dice_vals{kk} = tp(bin==kk,3);
%        
%        add = ones(size(dice_vals{kk},1)); 
%        if size(add,1)> 0
%         add = add(:,1);
%          
%        else
%         add = 1;
%        end
%        group = [group; kk * add];
%       
%    end
% % Create figure
% figure;
% hold on
% set(gcf,'color','w');
% % boxplot([dice_vals{1};dice_vals{2};dice_vals{3};dice_vals{4};dice_vals{5};dice_vals{6};...
% %     dice_vals{7};dice_vals{8};dice_vals{9};dice_vals{10};dice_vals{11};dice_vals{12};...
% %     dice_vals{13};dice_vals{14};dice_vals{15};dice_vals{16}],group);
% boxplot([dice_vals{1};dice_vals{2};dice_vals{3};dice_vals{4};dice_vals{5};dice_vals{6};...
%     dice_vals{7};dice_vals{8};dice_vals{9};],group);
%     
% 
% %set(gca,'XTickLabel',{'A','B','C'})
%    
%      
% end
end
%% Create whisker plot considering TP and FN metastases
if graph_5 == 1
    
    load('dsc_stratified_Attention.mat'); %load TP
    load('volume_FN_attention.mat'); %load FN
    fn = zeros(length(volume_FN_attention),2); %DSC is 0 for fn
    fn(:,1)=volume_FN_attention;
    tp = [dsc_stratified_Attention(:,1), dsc_stratified_Attention(:,3)];
    % Concatenate both
    var = [tp; fn];
    % Save var for future use
    %save('volume_FN_&_TP_attention.mat','var');
    gt=var(:,1); % extract ground truth volumes
    [N,edges,bin] = histcounts(log(gt) ,linspace(1, max(log(gt)),10)); %cluster GT volumes
    unique_bins = unique(bin);
    dice_vals = {};
    group = [];
    for kk = 0:max(bin(:))
       dice_vals{kk+1} = var(bin==kk,2);

       add = ones(size(dice_vals{kk+1},1)); 
       if size(add,1)> 0
        add = add(:,1);
       else
        add = 1;
       end
       group = [group; kk * add];

    end
    % Create figure
    figure;
    hold on
    set(gcf,'color','w');
    % boxplot([dice_vals{1};dice_vals{2};dice_vals{3};dice_vals{4};dice_vals{5};dice_vals{6};...
    %     dice_vals{7};dice_vals{8};dice_vals{9};dice_vals{10};dice_vals{11};dice_vals{12};...
    %     dice_vals{13};dice_vals{14};dice_vals{15};dice_vals{16}],group);
    boxplot([dice_vals{1};dice_vals{2};dice_vals{3};dice_vals{4};dice_vals{5};dice_vals{6};...
    dice_vals{7};dice_vals{8};dice_vals{9};dice_vals{10}],group);
    ylim([0.2152,1]);
    ylabel('DSC','fontweight','bold');
    xlabel('Metastasis size (mL), log scale','fontweight','bold');
    title('Segmentation performance considering all GT metastastases','fontweight','bold');

    middle_volume = zeros(1, max(bin(:)));
    for i=1:length(edges)-1
    low_vol = edges(i);
    high_vol = edges(i+1);
    middle_volume(i)=(low_vol + high_vol)/2;
    end
    middle_volume = round(exp(middle_volume), 1);
    %set(gca,'XTickLabel',{'A','B','C'})
    xticklabels(middle_volume)
end

%% Create whisker plot considering TP and FN metastases (normal scale)
if graph_6 == 1
    
    load('dsc_stratified_Attention.mat'); %load TP
    load('volume_FN_attention.mat'); %load FN
    load('dsc_stratified_nnUNet.mat'); %load TP
    var = dsc_stratified_nnUNet;
    %fn = zeros(length(volume_FN_attention),2); %DSC is 0 for fn
    %fn(:,1)=volume_FN_attention;
    %tp = [dsc_stratified_Attention(:,1), dsc_stratified_Attention(:,3)];
    % Concatenate both
    %var = [tp; fn];
    % Save var for future use
    %save('volume_FN_&_TP_attention.mat','var');
    gt=var(:,1); % extract ground truth volumes
    
    top_edge=[0.1, 0.5, 1, 5, 15, 30];
    group = [];
    dice_vals{1}=[];dice_vals{2}=[];dice_vals{3}=[];dice_vals{4}=[];dice_vals{5}=[];dice_vals{6}=[];dice_vals{7}=[];
    for kk=1:length(var)
        volume=var(kk,1);
        if volume<top_edge(1)
            dice_vals{1} = [var(kk,2), dice_vals{1}];
        elseif volume>top_edge(1) && volume<top_edge(2)
            dice_vals{2} = [var(kk,2), dice_vals{2}];
        elseif volume>top_edge(2) && volume<top_edge(3)
             dice_vals{3} = [var(kk,2), dice_vals{3}];
        elseif volume>top_edge(3) && volume<top_edge(4)
             dice_vals{4} = [var(kk,2), dice_vals{4}];
        elseif volume>top_edge(4) && volume<top_edge(5)
             dice_vals{5} = [var(kk,2), dice_vals{5}];
        elseif volume>top_edge(5) && volume<top_edge(6)
             dice_vals{6} = [var(kk,2), dice_vals{6}];
        elseif volume>top_edge(6) 
             dice_vals{7} = [var(kk,2), dice_vals{7}];
        end
    
    end
    
    % Create group variable
    group=[];
    for i=1:length(dice_vals)
        dv=dice_vals{i};
        for j=1:length(dice_vals{i})
            group = [group; i];
        end
        
    end
    
    

    
 
    


% Create figure
figure;
hold on
set(gcf,'color','w');
% boxplot([dice_vals{1};dice_vals{2};dice_vals{3};dice_vals{4};dice_vals{5};dice_vals{6};...
%     dice_vals{7};dice_vals{8};dice_vals{9};dice_vals{10};dice_vals{11};dice_vals{12};...
%     dice_vals{13};dice_vals{14};dice_vals{15};dice_vals{16}],group);
boxplot([dice_vals{1}';dice_vals{2}';dice_vals{3}';dice_vals{4}';dice_vals{5}';dice_vals{6}';dice_vals{7}'],group);
ylim([0,1]);
ylabel('DSC','fontweight','bold');
xlabel('Metastasis size (mL)','fontweight','bold');
title('Segmentation performance considering all GT metastastases','fontweight','bold');
    
    
%     [N,edges,bin] = histcounts(log(gt) ,linspace(1, max(log(gt)),10)); %cluster GT volumes
%     unique_bins = unique(bin);
%     dice_vals = {};
%     group = [];
%     for kk = 0:max(bin(:))
%        dice_vals{kk+1} = var(bin==kk,2);
% 
%        add = ones(size(dice_vals{kk+1},1)); 
%        if size(add,1)> 0
%         add = add(:,1);
%        else
%         add = 1;
%        end
%        group = [group; kk * add];
% 
%     end
%     % Create figure
%     figure;
%     hold on
%     set(gcf,'color','w');
%     % boxplot([dice_vals{1};dice_vals{2};dice_vals{3};dice_vals{4};dice_vals{5};dice_vals{6};...
%     %     dice_vals{7};dice_vals{8};dice_vals{9};dice_vals{10};dice_vals{11};dice_vals{12};...
%     %     dice_vals{13};dice_vals{14};dice_vals{15};dice_vals{16}],group);
%     boxplot([dice_vals{1};dice_vals{2};dice_vals{3};dice_vals{4};dice_vals{5};dice_vals{6};...
%     dice_vals{7};dice_vals{8};dice_vals{9};dice_vals{10}],group);
%     ylim([0.2152,1]);
%     ylabel('DSC','fontweight','bold');
%     xlabel('Metastasis size (mL), log scale','fontweight','bold');
%     title('Segmentation performance considering all GT metastastases','fontweight','bold');
% 
%     middle_volume = zeros(1, max(bin(:)));
%     for i=1:length(edges)-1
%     low_vol = edges(i);
%     high_vol = edges(i+1);
%     middle_volume(i)=(low_vol + high_vol)/2;
%     end
%     middle_volume = round(exp(middle_volume), 1);
%     %set(gca,'XTickLabel',{'A','B','C'})
%     xticklabels(middle_volume)
% end
end
%% Graph 7   % Box-whisker plots of proportion of occupied volume
if graph_7 == 1
    
   % all metastases
%     load('proportion_volume_small_mets.mat');
%     var = proportion_volume_small_mets;
    %load('proportion_volume_DeformUNet_2D.mat');
    %var = proportion_volume_proportion_volume_DeformUNet_2D;
%     load('proportion_volume_DeformAttention_2D.mat');
%     var = proportion_volume_DeformAttention_2D;
    load('proportion_volume_Attention_2D.mat');
    var = proportion_volume_Attention_2D;
%     load('proportion_volume_DUNetV1V2_2D.mat');
%     var = proportion_volume_DUNetV1V2_2D;
    %load('proportion_volume_DUNetV1V2_3D.mat');
    %var = proportion_volume_DUNetV1V2_3D;
    %load('proportion_volume_nnUNet_3D_16_01_23.mat');
    %var = proportion_volume_nnUNet_3D_16_01_23;
%     load('dsc_stratified_nnUNet_5_perc.mat'); %load TP
%     var0 = dsc_stratified_nnUNet_5_perc;
%     var1 = var0(:,1);
%     var2 = var0(:,3);
%     var = [var1,var2];

    %load('proportion_volume_Attention_3D_12_01_23_exp1.mat');
    %var = proportion_volume_Attention_3D_12_01_23_exp1;
    %load('proportion_volume_DeformAttention_3D_02_01_23.mat');
    %var = proportion_volume_DeformAttention_3D_02_01_23;
    
    %load('proportion_volume_small_mets_subpixel.mat');
    %var = proportion_volume_small_mets_subpixel;
   
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Notes:
    % 1 - proportion_volume_small_mets  corresponds to Attention U-Net
    %      proportion_volume_small_mets_subpixel corresponds to subpixel embedding A. Wong et al.
    % 2 - Column 1: size GT met (MetNet Paper definition)
    % 3 - Column 2: propotion of occupied volume
    % The variable named down dsc is not dsc but the proportion of occupied
    % volume
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    gt=var(:,1); % extract ground truth diameters
    myedges=(0:3:35)+0.01;
    [N,edges,bin] = histcounts(gt ,myedges); %cluster GT sizes
    unique_bins = unique(bin);
    dice_vals = {};
    group = [];
    for kk = 1:max(bin(:))
       dice_vals{kk} = var(bin==kk,2); 

       add = ones(size(dice_vals{kk},1)); 
       if size(add,1)> 0
        add = add(:,1);
       else
        add = 1;
       end
       group = [group; kk * add];

    end
% Create figure
figure;
hold on
set(gcf,'color','w');
boxplot([dice_vals{1};dice_vals{2};dice_vals{3};dice_vals{4};dice_vals{5};dice_vals{6};...
    dice_vals{7};dice_vals{8};dice_vals{9};dice_vals{10};dice_vals{11}],group);

ylim([0,100]);
ylabel('% of occupied volume in the prediction','fontweight','bold');
xlabel('Metastasis size (mm)','fontweight','bold');
title('Deform UNet (2D)','fontweight','bold');

middle_volume = zeros(1, max(bin(:)));
for i=1:length(edges)-1
    low_vol = edges(i);
    high_vol = edges(i+1);
    middle_volume(i)=(low_vol + high_vol)/2;
end
middle_volume = round(exp(middle_volume), 1);
%set(gca,'XTickLabel',{'A','B','C'})
xticklabels(middle_volume)



%Mean overlap
b=dice_vals{1,2};
c=b>0;
d=b(c);
mean(d)
end

%% Graph 8

if graph_8
    
    %load('dsc_stratified_nnUNet_5_perc.mat');
    %var = dsc_stratified_nnUNet_5_perc;
    %var = var(:,1:2);
    
    %load('dsc_stratified_Attention.mat');
    %var = dsc_stratified_Attention;
    %load('dsc_stratified_ThreeOffset_05_02_23_exp1.mat');
    %var = dsc_stratified_ThreeOffset_05_02_23_exp1;
    load('dsc_stratified_UNet_28_02_23_exp1.mat');
    var = dsc_stratified_UNet_28_02_23_exp1;
    load('dsc_stratified_TwoKernelUNet_28_02_23_exp1.mat');
    var = dsc_stratified_TwoKernelUNet_28_02_23_exp1;
    
%     load('dsc_stratified_DUNetV1V2.mat');
%     var = dsc_stratified_DUNetV1V2;
%     var = var(:,1:2);

%       load('dsc_stratified_Attention_21_01_23_exp3.mat');
%       var = dsc_stratified_Attention_21_01_23_exp3;
%       var = var(:,1:2);
        %load('dsc_stratified_Attention_30_01_23_exp2.mat');
        %var = dsc_stratified_Attention_30_01_23_exp2;


    
    
    
    
    gt=var(:,1); % extract ground truth diameters
    myedges=(0:3:34)+0.01;
    [N,edges,bin] = histcounts(gt ,myedges); %cluster GT sizes
    unique_bins = unique(bin);
    dice_vals = {};
    group = [];
    for kk = 1:max(bin(:))
       dice_vals{kk} = var(bin==kk,2); 

       add = ones(size(dice_vals{kk},1)); 
       if size(add,1)> 0
        add = add(:,1);
       else
        add = 1;
       end
       group = [group; kk * add];

    end
% Create figure
figure;
hold on
set(gcf,'color','w');
boxplot([dice_vals{1};dice_vals{2};dice_vals{3};dice_vals{4};dice_vals{5};dice_vals{6};...
    dice_vals{7};dice_vals{8};dice_vals{9};dice_vals{10};dice_vals{11}],group);
%Specify color of the box
h = findobj(gca,'Tag','Box');
%set(h,{'linew'},{1.5})
%set(h,{'MarkerEdgeColor'},'k')
for j=1:length(h)
    patch(get(h(j),'XData'),get(h(j),'YData'),[0.00,0.36,0.65],'FaceAlpha',.5); %[0 43 107]/255
end
ylim([0,1]);
ylabel('DSC','fontweight','bold');
xlabel('Metastasis size (mm)','fontweight','bold');
title('nnUNet (3D)','fontweight','bold');
set(gca,'LineWidth',0.75)
lines = findobj(gcf, 'type', 'line', 'Tag', 'Median');
granate = [151 0 10]/255;
set(lines, 'Color', granate);
set(lines, 'Linewidth', 1);


middle_volume = zeros(1, max(bin(:)));
for i=1:length(edges)-1
    low_vol = edges(i);
    high_vol = edges(i+1);
    middle_volume(i)=(low_vol + high_vol)/2;
end
%middle_volume = round(exp(middle_volume), 1);
%set(gca,'XTickLabel',{'A','B','C'})
xticklabels(middle_volume)



%Mean overlap
b=dice_vals{1,1};
c=b>0;
d=b(c);
mean(d)
end

