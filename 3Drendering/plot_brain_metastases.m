function plot_brain_metastases(pth,datafile_gt,datafile_pred,datafile_mri)
% Plot 3D figure of ground-truth (GT) and predicted metastases with MR main
% slices (sagittal, coronal and axial) as reference of the patient's
% anatomy. Maximal perceputally distinct colours chosen for distinguishing
% the different metastases present in a patient.

%-------------------------------------------------------------------------
% Code written by Cristina Almagro-Pérez,2022, ETH University (Zurich).
%--------------------------------------------------------------------------

% Variables to be given by the user:
% 'pth': path where the files are saved. It is also the output path where the image will be saved.
% 'datafile_gt': .mat file containing the 3D ground truth binary segmentation mask.
%                inside this .mat file the variable should be named gt
% 'datafile_pred': .mat file containing the 3D predicted binary segmentation mask.
%                 inside this .mat file the variable should be named pred
% 'datafile_mri': 3D MR image of the patient in .mat format
%                 inside this .mat file the variable should be named img
%% Load files and pre-process
load(datafile_gt,'gt');% Load GT metastases
load(datafile_pred,'pred');% Load predicted metatases
load(datafile_mri,'img'); % Load MRI image

nm_gt = ['fig_', datafile_gt(1:end-4)];% name for output GT figure
nm_pred = ['fig_', datafile_pred(1:end-4)];% name for output predicted figure


% Label components
gt = bwlabeln(gt);
pred = bwlabeln(pred);
num_met_gt = max(gt(:));
num_met_pred = max(pred(:));

sz=1;sx=1;szsx=[sz sx]; % um/pixel in volume x/y dimension; % um/pixel in volume z dimension
C=distinguishable_colors(max(num_met_gt,num_met_pred)+5);
%% Make plot GT
h=figure;
for kk=1:num_met_gt
volbody=gt==kk;
if kk == 1
    Cmet=C(kk+5,:); % colour of the 3D-rendered metastasis
    voltop=volbody;voltop(1,1,end)=1;
    volbot=volbody;volbot(end,end,1)=1;
else
    Cmet=C(kk+5,:); % colour of the 3D-rendered metastasis
    voltop=volbody;voltop(:,:,1:end-1)=0;
    volbot=volbody;volbot(:,:,2:end)=0;
end
fprintf('\n plotting metastasis %d',kk);
% plot and save
hold on
h = make_3D_surface_no_outline(volbody,voltop,volbot,Cmet,[],szsx);
end
disp('plotted metastases')
%%Add images
add_images(img)
savefig(h,[pth,nm_gt,'_mri_slices','.fig']);

%% Make plot pred
h=figure;
for kk=1:num_met_pred
volbody=pred==kk;
if kk == 1
    Cmet=C(kk+5,:);
    voltop=volbody;voltop(1,1,end)=1;
    volbot=volbody;volbot(end,end,1)=1;
else
    Cmet=C(kk+5,:);
    voltop=volbody;voltop(:,:,1:end-1)=0;
    volbot=volbody;volbot(:,:,2:end)=0;
end
fprintf('\n plotting metastasis %d',kk);
% plot and save
hold on
h = make_3D_surface_no_outline(volbody,voltop,volbot,Cmet,[],szsx);
end
disp('plotted metastases')

% Add images
add_images(img)
savefig(h,[pth,nm_pred,'_mri_slices','.fig']);
%% Auxiliary functions 
function h = make_3D_surface_no_outline(volbody,voltop,volbot,volmap,cmap,szsx)
    
    xx0=1;zz0=1;
    sz=szsx(1);
    sx=szsx(2);
    h = figure;
 
    hold on;
    if size(volmap,1)==size(volbody,1) % map is 3D object
        voltopmap=volmap;voltopmap(:,:,1:end-1)=0;
        volbotmap=volmap;volbotmap(:,:,2:end)=0;
        if size(szsx,2)>1 % axes are defined explicitly
            
            [xg,yg,zg]=meshgrid(1:size(volbody,2),1:size(volbody,1),1:size(volbody,3));
            xg=xg*xx0*sx; % change from subsampled volume to mm
            yg=yg*xx0*sx;
            zg=zg*zz0*sz;
            
            patch(isosurface(xg,yg,zg,volbody,volmap),'FaceColor','flat','EdgeColor','none')
            patch(isosurface(xg,yg,zg,volbot,volbotmap),'FaceColor','flat','EdgeColor','none')
        	patch(isosurface(xg,yg,zg,voltop,voltopmap),'FaceColor','flat','EdgeColor','none')
            
        else
            patch(isosurface(volbody,volmap),'FaceColor','flat','EdgeColor','none')
            patch(isosurface(volbot,volbotmap),'FaceColor','flat','EdgeColor','none')
        	patch(isosurface(voltop,voltopmap),'FaceColor','flat','EdgeColor','none')
            
        end
        colormap(cmap)
        
    else % color is flat
        if size(szsx,2)>1 % axes are defined explicitly
            sz=szsx(1);
            sx=szsx(2);
            [xg,yg,zg]=meshgrid(1:size(volbody,2),1:size(volbody,1),1:size(volbody,3));
            xg=xg*xx0*sx;
            yg=yg*xx0*sx;
            zg=zg*zz0*sz;
            
            patch(isosurface(xg,yg,zg,volbody),'FaceColor',volmap,'EdgeColor','none')
            patch(isosurface(xg,yg,zg,voltop),'FaceColor',volmap,'EdgeColor','none')
            patch(isosurface(xg,yg,zg,volbot),'FaceColor',volmap,'EdgeColor','none')
           
        else
            patch(isosurface(volbody),'FaceColor',volmap,'EdgeColor','none')
            patch(isosurface(voltop),'FaceColor',volmap,'EdgeColor','none')
            patch(isosurface(volbot),'FaceColor',volmap,'EdgeColor','none')
            
        end
    end
plot_settings(1)
end

function plot_settings(ll)
if nargin==0;ll=0;end
    
    camva(9.72)
    set(gcf,'color','w');
    set(gca,'color','w');
    axis on;
    box on;
    set(gca,'xtick',[],'ytick',[],'ztick',[])
    xlabel('');zlabel('');ylabel('');

    view(-8,14)

if ll
    delete(findall(gcf,'Type','light'))
    set(gca,'CameraViewAngleMode','Manual')
    camlight
    lighting gouraud
    lightangle(-8,14)
end
axis tight
daspect([1 1 1]);
material dull
end

function add_images(img)
% Select central axial, sagital and coronal views

sz = size(img);central=round(sz./2);
axial = img(:,:,central(3));
coronal=(squeeze(img(central(1),:,:))); coronal = imrotate(coronal,-90); coronal=flip(coronal);
sagital=(squeeze(img(:,central(2),:)));sagital = imrotate(sagital,90);
% NOTE: ADJUST THE ROTATION ACCORDING TO YOUR PATIENT IF SECTIONS NOT
% DISPLAYED PROPERLY. 

%% Insert images
x=xlim;y=ylim;z=zlim;

% Insert axial view
a=[0    x(2)];
b=[0    y(2)];
c=central(3);
%axial = imrotate(axial,180);
xImage = [a(1) a(2);a(1) a(2)];   % The x data for the image corners
yImage = [b(1) b(1);b(2) b(2)];             % The y data for the image corners
zImage = [c c; c c];   % The z data for the image corners
s = surf(xImage,yImage,zImage,...    % Plot the surface
'CData',axial,...
'FaceColor','texturemap');

% Insert sagital view
a=central(2);
b=[0  x(2)];
c=[0  z(2)];
sagital = imrotate(sagital,180);
sagital = flip(sagital, 2);
xImage = [a a;a a];   % The x data for the image corners
yImage = [b(1) b(2);b(1) b(2)];             % The y data for the image corners
zImage = [c(1) c(1);c(2) c(2)];   % The z data for the image corners
s=surf(xImage,yImage,zImage,...    % Plot the surface
'CData',sagital,...
'FaceColor','texturemap');
   

% Insert coronal view
a=[0  x(2)];
b=central(1);
c=[0  z(2)];
coronal = imrotate(coronal,180);
%coronal = flip(coronal, 2);
xImage = [a(1) a(2);a(1) a(2)];   % The x data for the image corners
yImage = [b b;b b];             % The y data for the image corners
zImage = [c(1) c(1);c(2) c(2)];   % The z data for the image corners
s=surf(xImage,yImage,zImage,...    % Plot the surface
'CData',coronal,...
'FaceColor','texturemap');
colormap('gray');
axis tight
end
end
