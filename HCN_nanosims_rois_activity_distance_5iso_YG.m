%% load raw data and calculate activity
clear all;

% cd('PC_2nd_new_sp3/mat');

N14C12 = load('14N12C.mat');
N15C12 = load('15N12C.mat');
C12 = load('12C.mat');
C13 = load('13C.mat');
O16 = load('16O.mat');
O17 = load('17O.mat');
O18 = load('18O.mat');
% H = load('1H.mat');
% D = load('2H.mat');

ESI = load('Esi.mat');

aquisition_size = size(N14C12.IM,1);
N14C12raw = N14C12.IM(2:aquisition_size-1,2:aquisition_size-1);
N15C12raw = N15C12.IM(2:aquisition_size-1,2:aquisition_size-1);
C12raw = C12.IM(2:aquisition_size-1,2:aquisition_size-1);
C13raw = C13.IM(2:aquisition_size-1,2:aquisition_size-1);
O16raw = O16.IM(2:aquisition_size-1,2:aquisition_size-1);
O17raw = O17.IM(2:aquisition_size-1,2:aquisition_size-1);
O18raw = O18.IM(2:aquisition_size-1,2:aquisition_size-1);
% Hraw = H.IM(2:aquisition_size-1,2:aquisition_size-1);
% Draw = D.IM(2:aquisition_size-1,2:aquisition_size-1);
ESIraw = ESI.IM(2:aquisition_size-1,2:aquisition_size-1);

C12img = uint8(C12raw.*(255/max(C12raw(:))));
C13img = uint8(C13raw.*(255/max(C13raw(:))));
N14C12img = uint8(N14C12raw.*(255/max(N14C12raw(:))));
N15C12img = uint8(N15C12raw.*(255/max(N15C12raw(:))));
O16img = uint8(O16raw.*(255/max(O16raw(:))));
O17img = uint8(O17raw.*(255/max(O17raw(:))));
O18img = uint8(O18raw.*(255/max(O18raw(:))));
% Himg = uint8(Hraw.*(255/max(Hraw(:))));
% Dimg = uint8(Draw.*(255/max(Draw(:))));
% ESIimg = uint8(ESIraw.*(255/max(ESIraw(:))));


% ratio process
N15gauss = imgaussfilt(N15C12raw,1);
N14gauss = imgaussfilt(N14C12raw,1);
N15ratioimg = uint8(double(N15gauss./(N15gauss+N14gauss)).*(255/double(max(N15gauss(:)./(N15gauss(:)+N14gauss(:))))));

% Dgauss = imgaussfilt(Draw,1);
% Hgauss = imgaussfilt(Hraw,1);
% Dratioimg = uint8(double(Dgauss./(Dgauss+Hgauss)).*(255/double(max(Dgauss(:)./(Dgauss(:)+Hgauss(:))))));

C12gauss = imgaussfilt(C12raw,1.5);
C13gauss = imgaussfilt(C13raw,1.5);
N14C12C12ratio = uint8(double(N14gauss./(C12gauss)).*(255/double(max(N14gauss(:)./(C12gauss(:))))));
C13ratioimg = uint8(double(C13gauss./(C13gauss+C12gauss)).*(255/double(max(C13gauss(:)./(C13gauss(:)+C12gauss(:))))));

O16gauss = imgaussfilt(O16raw,1);
O17gauss = imgaussfilt(O17raw,1);
O18gauss = imgaussfilt(O18raw,1);
O17ratioimg = uint8(double(O17gauss./(O18gauss+O17gauss+O16gauss)).*(255/double(max(O17gauss(:)./(O18gauss(:)+O17gauss(:)+O16gauss(:))))));
O18ratioimg = uint8(double(O18gauss./(O18gauss+O17gauss+O16gauss)).*(255/double(max(O18gauss(:)./(O18gauss(:)+O17gauss(:)+O16gauss(:))))));

ESIgauss = imgaussfilt(ESIraw,1.5);
N14C12ESIratio = uint8(double(N14gauss./(ESIgauss)).*(255/double(max(N14gauss(:)./(ESIgauss(:))))));
N14C12ESIratio = uint8(double(N14C12raw./(ESIraw)).*(255/double(max(N14C12raw(:)./(ESIraw(:))))));

N15ratimg = uint8(double(N15C12raw./(N15C12raw+N14C12raw)).*(255/double(max(N15C12raw(:)./(N15C12raw(:)+N14C12raw(:))))));
C13ratimg = uint8(double(C13raw./(C13raw+C12raw)).*(255/double(max(C13raw(:)./(C13raw(:)+C12raw(:))))));
O17ratimg = uint8(double(O17raw./(O18raw+O17raw+O16raw)).*(255/double(max(O17raw(:)./(O18raw(:)+O17raw(:)+O16raw(:))))));
O18ratimg = uint8(double(O18raw./(O18raw+O17raw+O16raw)).*(255/double(max(O18raw(:)./(O18raw(:)+O17raw(:)+O16raw(:))))));
% Dratimg = uint8(double(Draw./(Draw+Hraw)).*(255/double(max(Draw(:)./(Draw(:)+Hraw(:))))));

% %% plot ratio image
% 
% N15ratio = ((N15C12.IM(2:512,:))./((N15C12.IM(2:512,:))+(N14C12.IM(2:512,:))));
% Ratiomax_N = max(max(N15ratio));
% Ratiomin_N = min(min(N15ratio));
% clims = [Ratiomin_N, Ratiomax_N]; 
% figure; imshow(N15ratio, clims); colormap(jet); axis equal; axis off; colorbar('YTickLabel',{' '}); 
% export_fig('-m3', 'ion_images_N15.png')

%% rois process, extract position info, and calculate distance
rois_uncropped = imread('rois.png');
mask = rois_uncropped(:,:,3)<200;
maskprops = regionprops(mask,'BoundingBox');
rois = imcrop(rois_uncropped, maskprops.BoundingBox);

figure, imshow(rois,[]);
export_fig('-m3', 'rois_clear.png')
export_fig('-m3', 'rois_clear.svg')

red_rois = rois(:,:,1)-rois(:,:,3);
green_rois = rois(:,:,2)-rois(:,:,3);

%red = red_rois>175;
%green = green_rois>100;

%here is a way to improve the resolution to recognize ROIs
red = red_rois==255;
green = green_rois==255;

red = red(:,:,1);
green = green(:,:,1);

redprops = regionprops(red,'all');
red_roi_image_N = double(zeros(size(red)));
red_roi_image_C = double(zeros(size(red)));
red_roi_image_O17 = double(zeros(size(red)));
red_roi_image_O18 = double(zeros(size(red)));
% red_roi_image_H = double(zeros(size(red)));
all_data = [];

figure, imshow(rois,[])
hold on

% a_C13act = [];
% a_N15act = [];
% a_O17act = [];
% a_O18act = [];
% a_Dact = [];

a_positions = [];
for i = 1:size(redprops)
    holder = zeros(size(red));
    holder(redprops(i).PixelIdxList) = 1;
    roimask = imresize(holder,[aquisition_size-2 aquisition_size-2]);
    N15holder = sum(sum(N15C12raw.*roimask));
    N14holder = sum(sum(N14C12raw.*roimask)); 
    C12holder = sum(sum(C12raw.*roimask));
    C13holder = sum(sum(C13raw.*roimask));
    O16holder = sum(sum(O16raw.*roimask));
    O17holder = sum(sum(O17raw.*roimask));
    O18holder = sum(sum(O18raw.*roimask));
%     Hholder = sum(sum(Hraw.*roimask));
%     Dholder = sum(sum(Draw.*roimask));

    N15act = N15holder/(N14holder+N15holder);
    C13act = C13holder/(C13holder+C12holder);
    O17act = O17holder/(O18holder+O17holder+O16holder);
    O18act = O18holder/(O18holder+O17holder+O16holder);
%     Dact = Dholder/(Hholder+Dholder);
    
    holderN = double(holder)*N15act;
    red_roi_image_N = red_roi_image_N+holderN;    
    holderC = double(holder)*C13act;
    red_roi_image_C = red_roi_image_C+holderC;
    holderO17 = double(holder)*O17act;
    red_roi_image_O17 = red_roi_image_O17+holderO17; 
    holderO18 = double(holder)*O18act;
    red_roi_image_O18 = red_roi_image_O18+holderO18;    
%     holderH = double(holder)*Dact;
%     red_roi_image_H = red_roi_image_H+holderH;
    
%     data = [1, i, N14holder, N15holder, C12holder, C13holder, Hholder, Dholder, N15act, C13act, Dact, N15act.*100, C13act.*100, Dact.*100];
    data = [1, i, C12holder, C13holder, N14holder, N15holder, O16holder, O17holder, O18holder, C13act, N15act, O17act, O18act, C13act*100, N15act.*100, O17act.*100, O18act.*100];

    all_data = [all_data;data];
    
%     a_N15act = [a_N15act; N15act];
%     a_C13act = [a_C13act; C13act];
%     a_Dact = [a_Dact; Dact];
% %     a_O17act = [a_O17act; O17act];
% %     a_O18act = [a_O18act; O18act];
    
    roi_props = regionprops(roimask,'Centroid');
    a_positions = [a_positions; roi_props.Centroid];    
    
    t = text(redprops(i).Centroid(1),redprops(i).Centroid(2),int2str(i));
    t.FontSize = 6;
    t.Color = 'w';
end


greenprops = regionprops(green,'all');
green_roi_image_N = double(zeros(size(green)));
% green_roi_image_H = double(zeros(size(green)));
green_roi_image_C = double(zeros(size(green)));
green_roi_image_O17 = double(zeros(size(green)));
green_roi_image_O18 = double(zeros(size(green)));

% b_N15act = [];
% b_C13act = [];
% b_Dact = [];
% % b_O17act = [];
% % b_O18act = [];
b_positions = [];
for i = 1:size(greenprops)
    holder = zeros(size(green));
    holder(greenprops(i).PixelIdxList) = 1;
    roimask = imresize(holder,[aquisition_size-2 aquisition_size-2]);
    N15holder = sum(sum(N15C12raw.*roimask));
    N14holder = sum(sum(N14C12raw.*roimask)); 
    C12holder = sum(sum(C12raw.*roimask));
    C13holder = sum(sum(C13raw.*roimask));
    O16holder = sum(sum(O16raw.*roimask));
    O17holder = sum(sum(O17raw.*roimask));
    O18holder = sum(sum(O18raw.*roimask));
%     Hholder = sum(sum(Hraw.*roimask));
%     Dholder = sum(sum(Draw.*roimask));   
    
    C13act = C13holder/(C13holder+C12holder);    
    N15act = N15holder/(N14holder+N15holder);
    O17act = O17holder/(O18holder+O17holder+O16holder);
    O18act = O18holder/(O18holder+O17holder+O16holder);
%     Dact = Dholder/(Hholder+Dholder);
    
    holderN = double(holder)*N15act;
    green_roi_image_N = green_roi_image_N+holderN;
    holderC = double(holder)*C13act;
    green_roi_image_C = green_roi_image_C+holderC;
    holderO17 = double(holder)*O17act;
    red_roi_image_O17 = red_roi_image_O17+holderO17; 
    holderO18 = double(holder)*O18act;
    red_roi_image_O18 = red_roi_image_O18+holderO18; 
%     holderH = double(holder)*Dact;
%     green_roi_image_H = green_roi_image_H+holderH;
    
%     data = [2, i, N14holder, N15holder, C12holder, C13holder, Hholder, Dholder, N15act, C13act, Dact, N15act.*100, C13act.*100, Dact.*100];
    data = [2, i, C12holder, C13holder, N14holder, N15holder, O16holder, O17holder, O18holder, C13act, N15act, O17act, O18act, C13act*100, N15act.*100, O17act.*100, O18act.*100];
    
    all_data = [all_data;data];
    
%     b_N15act = [b_N15act; N15act];
%     b_C13act = [b_C13act; C13act];
%     b_Dact = [b_Dact; Dact];
% %     b_O17act = [b_O17act; O17act];
% %     b_O18act = [b_O18act; O18act];
    
    roi_props = regionprops(roimask,'Centroid');
    b_positions = [b_positions; roi_props.Centroid];
    
    t = text(greenprops(i).Centroid(1),greenprops(i).Centroid(2),int2str(i));
    t.FontSize = 6;
    t.Color = 'w';
end

export_fig('-m3', 'annotations.png')
csvwrite('data.csv',all_data)

% figure, imshow(green_roi_image_N+red_roi_image_N,[]);
% export_fig('-m3', 'single_activity_rois_ratio.png')
% export_fig('-m3', 'single_activity_rois_ratio.svg')

hold off

%%
figure, imshow(N14C12img,[])
hold on
scatter(a_positions(:,1),a_positions(:,2),'.r')
scatter(b_positions(:,1),b_positions(:,2),'.g')
export_fig('-m3', 'cell position.png')

%% export x,y coordinates

xy = [a_positions; b_positions];
data_xy = [all_data xy];
csvwrite('data_xy.csv',data_xy)


%% Find the nearest distance
distances = pdist2(a_positions,b_positions);
a_nearest = min(distances');
b_nearest = min(distances);
nearest = [a_nearest b_nearest];

raster = 19; % remember to change the raster size!!!

all_data_dist_nearest = [all_data nearest'./(512/raster)];
csvwrite('data_dist_nearest.csv',all_data_dist_nearest)

%% rois process, find the aggregate area
rois_uncropped = imread('bound.png');
mask = rois_uncropped(:,:,3)<200;
maskprops = regionprops(mask,'BoundingBox');
rois = imcrop(rois_uncropped, maskprops.BoundingBox);

figure, imshow(rois,[]);
export_fig('-m3', 'bound_paint_clear.png')

red_rois = rois(:,:,1)-rois(:,:,3);
red = red_rois>175;
red = red(:,:,1);
redprops = regionprops(red,'all');

for i = 1:size(redprops)
    holder = zeros(size(red));
    holder(redprops(i).PixelIdxList) = 1;
    roimask = imresize(holder,[aquisition_size-2 aquisition_size-2]);   
end

%% Define the electrode roi and related boundary
bd = bwboundaries(red);
bd_position = cell2mat(bd);

figure, imshow(N14C12img)
hold on
visboundaries(bd)
export_fig('-m3', 'agg_boundary.png')

%% Distance from ROIs to aggregate boundary

a_distances_bd = pdist2(a_positions, bd_position);
a_min_dist_bd = min(a_distances_bd');
b_distances_bd = pdist2(b_positions, bd_position);
b_min_dist_bd = min(b_distances_bd');

min_dist_bd = [a_min_dist_bd b_min_dist_bd];

all_data_dist_nearest_bd = [all_data_dist_nearest min_dist_bd'./(512/raster)];
csvwrite('data_dist_nearest_bound.csv', all_data_dist_nearest_bd)

%%



% imwrite(N14C12ESIratio,'cells.png')
% imwrite(N14C12C12ratio,'cells_C12.png')
% imwrite(N15ratioimg,'N15activity.png')
% imwrite(N15ratimg,'N15activitynoblur.png')
% imwrite(C13ratimg,'C13rat.png')
% imwrite(C13ratioimg,'C13blur.png')
% imwrite(C12img, 'C12.png')
% imwrite(C13img, 'C13.png')
% % imwrite(O18ratimg,'O18rat.png')
% % imwrite(O18ratioimg,'O18blur.png')
% % imwrite(O16img, 'O16.png')
% % imwrite(O18img, 'O18.png')
% imwrite(N14C12img, 'N14C12.png')
% imwrite(N15C12img, 'N15C12.png')
% %imwrite(P31img, 'P31.png')
% imwrite(Himg,'H.png')
% imwrite(Dimg,'D.png')
% imwrite(ESIimg, 'ESIimg.png')
% imwrite(Dratioimg, 'Dblur.png')
% imwrite(Dratimg, 'Drat.png')

% export_fig('annotations.png')
% figure, imshow(green_roi_image_N+red_roi_image_N,[]); colorbar('YTickLabel',{' '});
% export_fig('single_activity_N.png')
% figure, imshow(green_roi_image_H+red_roi_image_H,[]); colorbar('YTickLabel',{' '});
% export_fig('single_activity_H.png')
% figure, imshow(green_roi_image_C+red_roi_image_C,[]); colorbar('YTickLabel',{' '});
% export_fig('single_activity_C.png')
% % figure, imshow(green_roi_image_O+red_roi_image_O,[]); colorbar('YTickLabel',{' '});
% % export_fig('single_activity_O.png')
% 
% csvwrite('data.csv',all_data)
