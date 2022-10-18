%Creating masks from data points in VS
%Sonia Laguna - ETH MSc Thesis, 05/08/2022, adapted from Dieter Schweizer

clear all,
close all,
addpath('/scratch_net/biwidl307/sonia/USImageReconstruction-Sonia/codes/data_generation/uncertainty_project_Sonia/clinical/mask_generator/ndcrdspinterp')
data_path = '/scratch_net/biwidl307/sonia/data_original/VS/subjects/mat_sos/';
sub = dir(data_path);

% The files named '*_large' include the b-modes
name = 'mpBUS091_L1_large.mat'
file = [data_path,name];
i = 0;
load(file, 'bf_im')
sub_path = ['/scratch_net/biwidl307/sonia/data_original/VS/subjects/mpBUS0', name(7:8)];
sub_1 = dir(sub_path);

frames = {'02','03','04','05'};
%frames = {'02','03','04','05'}
%frames = {'03','04','05','06'};
%frames = {'07','08','09','10','11'};
%frames = {'07','08','09','10'}
%frames = {'05','06','07','08','09'}

for frame = frames
    data_path_1 = strcat(['/scratch_net/biwidl307/sonia/data_original/VS/subjects/mpBUS0', name(7:8), '/0', name(7:8),'_0'],string(frame));
    if length(dir(data_path_1))<=2
         continue
     end
    load(strcat(data_path_1, '/lesionPoints.mat'))
    i = i +1;
    bf_im_id = bf_im(1:1067,:,:,i);
    n = 100;
    % when Tension=0 the class of Cardinal spline is known as Catmull-Rom spline
    Tension=0; 
    lesionPoints = lp;
    Px = single(lesionPoints(1,:));
    Px = [Px(end-1:end), Px, Px(1:2)]; 
    Py = single(lesionPoints(2,:));
    Py = [Py(end-1:end), Py, Py(1:2)];
    
    curve = [];
    for k=1:length(Px)-3
        [XiYi]=crdatnplusoneval([Px(k),Py(k)],[Px(k+1),Py(k+1)],[Px(k+2),Py(k+2)],[Px(k+3),Py(k+3)],Tension,n);
        % % XiYi is 2D interpolated data
        curve = [curve, XiYi];
    end
    %%
    %Computing bmode
    BF = bf_im_id;
    BF = abs(hilbert(mean(BF,3))); BF = BF./max(abs(BF(:)));
    BF = 20 * log10(BF);
    pixelsize_sos =  [  2,   2].*0.0003;
    xax_sos = [-0.0189 : pixelsize_sos(2) :  0.0189];  
    zax_sos = [0:pixelsize_sos(1):0.04]; %Change btwn 0.05 and 0.04, depending on depth
    
    pixelsize_bf = [1/8, 1]*0.0003;
    xax_bf = xax_sos(1):pixelsize_bf(2):xax_sos(end);
    zax_bf = [zax_sos(1):pixelsize_bf(1):0.04];
    [X_sos,Z_sos] = meshgrid(xax_sos,zax_sos);
    [X_fine,Z_fine] = meshgrid(xax_bf,zax_bf);

    bimage = NaN * ones(size(BF));
    curve(1,:) = round((curve(1,:)  - xax_bf(1)*1000) * size(bimage,2) / (xax_bf(end)*1000 - xax_bf(1)*1000),0);
    curve(2,:) = round(curve(2,:) * size(bimage,1) / (zax_bf(end)*1000  - zax_bf(1)*1000),0);
    
    Px = round((Px  - xax_bf(1)*1000) * size(bimage,2) / (xax_bf(end)*1000 - xax_bf(1)*1000),0);
    Py = round(Py * size(bimage,1) / (zax_bf(end)*1000 - zax_bf(1)*1000),0);
    
    xmin = round(min(curve(1,:)),0);
    xmax = floor(max(curve(1,:)));
    areacoordinates = NaN* ones((xmax-xmin)+1,3);
    for ll = xmin : xmax
        cidx = find((curve(1,:) >= ll) & (curve(1,:) < ll+1));
        areacoordinates(ll-xmin+1,:) = [ll, min(curve(2,cidx)), max(curve(2,cidx))];
    end
    for jj = 1:length(areacoordinates)
        bimage(areacoordinates(jj,2):areacoordinates(jj,3),areacoordinates(jj,1)) = 1;
    end
    bimage_mask = bimage;
    bimage = bimage .* BF;
    
    %Reshaping to SoS
    sos_mask_in = interp2(X_fine,Z_fine,bimage_mask(1:1067,:),X_sos,Z_sos);
    sos_mask_re = imresize(bimage_mask, [84,64], 'bilinear');
    
    sos_mask(:,:,i) = sos_mask_in; 
    figure
    imagesc(BF)
    hold on
    plot(curve(1,:),curve(2,:),'b','linewidth',2) % interpolated data
    plot(Px,Py,'ro','linewidth',1)          % control points
    title(strcat('BF',string(frame)))
    sp2 = gca;
    colormap(sp2,gray(256))
    caxis([-70 0])
    
    end
   save([data_path,name(1:end-4),'_mask.mat'], 'sos_mask')