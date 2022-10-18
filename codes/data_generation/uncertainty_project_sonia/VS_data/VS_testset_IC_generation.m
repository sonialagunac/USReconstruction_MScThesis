%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generates inverse crime data test set for VS from fullpipeline ground
% truth
% Generate synthetic training data set using Richard's new L matrix.
% Using code from Valeriy Vishnevskiy and Richard Rau
% Adapted from  Melanie Bernhardt 
% Sonia Laguna - M.Sc. Thesis - ETH Zurich
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clearvars;
close all hidden
clear all
cd '/scratch_net/biwidl307/sonia/USImageReconstruction-Sonia/codes/data_generation_Sonia/revised_jason_codes';
data_dir = './';
addpath '/scratch_net/biwidl307/sonia/USImageReconstruction-Sonia/codes/data_generation_Melanie'
addpath(cd);
opts.postprocess.general.DistancePrecomp_Path = '/scratch_net/biwidl307/sonia/USImageReconstruction-Sonia/codes/evaluation/Precomp_Distance_Matrix/';
METHOD = 'VS';
Nimgs = 30;
fname = 'train_VS_15comb_IC_30.mat';
include_smooth = true;
fix = true;
max_number_incl = 1;
%% geometry parameters
% angles
chan_comb = [16, 28;
    28, 40;
    40, 52;
    52, 64;
    64, 76;
    76, 88;
    88, 100;
    100, 112;
    22, 34;
    34, 46;
    46, 58;
    58, 70;
    70, 82;
    82, 94;
    94, 106]; 

Ncomb = size(chan_comb, 1);


%% Transducer element positions
NC = 128; % channels of the transducer
pitch = 3e-4; % pitch of transducer
ElementPositions = [0:NC-1].*pitch;
ElementPositions = ElementPositions-mean(ElementPositions);

%% Imaging region size
% Depth = 38e-3; % imaging depth (For phantom)
Depth = 50e-3; % imaging depth (For kwave)
Width = (NC-1)/2*pitch; % imaging width

%% High-resolution grid for forward simulation

pixelsize_recon_hr = [2,2] .* pitch; % axial/lateral resolution for sos recon. grid

% lateral axis
xax_recon_hr = [-Width : pixelsize_recon_hr(2) : Width];
xax_recon_hr = xax_recon_hr-mean(xax_recon_hr);

% axial axis
zax_recon_hr = [0e-3:pixelsize_recon_hr(1):Depth+0.0001];

% digital resolution
NX_hr = numel(xax_recon_hr); 
NZ_hr = numel(zax_recon_hr);
[X_hr, Z_hr] = meshgrid(xax_recon_hr, zax_recon_hr);

%% Low-resolution grid for reconstruction
pixelsize_recon_lr = [2, 2] .* pitch; % axial/lateral resolution for sos recon. grid
% lateral axis
% xax_recon_lr = [-Width : pixelsize_recon_lr(2) : Width+0.00001];  
xax_recon_lr = [-Width : pixelsize_recon_lr(2) : Width];
xax_recon_lr = xax_recon_lr-mean(xax_recon_lr);
% axial axis
zax_recon_lr = [0e-3:pixelsize_recon_lr(1):Depth];
% digital resolution
NX_lr = numel(xax_recon_lr);
NZ_lr = numel(zax_recon_lr);
[X_lr, Z_lr] = meshgrid(xax_recon_lr, zax_recon_lr);

%% SoS Postprocess Parameters
opts.postprocess.sos_minus.f_number_mask = 0.5; % dynamic aperture
opts.postprocess.sos_minus.mask_nearfield = 5e-3; % for masking the data in the nearfield (for MS usually 2e-3 is good, for PW nearfield effects are larger, e.g. chose 10e-3)
opts.postprocess.sos_minus.mask_farfield = 2e-3; % for masking the data in the farfield (2e-3 seems to work well)
opts.postprocess.sos_minus.mask_maskedge = 4e-3; % for masking the data at the edges (for PW 8e-3 is good, for MS 2e-3)
%% Load L Matrix
load('/scratch_net/biwidl307/sonia/USImageReconstruction-Melanie/codes/evaluation/jason_reconstruction/SoSReconstruction/data_generation_Sonia/revised_jason_codes/L_VS_revised.mat')
L = L_raw;
L_lr = L;
L_hr = L;
%%
f_number_mask = opts.postprocess.sos_minus.f_number_mask;
rx_aperture = [Z_lr]/f_number_mask;
roll = 0.000001;
Apod_mask = zeros(NZ_lr,NX_lr,NC);
for nc = 1:NC
    rx_aperture_distance = abs([X_lr]-ElementPositions(nc));
    Apod_mask(:,:,nc) =(rx_aperture_distance<(rx_aperture/2*(1-roll))) +...
        (rx_aperture_distance>(rx_aperture/2*(1-roll))).*(rx_aperture_distance<(rx_aperture/2)).* ...
        0.5.*(1+cos(2*pi/roll*(rx_aperture_distance./rx_aperture-roll/2-1/2))); % tukey apod
end
Apod_mask(isnan(Apod_mask)) = 0;
for n = 1:Ncomb
    disp(n)
    disp(chan_comb(n,1))
    disp(chan_comb(n,2))
    Mask_lr(:,:,n) = (Apod_mask(:,:,chan_comb(n,1)).*Apod_mask(:,:,chan_comb(n,2)) > 0);
end
Mask_lr(zax_recon_lr < opts.postprocess.sos_minus.mask_nearfield,:,:) = 0;
Mask_lr(zax_recon_lr > (zax_recon_lr(end) - opts.postprocess.sos_minus.mask_farfield),:,:) = 0;
Mask_lr(:,xax_recon_lr < (xax_recon_lr(1)  + opts.postprocess.sos_minus.mask_maskedge),:) = 0;
Mask_lr(:,xax_recon_lr > (xax_recon_lr(end)-opts.postprocess.sos_minus.mask_maskedge),:) = 0;
maskFixed_lr = logical(Mask_lr(:));
Mask_lr = gather(Mask_lr);
maskFixed = Mask_lr;

%% simulate measurements
% % generate ground-truth images
imsz = [NZ_hr, NX_hr];
rsz = [NZ_lr, NX_lr];
msz = [NZ_lr, NX_lr];
NA = Ncomb;
imgs_gt = zeros([rsz, Nimgs]);
measmnts = zeros([msz(1)*msz(2)*NA, Nimgs]);

[n1, n2] = ndgrid(linspace(-1,1, imsz(1)), linspace(-1,1,imsz(2)));


%% Loading FP data
FP_dir = '/scratch_net/biwidl307/sonia/data_original/VS/train_VS_15comb_fullpipeline_30.mat'
fp = load(FP_dir, 'imgs_gt');

for i = 1 : Nimgs
     img = 1./fp.imgs_gt(:,:,i);
     % for SoS data
     img_lr = imresize(img, rsz, 'bilinear');
     imgs_gt(:,:,i) = 1./img_lr; % note that we convert to slowness after interpolation
     d_hr = L_hr * (1./img(:));
     d_hr(d_hr == 0) = nan;
     d_hr_reshaped = reshape(d_hr, [NZ_hr, NX_hr, Ncomb]);
     d_lr_reshaped = nan(NZ_lr, NX_lr, Ncomb);
     for iA = 1:Ncomb
         d_lr_reshaped(:,:,iA) = interp2(X_hr, Z_hr, d_hr_reshaped(:,:,iA), X_lr, Z_lr);
     end
     d_lr = d_lr_reshaped(:);
     clf; 
     measmnts(:,i) = d_lr; 
 end
 
%% Compute SVD
 L_fact = svds(L_lr, 1);
 L = L_lr / L_fact;
 % SVD of L
 tic; [U, S, V] = svd(L, 'econ'); toc;
 % convert all quantities to single precision
 L_fact = single(L_fact); L = single(L);
 U = single(U); S = single(S); V = single(V);
 
 t = diag(S);
 t(t<=0.1) = 0 ;
 t(t>0) = 1./ t(t>0) ;
 S2 = diag(t);
 Linv = V * S2 * U';
 measmnts = single(measmnts); imgs_gt = single(imgs_gt);
 
%% Save data
% save(fullfile(data_dir, fname), 'imgs_gt', 'measmnts', 'L', 'Linv', 'L_fact', 'maskFixed', 'U', 'S', 'V', '-v7');
