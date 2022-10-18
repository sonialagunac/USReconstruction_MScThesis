% Sonia Laguna ETH Zurich
% Draft version of MS dispalcement estimation and LBFGS computation
clc
clearvars;
close all hidden
addpath(genpath('/scratch_net/biwidl307/sonia/USImageReconstruction-Sonia/codes/data_generation/Generic_Data_Processing_Structure-master/Examples'));
addpath(genpath('/scratch_netbiwidl307/sonia/USImageReconstruction-Sonia/codes/data_generation/Generic_Data_Processing_Structure-master/Codes'));
addpath(genpath('/scratch_net/biwidl307/sonia/USImageReconstruction-Sonia/codes/data_generation/Generic_Data_Processing_Structure-master/'));
addpath(genpath('/scratch_net/biwidl307/sonia/SimulationModels'))
addpath(genpath('/scratch_net/biwidl307/sonia/USImageReconstruction-Sonia/codes/data_generation/SimulationModels-melTest'))
METHOD = 'MS';
distance_path = '/scratch_net/biwidl307/sonia/USImageReconstruction-Sonia/codes/data_generation/Precomp_Distance_Matrix/'
%METHOD = 'MS';

average_SoS_value = 1475;
%test = load('/scratch_net/biwidl307/sonia/data_original/phantom_mert/01_CIRS_Breast_OneHardInclusion/01_CIRS_Breast_OneHardInclusion.mat');  % load the data here
%load('/scratch_net/biwidl307/sonia/data_original/phantom_mert/newMSphantom_01.mat/','measmnts')
test = load('/scratch_net/biwidl307/sonia/data_original/fp_raw/test_raw/test_image_2.mat')
RF_i = test.ch.data;
RF = zeros(size(RF_i,1), size(RF_i,2), 128);
RF(:,:,1:115) = RF_i;
%RF = permute(RF, [1 3 2]);
chan_comb = [15, 25 ; 35, 45 ; 55, 65; 75, 85; 95, 105;105, 115]; %channels to be used in DT
Ncomb = size(chan_comb, 1);
%RF = test.RF;
opts = test.opts; % Necessary information for transducer configuration
%opts.acq.RX.depth = 40e-3;
opts.acq.RX.depth = 50e-3;
opts = defaultPostprocessOpts(opts,'MS');   % load the default opts
opts.postprocess.BF.noPSFalign = 1;    % PSF alignment (i.e. if 0 then adapted Rx aperture as in the paper: https://arxiv.org/abs/1910.05935)
opts.postprocess.BF.fastBF = 1;             % stores delay and apodization matrix in workspace for fast computation.
opts.postprocess.general.processingUnit = 'cpu'; % write 'gpu' if GPU memory is sufficient. Otherwise use 'cpu'
opts.postprocess.general.DistancePrecomp_Path = distance_path;
opts.postprocess.RFselect.TX_selection = sort(unique(chan_comb(:)));
opts.acq.TX.apod = eye(128);

RF = bandpassRF(RF,opts);
opts.postprocess.BF.sos_data = average_SoS_value * ones(numel(opts.postprocess.BF.sos_z_axis),numel(opts.postprocess.BF.sos_x_axis)); % this sos map is used for beamforming

opts.postprocess.pipeline = {'BF'}; % For the beamforming reconstruction iterative pipeline
%%
%opts.acq.Transducer.channels = 115
[BF,opts] = postprocess(opts,RF); % run the processing
%BF = RF;
bf_im_new = BF;
%% 
% % low-resolution grid for reconstruction
 pitch = 0.0003;
 Depth = 50e-3 ;
 NC = 128;
 Width = (NC-1)/2*pitch;
 pixelsize_recon_lr = [2, 2] .* pitch; % axial/lateral resolution for sos recon. grid
 % lateral axis 
 xax_recon_lr = [-Width : pixelsize_recon_lr(2) : Width];
 xax_recon_lr = xax_recon_lr-mean(xax_recon_lr);
 xax_fine = xax_recon_lr;
% % axial axis
 zax_recon_lr = [0e-3:pixelsize_recon_lr(1):Depth+0.00001];
 zax_fine = zax_recon_lr;
% % digital resolution
 NX_lr = numel(xax_recon_lr); 
 NZ_lr = numel(zax_recon_lr);

%% Getting the disp with NCC
%load('/scratch_net/biwidl307/sonia/data_original/phantom/fukuda_1_imgs.mat')
%optsN = load('/scratch_net/biwidl307/sonia/data_original/phantom/fukuda_1_imgs.mat', 'optsN');
%opts = optsN.optsN;
%load('/scratch_net/biwidl307/sonia/data_original/phantom/phantom_trial1_14_imgs.mat');
%opts = optsN;
opts.postprocess.DispTrack.threshhold = 0.2;
%bf_im_new = bf_im(:,:,:,1);
%bf_im_new = bf_im;
%bf_im_new = BF(:,:,chan_comb);
%bf_im_new = BF(:,:,1:11);
NC = 128;
pitch = 0.0003;
%opts.postprocess.DispTrack = rmfield(opts.postprocess.DispTrack,'upsample_res')
xax_sos = opts.postprocess.sos_minus.x_axis;
zax_sos = opts.postprocess.sos_minus.z_axis;
xax_fine = opts.postprocess.BF.x_axis;
zax_fine = opts.postprocess.BF.z_axis;
zax_sos = [0,0.05];
%[X_sos,Z_sos] = meshgrid(xax_sos,zax_sos);
%[X_fine,Z_fine] = meshgrid(xax_fine,zax_fine);
NX_lr = 64; 
NZ_lr = 84;
Ncomb = 6;
%zax_bf = [zax_sos(1):pixelsize_bf(1):zax_sos(end)];
%[X_sos,Z_sos] = meshgrid(xax_fine,zax_fine);
%[X_fine,Z_fine] = meshgrid(xax_bf,zax_bf);
msz = [NZ_lr, NX_lr];
opts.postprocess.pipeline = {'DispTrack'} ;
opts.postprocess.DispTrack.method = 'NCC'
[outNCC, opts] = postprocess(opts, bf_im_new);
%This outNCC has the size 2 of the fourth dim corresponding to the correlations

for iA = 1:Ncomb % downscale to suitable measrument size
    outNCC_lr(:,:,iA) = imresize(squeeze(outNCC(:,:,iA, 1)), msz, 'bilinear');
    meas_lr(:,:,iA) = imresize(squeeze(outNCC(:,:,iA, 2)), msz, 'bilinear');
end
r_meas_lr = reshape(meas_lr, [msz(1)*msz(2)*Ncomb, 1]);
%for n = 1:size(outNCC_lr,3)
%    data_dec(:,:,n) = interp2(X_fine,Z_fine,outNCC(:,:,n,1),X_sos,Z_sos); % axial displacments in smaples
%    CorrCoeff(:,:,n) = interp2(X_fine,Z_fine,meas(:,:,n,2),X_sos,Z_sos); % this matrix gives the correlation values indicating the quality of the dispTrack
%end
%% To get the disp (postprocc)
%The output is the lbfgs and the outNCC_lr_s is the orig measurements but
%slightly changed with a mask
%outNCC_lr = data_dec;
%meas_lr = CorrCoeff;
[recon_lbfgs,opts,outNCC_lr_s,corrcoeff] = sos_minus_juan(outNCC_lr, meas_lr, opts);

%r_outNCC_lr_s_orig = reshape(outNCC_lr_s, [msz(1)*msz(2)*Ncomb, 1]);
%delays(:,1) = r_outNCC_lr_s;

%% Postprocessing the displacements 1, getting the same as the one above
%opts = defaultPostprocessOpts(opts,'MS');
%sos = opts.postprocess.BF.sos_data;
BF_SoS = 1470;
sos = BF_SoS * ones(numel(opts.postprocess.BF.sos_z_axis),numel(opts.postprocess.BF.sos_x_axis)); % this sos map is used for beamforming
chan_comb = [15, 25 ; 35, 45 ; 55, 65; 75, 85; 95, 105;105, 115]; %channels to be used in DT
Ncomb = 6;
data_dec =outNCC_lr ;
CorrCoeff = meas_lr;
%data_dec = outNCC_lr_s; 
%NZ = numel(zax_fine);
NZ = 84;
NC = 128;
pitch = 0.0003;
%NX = numel(xax_fine);
NX = 64;
xax_sos = opts.postprocess.sos_minus.x_axis;
zax_sos = opts.postprocess.sos_minus.z_axis;
xax_fine = opts.postprocess.DispTrack.x_axis;
zax_fine = opts.postprocess.DispTrack.z_axis;
[X_fine,Z_fine] = meshgrid(xax_fine,zax_fine);
[X_sos,Z_sos] = meshgrid(xax_sos,zax_sos);
data_dec(isinf(data_dec))= nan;
ElementPositions = [0:NC-1].*pitch;
ElementPositions = ElementPositions-mean(ElementPositions);
% postprocessing step as in sos\_minus is needed to add the background value again

Mask = ones(size(data_dec));
%chan_comb = opts.postprocess.DispTrack.FrameCombinations;
opts.postprocess.sos_minus.f_number_mask = 0.5
f_number_mask = opts.postprocess.sos_minus.f_number_mask;
rx_aperture = [Z_sos]/f_number_mask;
roll = 0.000001;
Apod_mask = zeros(NZ,NX,NC);
for nc = 1:NC
    rx_aperture_distance = abs([X_sos]-ElementPositions(nc));
    Apod_mask(:,:,nc) =(rx_aperture_distance<(rx_aperture/2*(1-roll))) +...
        (rx_aperture_distance>(rx_aperture/2*(1-roll))).*(rx_aperture_distance<(rx_aperture/2)).* ...
        0.5.*(1+cos(2*pi/roll*(rx_aperture_distance./rx_aperture-roll/2-1/2))); % tukey apod
end
Apod_mask(isnan(Apod_mask)) = 0;
for n = 1:size(chan_comb,1)
    Mask(:,:,n) = (Apod_mask(:,:,chan_comb(n,1)).*Apod_mask(:,:,chan_comb(n,2)) > 0);
end

Mask(isnan(data_dec)) = 0;

mask = logical(Mask(:));
Mask = gather(Mask);
Mask=logical(Mask);
load('/scratch_net/biwidl307/sonia/USImageReconstruction-Melanie/codes/evaluation/Precomp_Distance_Matrix/Pitch300mu_Depth49.8mm_Width18.9mm.mat')
[ZZ,XX] = ndgrid(1:NZ,1:NX);
idx_rel1 = nan(NZ*NX,Ncomb);
idx_rel2 = nan(NZ*NX,Ncomb);
for n_comb = 1:Ncomb
    ch1 = chan_comb(n_comb,1);
    ch2 = chan_comb(n_comb,2);
    idx_temp = sub2ind([NZ,NX,NC],ZZ,XX,repmat(ch1,NZ,NX));
    idx_rel1(Mask(:,:,n_comb),n_comb) = find(ismember(IDX',idx_temp(Mask(:,:,n_comb))));
    idx_temp = sub2ind([NZ,NX,NC],ZZ,XX,repmat(ch2,NZ,NX));
    idx_rel2(Mask(:,:,n_comb),n_comb) = find(ismember(IDX',idx_temp(Mask(:,:,n_comb))));
end
idx_rel1 = reshape(idx_rel1,NZ,NX,Ncomb);
idx_rel2 = reshape(idx_rel2,NZ,NX,Ncomb);
L_1 = Distances_(:,idx_rel1(mask)) - Distances_(:,idx_rel2(mask));
L_1 = L_1';
c_mean = sum(Distances_.*sos(:)./sum(Distances_,1),1); % compute mean sos along paths, such that displacements later can be converted to s
c_mean_comb = zeros(NZ,NX,Ncomb);
c_mean_comb(mask) = (c_mean(idx_rel1(mask)) + c_mean(idx_rel2(mask)))/2;
c_mean_comb = reshape(c_mean_comb,NZ,NX,Ncomb); % accumulated speed of sound values along rays

deltaT = 2*mean(diff(zax_fine))./c_mean_comb; %Apparently its the old way(what i have with sos_minus_juan)
data_dec_s = nan(NZ,NX,Ncomb);

data_dec_s(mask) = data_dec(mask);
data_dec_s = data_dec_s.*deltaT;
outNCC_lr_s_sonia1 = data_dec_s;
data_dec_s = data_dec_s(:)

slow0 = 1./gather(sos(:));
t_homo = L_1*slow0;
tm_1 = data_dec_s(mask) + t_homo ;
outNCC_lr_s_sonia1(mask) = outNCC_lr_s_sonia1(mask) + t_homo;
msz=[84,64];
r_outNCC_lr_s_sonia1 = reshape(data_dec_s, [msz(1)*msz(2)*Ncomb, 1]);
measmnts = outNCC_lr_s_sonia1(:);


%% Postprocessing the displacements  2


%Mask = ones(size(data_dec));
%Mask(isnan(data_dec)) = 0;
%if isfield(opts.postprocess.sos_minus,'limit_dtrecordings')
%    Nrec = opts.postprocess.sos_minus.limit_dtrecordings;
%    idxx = find(Mask == 1);
%    while nnz(Mask) > Nrec
%        Mask(idxx(round(rand*(numel(idxx)-1))+1)) = 0;
%    end
%end
%mask = logical(Mask(:));

%L_TX = sparse(reshape(L_TX,NZ_NX,NZ_NX_Ncomb))';
%L_RX = sparse(reshape(L_RX,NZ_NX,NZ_NX_Ncomb))';
%L_raw = L_TX + L_RX;
L_raw =  L_1;
% Add delay values from beamforming
% deltaT = 2\*mean(diff(zax\_fine))./c\_mean\_comb(1); % old way
data_dec_s = nan(NZ,NX,Ncomb);
tm = data_dec_s;
data_dec_s(mask) = data_dec(mask);

try
    dt_mean = nan(NZ,NX,Ncomb);
    dt_mean(mask) = data_dec_s(mask);
    dt_mean = nanmean(reshape(abs(dt_mean),NZ*NX,Ncomb));
    opts.postprocess.sos_minus.DisplacementError = dt_mean;
    opts.postprocess.sos_minus.NumElInput = sum(mask(:));
    fprintf('%2.0f pixels for measured displacements\\n',opts.postprocess.sos_minus.NumElInput)
end

% data_dec_s = data_dec_s(:);
slow0 = 1./gather(sos(:));
t_homo = L_raw*slow0;
data_dec_s = reshape(data_dec_s,[84*64*6,1]);
tm(mask) = data_dec_s(mask) + t_homo;


%% Getting the LBFGS
clear all;
load('/scratch_net/biwidl307/sonia/data_original/angles/mat/mpBUS045_L1.mat')

opts.postprocess.sos_minus.mask_nearfield = 5e-3; % for masking the data in the nearfield (for MS usually 2e-3 is good, for PW nearfield effects are larger, e.g. chose 10e-3)
opts.postprocess.sos_minus.mask_farfield = 2e-3; % for masking the data in the farfield (2e-3 seems to work well)
opts.postprocess.sos_minus.mask_maskedge = 4e-3; % for masking the data at the edges (for PW 8e-3 is good, for MS 2e-3)
opts.postprocess.sos_minus.RegularizationDensityWeight = 2; % for Regularization, low values mean no Angular weighting (down to 0), high values mean strong influence
opts.postprocess.sos_minus.optimization_c1 = 1e-4; % for the gradient in lbfgs optimization algorithm
opts.postprocess.sos_minus.optimization_c2 = 1e-4; % for the gradient in lbfgs optimization algorithm
opts.postprocess.sos_minus.optimization_max_iters_lbfgs = 5000; % for the lbfgs optimization algorithm
opts.postprocess.sos_minus.smoothing = .5e-3; % blurring the reconstructed sos map by this sigma value to avoid artifacts (in m), usually set to ~.5e-3
opts.postprocess.sos_minus.RegularizationLambda = 0.001; % initial regularization weight in lambda*||D*x||_1, for MS .1 seems good, for PW smaller
opts.postprocess.sos_minus.f_number_mask = .7;
chan_comb = [15, 25 ; 35, 45 ; 55, 65; 75, 85; 95, 105;105, 115]; 
opts.postprocess.DispTrack.FrameCombinations = chan_comb;
Nimgs = size(measmnts,2);
for p =1:Nimgs
    out = measmnts(:,p)
    out(out==0) = nan ;
    NA = size(chan_comb,1);
    out = reshape(out, [84, 64, NA, 1]);
    opts.postprocess.pipeline = {'sos_minus'};
    [outputs, opts1] = sos_minus_mel(out, opts, 'processed');
    recon_lbfgs(p,:,:) = outputs;
end
clearvars out outputs p NA Nimgs chan_comb
