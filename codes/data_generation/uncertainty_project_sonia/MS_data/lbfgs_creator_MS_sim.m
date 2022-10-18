% Sonia Laguna ETH Zurich
% Codes to generate LBFGS of MS simulated data
% Adapted from Mélanie Bernhardt
%% Load data
clear all
addpath(genpath('/scratch_net/biwidl307/sonia/SimulationModels'))
addpath(genpath('/scratch_net/biwidl307/sonia/USImageReconstruction-Sonia/codes/evaluation/Generic_Data_Processing_Structure-master\'))
addpath(genpath('/scratch_net/biwidl307/sonia/USImageReconstruction-Sonia/codes/evaluation'))
save_dir = '/scratch_net/biwidl307/sonia/USImageReconstruction-Sonia/runs/4_ICFP_reg1e5_tau5_32filt_15lay_VS/eval-vn-120000/LBFGS_checks';
data_dir = '/scratch_net/biwidl307/sonia/USImageReconstruction-Sonia/runs/4_ICFP_reg1e5_tau5_32filt_15lay_VS/eval-vn-120000/';
lbfgs_dir = '/scratch_net/biwidl307/sonia/USImageReconstruction-Sonia/runs/lbfgs';
%%
filenames = [
%"test-syn-0.0-0.2-patchy-val_MS_6comb.mat",
%"test-syn-0.0-0.4-patchy-val_MS_6comb.mat",
%"test-syn-0.0-0.6-patchy-val_MS_6comb.mat",
%"test-syn-0.0-0.7-patchy-val_MS_6comb.mat",
%"test-syn-0.0-0.0-test-fullpipeline_val_64_6comb_64_imgs.mat",
%"test-syn-0.0-0.0-test-fullpipeline_testset_6comb_32_imgs.mat",
%"test-syn-0.0-0.4-patchy-testset_ideal_MS_32_imgs.mat"
%"test-syn-0.0-0.9-patchy-testset_ideal_MS_32_imgs.mat",
%"test-syn-0.0-0.5-patchy-testset_ideal_MS_32_imgs.mat",
%"test-syn-0.0-0.1-patchy-testset_ideal_MS_32_imgs.mat"
%"test-syn-0.0-0.0-patchy-new_test_syn.mat",
%"test-syn-0.0-0.0-patchy-val_MS_6comb.mat",
%"test-syn-0.0-0.0-test-fullpipeline_big_32.mat"
"test-syn-0.0-0.0-test-train_VS_15comb_fullpipeline_30.mat",
"test-syn-0.0-0.1-patchy-train_VS_15comb_IC_30.mat",
%"test-syn-0.0-0.5-patchy-train_VS_15comb_IC_30.mat"
    ];
PLOT = false;
        
%% Set the params for the reconstruction
chan_comb = [15, 25 ; 
            35, 45 ; 
            55, 65;
            75, 85;
            95, 105;
            105, 115];
load('/scratch_net/biwidl307/sonia/USImageReconstruction-Sonia/codes/evaluation/Data_kwave_inclusion1520_raw.mat')
opts.postprocess.general.DistancePrecomp_Path = 'scratch_net/biwidl307/sonia/USImageReconstruction-Sonia/codes/data_generation/Precomp_Distance_Matrix/';
aperture = opts.acq.Transducer.channels * opts.acq.Transducer.pitch;
% transducer element positions
NC = opts.acq.Transducer.channels; % channels of the transducer
pitch = opts.acq.Transducer.pitch; % pitch of transducer
ElementPositions = [0:NC-1].*pitch;
ElementPositions = ElementPositions-mean(ElementPositions);
Width = (NC-1)/2*pitch; % imaging width
Depth = opts.acq.RX.depth; % imaging depth (For kwave)
pixelsize_recon_lr = [2, 2] .* pitch; % axial/lateral resolution for sos recon. grid
pixelsize_sos = [  2,   2].*opts.acq.Transducer.pitch; % ax/lat resolution for the sos recon
pixelsize_bf =  [1/8, 1/2].*opts.acq.Transducer.pitch; % ax/lat resolution for the bf/dispTrack
xax_sos = [-Width : pixelsize_sos(2) : Width+0.0001];  
xax_sos = xax_sos-mean(xax_sos);
zax_sos = [0e-3:pixelsize_sos(1):Depth];
xax_recon_lr = [-Width : pixelsize_recon_lr(2) : Width];
xax_recon_lr = xax_recon_lr-mean(xax_recon_lr);
zax_recon_lr = [0e-3:pixelsize_recon_lr(1):Depth];
xax_bf = [xax_sos(1):pixelsize_bf(2):xax_sos(end)];  
xax_bf = xax_bf-mean(xax_bf);
zax_bf = [zax_sos(1):pixelsize_bf(1):zax_sos(end)];
opts.postprocess.sos_minus.x_axis = xax_recon_lr; % define the axis of the sos recon grid
opts.postprocess.sos_minus.z_axis = zax_recon_lr; % define the axis of the sos recon grid
opts.postprocess.DispTrack.x_axis = xax_recon_lr; % define the axis of the sos recon grid
opts.postprocess.DispTrack.z_axis = zax_recon_lr; % define the axis of the sos recon grid
opts.postprocess.BF.x_axis = xax_recon_lr; % define the axis of the sos recon grid
opts.postprocess.BF.z_axis = zax_recon_lr; % define the axis of the sos recon grid
sos_initial = opts.acq.sequence.c .* ones(numel(zax_recon_lr),numel(xax_recon_lr)); % this is the initial sos map for used for the first iteration
opts.postprocess.sos_minus.input = 'MS'; %'MS' for multistatic based sos reconstruction 
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
opts.postprocess.DispTrack.FrameIndex = 1:opts.acq.Transducer.channels;
opts.postprocess.DispTrack.FrameCombinations = chan_comb;
NA = size(chan_comb,1);
opts.postprocess.sos_minus.RegularizationLambda = 0.001;
%%
fileID = fopen(fullfile(save_dir, 'RMSE_check.txt'),'w');
for idx=1:numel(filenames)
    filename = filenames(idx);
    m = load(fullfile(data_dir, sprintf('%s', filename)));
    [Nimgs,nx, ny] = size(m.init_img) ;
        for p = 1:Nimgs
            disp(p);
            try
                gt = reshape(1./m.gt_slowness(p,:,:), [64, 84])'; 
            catch
                gt = reshape(m.gt_sos(p,:,:), [64, 84])';
            end
            opts.postprocess.BF.sos_data = mean(gt(:)).* ones(numel(zax_recon_lr),numel(xax_recon_lr));
            out = m.din(:,p);
            out(out==0) = nan ;
            out = reshape(out, [84, 64, NA, 1]);
            opts.postprocess.pipeline = {'sos_minus'};
            [outputs, opts1] = sos_minus_mel(out, opts, 'processed');
            if PLOT == true
                subplot(1,2,1);
                imagesc(outputs); colorbar;
            end
            recon_lbfgs(p,:,:) = outputs;
        end
        save(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)), 'recon_lbfgs');
end
 