%% Sonia Laguna - ETH Zurich
%% Getting the LBFGS of the clinical VS data

clear all;
load('/scratch_net/biwidl307/sonia/data_original/MS_clinical/mpBUS042_L1.mat')

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
