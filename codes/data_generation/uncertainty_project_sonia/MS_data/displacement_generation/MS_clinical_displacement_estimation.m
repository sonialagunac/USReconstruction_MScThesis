% Sonia Laguna, ETH Zurich MSc Thesis
% Adapted from previous displacement estimation codes
% Displacement estimation and LBFGS from raw clinical data, MS sequence

clear all
close all
clc

addpath(genpath(['/scratch_net/biwidl307/sonia/USImageReconstruction-Sonia/codes/data_generation/jason_reconstruction/SoSReconstruction']))

ierationOverMean = 1;

%% Data Loading

sub_1 = 'mpBUS042/'
sub_2 = dir (['/scratch_net/biwidl307/sonia/data_original/MS_clinical/',sub_1])
for ind = [3: length(sub_2)]
     name = sub_2(ind).name;
     sub = [sub_1, name, '/'];
      data_path = ['/scratch_net/biwidl307/sonia/data_original/MS_clinical/' , sub];
      if length(dir(data_path))<=3
         continue
      end
    data_file = dir ([data_path '*MS1.mat']).name;
    load([data_path, data_file])
    opts.acq.sequence.chan_comb = [15, 25 ; 35, 45 ; 55, 65; 75, 85; 95, 105;105, 115];
    chan_comb = opts.acq.sequence.chan_comb;
    opts.postprocess.general.c = 1500; 
    %% Parameter Setting
    %mask_NF = 05e-3;                            % mask the first couple of mm, which not be taken into account for reconstruction due to nearfield effects.
    mask_NF = 0;    
    opts.postprocess.general.c_step = 5;        % SoS step size for multi iteration c = c0 + n * c_step, n = 0..N_it-1
    N_it = 1;                                   % number of iterations for SoS recon
    opts.acq.RX.depth = 0.0498; %Manually changing it for consistance with the training data
    opts = defaultPostprocessOpts(opts,'MS'); 
    opts.postprocess.sos_minus.RegularizationLambda = 1e-1; %linspace(0.02,0.2,5); % multiple regularization weights for SoS recon
    N_reg = length(opts.postprocess.sos_minus.RegularizationLambda);
    opts.postprocess.BF.noPSFalign = 1;         % PSF alignment not yet implemented for VS. Leave at 1!
    opts.postprocess.BF.psf_angles = nan;       % Because noPSFalign is 1, we have to set this to nan.
    opts.postprocess.sos_minus.limit_dtrecordings = Inf; % limit the number of delay data readings for computation efficiency.
    opts.postprocess.sos_minus.centerpathonly = 1; % use wide beams (0) or center path only (1)?
    opts.postprocess.BF.fastBF = 0;             % stores delay and apodization matrix in workspace for fast computation. 
    opts.postprocess.general.processingUnit = 'cpu'; 
    
    global sharedVariables % this is where other shared variables will be stored, e.g. RX_Apod, etc.  
    sharedVariables = [];
    
    DeltaChannel = 12;
    Ncomb = 15;
    
    %% Working with real MS data
     ntx_required = unique(chan_comb(:));   
    chan_comb_raw = chan_comb;
    clear chan_comb
    for ncc = 1:size(chan_comb_raw,1)
        chan_comb(ncc,1) = find(chan_comb_raw(ncc,1) == ntx_required);
        chan_comb(ncc,2) = find(chan_comb_raw(ncc,2) == ntx_required);
    end
    RF = RF(:,:,ntx_required);
    opts.postprocess.RFselect.TX_selection = [1:size(RF,3)]; % leave untouched
    
    %% Resample
    %RF = ch.data; %depends on the loaded dataset
    opts.postprocess.pipeline = {'ResampleRF'};
    [RF,opts] = postprocess(opts,RF); % run the processing
    
    %% Distance Load
    opts.postprocess.general.DistancePrecomp_Path = ['/scratch_net/biwidl307/sonia/USImageReconstruction-Sonia/codes/data_generation/jason_reconstruction/SoSReconstruction/']; % this is the path where I previously save Distance matrices.
    Distances_ = DistanceMatrixLoad(opts); % load the distance matrix, which will be used for beamforming and L-Matrix computation
    [RX_Apod, ElementDirectivity] = computeAperture(opts);
    
    %% Band-Pass Filtering
    RF = bandpassRF(RF,opts);
    sos_recon_saved = zeros(numel(opts.postprocess.BF.sos_z_axis),numel(opts.postprocess.BF.sos_x_axis),N_reg,N_it);
    
    %% Displacement Map Generator
    sos_init = opts.postprocess.general.c;
    opts.postprocess.BF.sos_data = opts.postprocess.general.c .* ones(numel(opts.postprocess.BF.sos_z_axis),numel(opts.postprocess.BF.sos_x_axis)); % this is the initial sos map for used for the first iteration
    opts.postprocess.pipeline = {'BF'}; % For the beamforming reconstruction iterative pipeline
    [BF,opts] = postprocess(opts,RF); % run the processing
    
    opts = createCombinations(opts,[],'customComb',chan_comb);
    opts.postprocess.pipeline = {'DispTrack'}; % For the sos reconstruction iterative pipeline
    [DT,opts] = postprocess(opts,BF); % run the processing
    Mask = createDTmask(opts);
    Mask(opts.postprocess.BF.z_axis < mask_NF,:) = nan;
    DT = DT.*Mask;
    %% Displacement Map Creation
    opts.postprocess.pipeline = {'var_net_disp_creator'}; % For the sos reconstruction iterative pipeline
    displacement_map_var_net = postprocess(opts,DT); % run the processing
    %% SoS Recon
    opts.postprocess.pipeline = {'sos_minus'}; % For the sos reconstruction iterative pipeline
    [sos_recon,opts] = postprocess(opts,DT); % run the processing
    
    bmode = plot_sos_results(BF,sos_recon,opts,1);
    
    
    %% Saving
    recon_lbfgs = sos_recon; 
    measmnts = displacement_map_var_net(:,1);
    CorrCoeff = displacement_map_var_net(:,2);
    %save([data_path, '/output.mat'], 'recon_lbfgs', 'measmnts', 'BF','RF','CorrCoeff')
    %fig = gcf;
    %saveas(fig, [data_path, 'output.png'])
end