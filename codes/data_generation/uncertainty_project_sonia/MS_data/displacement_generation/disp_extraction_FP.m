%Adapted from previous versions, displacement extraction from raw data, MS
%Recomputing displacements for fullpipeline data to obtain CCorrelation map
%Sonia Laguna, ETH Zurich July 202
clc
%clearvars;
%close all hidden
addpath(genpath('/scratch_net/biwidl307/sonia/USImageReconstruction-Sonia/codes/data_generation/Generic_Data_Processing_Structure-master/Examples'))
addpath(genpath('/scratch_net/biwidl307/sonia/USImageReconstruction-Sonia/codes/data_generation/Generic_Data_Processing_Structure-master/Codes'));
addpath(genpath('/scratch_net/biwidl307/sonia/USImageReconstruction-Sonia/codes/data_generation/Generic_Data_Processing_Structure-master/'));
addpath(genpath('/scratch_net/biwidl307/sonia/USImageReconstruction-Sonia/codes/data_generation/SimulationModels-melTest'))
addpath(genpath('/scratch_net/biwidl307/sonia/data_original/test'))
METHOD = 'MS';
distance_path = '/scratch_net/biwidl307/sonia/USImageReconstruction-Sonia/codes/data_generation/Precomp_Distance_Matrix/';

%% Define the Parameters
displacement_threshold_correction     = 0.2; % displacement threshold for the correction data
displacement_threshold_reconstruction = 0.2; % displacement threshold for LBFGS
measurement_limiter = 1e7; % orig, 1e7 specifies how many measurements will be used in recons. Set to a high value such such that all of them are used
regularization_parameter = 5e-3; % regularization for the LBGFS


chan_comb = [15, 25 ; 35, 45 ; 55, 65; 75, 85; 95, 105;105, 115]; %channels to be used in DT
Ncomb = size(chan_comb, 1);

%% Define the Matrices/Vectors to be saved
data_index = 5;
BF_SoS = 1500; %1500 for simulations, 1470 for phantom data
%% The main loop for reconstruction
test = load('/scratch_net/biwidl307/sonia/data_original/fp_raw/test_raw/test_image_1.mat')
RF_i = test.ch.data;
RF = zeros(size(RF_i,1), size(RF_i,2), 128);
RF(:,:,1:115) = RF_i;

opts = test.opts; % Necessary information for transducer configuration
opts.acq.RX.depth = 50e-3;
opts = defaultPostprocessOpts(opts,'MS')
opts.postprocess.general.c = BF_SoS;
opts.postprocess.BF.sos_data = BF_SoS * ones(numel(opts.postprocess.BF.sos_z_axis),numel(opts.postprocess.BF.sos_x_axis)); % this sos map is used for beamforming
opts.postprocess.BF.noPSFalign = 1;    % PSF alignment (i.e. if 0 then adapted Rx aperture as in the paper: https://arxiv.org/abs/1910.05935)
opts.postprocess.sos_minus.limit_dtrecordings = measurement_limiter;
opts.postprocess.DispTrack.threshhold = displacement_threshold_reconstruction;
opts.postprocess.sos_minus.RegularizationLambda  = regularization_parameter;
opts.postprocess.sos_minus.centerpathonly = 1; % use wide beams (0) or center path only (1)?
opts.postprocess.BF.fastBF = 1;             % stores delay and apodization matrix in workspace for fast computation.
opts.postprocess.general.processingUnit = 'cpu'; % write 'gpu' if GPU memory is sufficient. Otherwise use 'cpu'
opts.postprocess.general.DistancePrecomp_Path = distance_path;
opts.postprocess.RFselect.TX_selection = sort(unique(chan_comb(:)));
a = [1:128]'; b = nan(128,1);
opts.postprocess.BF.TxNumbPsfAngCombinations = [a,b];
opts.acq.TX.apod = eye(128);
opts.postprocess.sos_minus.f_number_mask = 0.5

%% Beamforming
RF = bandpassRF(RF,opts);
          
Distances_ = DistanceMatrixLoad(opts); % load the distance matrix, which will be used for beamforming and L-Matrix computation
[RX_Apod, ElementDirectivity] = computeAperture(opts);

opts.postprocess.pipeline = {'BF'}; % For the beamforming reconstruction iterative pipeline
[BF,opts] = postprocess(opts,RF); % run the processing

opts = createCombinations(opts,[],'customComb',chan_comb);
opts.postprocess.DispTrack.FrameCombinations = chan_comb;
%% Disp tracking
opts.postprocess.DispTrack.threshhold = displacement_threshold_reconstruction;
opts.postprocess.pipeline = {'DispTrack'}; % For the sos reconstruction iterative pipeline
[DT,opts] = postprocess(opts,BF); % run the processing

for iA = 1:Ncomb % downscale to suitable measrument size, from the old versions, not the same effect
    outNCC_lr_new(:,:,iA) = imresize(squeeze(DT(:,:,iA, 1)), [84,64], 'bilinear');
    meas_lr_new(:,:,iA) = imresize(squeeze(DT(:,:,iA, 2)), [84,64], 'bilinear');
end

%%
input = DT;

if isfield(opts.postprocess.sos_minus,'centerpathonly') && opts.postprocess.sos_minus.centerpathonly == 1
    CENTERpathONLY = 1; % only if the PSFs are centered axially. If this is 1, the L Matrix will take the RX path as a single line and not a wide beam.
else
    CENTERpathONLY = 0;
end

if isfield(opts.postprocess.sos_minus,'Solver') && strcmpi(opts.postprocess.sos_minus.Solver,'cvx')
    CVX = 1;
%     cvx_setup; % for the optimization
else
    CVX = 0;
end

xax_sos = opts.postprocess.sos_minus.x_axis;
zax_sos = opts.postprocess.sos_minus.z_axis;
xax_fine = opts.postprocess.BF.x_axis;
zax_fine = opts.postprocess.BF.z_axis;
NX = numel(xax_sos);
NZ = numel(zax_sos);
NX_fine = numel(xax_fine);
NZ_fine = numel(zax_fine);
NC = opts.acq.Transducer.channels; % channels of the transducer
Ncomb = size(input,3);
pitch = opts.acq.Transducer.pitch; % pitch of transducer
fs = opts.acq.RX.fs; % sampling frequency
ElementPositions = [0:NC-1].*pitch;
ElementPositions = ElementPositions-mean(ElementPositions);
sos = opts.postprocess.BF.sos_data;
TxNumbPsfAng = opts.postprocess.BF.TxNumbPsfAngCombinations;

DispComb = opts.postprocess.DispTrack.FrameCombinations;
if isfield(opts.postprocess.BF,'psf_angles') && any(~isnan(opts.postprocess.BF.psf_angles))
    psf_angles = opts.postprocess.BF.psf_angles;
    psfalign = 1;
else
    psfalign = 0;
end

[X_sos,Z_sos] = meshgrid(xax_sos,zax_sos);
[X_fine,Z_fine] = meshgrid(xax_fine,zax_fine);

%% FOR FOUCESSED BEAMS THE L MATRIX COMPUTATION IS VERY CRUDE AT THE MOMENT

fprintf('Reconstruction Step -- Arrange L-Matrix'); comp_timing = tic;
try % first try to pull variables from workspace
    TxCenterElement = evalin('base','TxCenterElement');
    RxCenterElement = evalin('base','RxCenterElement');
    L_TX = evalin('base','L_TX');
    L_RX = evalin('base','L_RX');
    
    fprintf(' loaded from base workspace \n');
catch
    %% Put together L-Matrix
    Distances_ = DistanceMatrixLoad(opts); % load the distance matrix, which will be used for beamforming and L-Matrix computation    
    % in TX
    fprintf(' in tx ');
    if CENTERpathONLY
        [L_TX, TxCenterElement] = WavefrontRelevantChannels(opts,'maxoffset',1e-9);
    else
        [L_TX, TxCenterElement] = WavefrontRelevantChannels(opts);
    end  
    % in RX
    fprintf('and in rx  ');
    if strcmpi(opts.postprocess.sos_minus.input,'FB')
        %Not the current case
    else 
        [TXDelay, TXapod_emit, NTX] = readTxParameters(opts);
        if isfield(opts.postprocess.BF,'maxRXchannels')
            max_RXchannels = opts.postprocess.BF.maxRXchannels; % only these TX channels will be used for beamforming (in PW this will be set to all channels)
        else
            max_RXchannels = NC;
        end
        if isfield(opts.postprocess.BF,'noPSFalign') && opts.postprocess.BF.noPSFalign == 1
            noPSFalign = 1;
            psf_angles = nan;
        else 
            noPSFalign = 0;
            psf_angles = opts.postprocess.BF.psf_angles;
            opts.postprocess.BF.noPSFalign = 0;
        end
        cycleperiod = 1/opts.acq.TX.TXfreq;
        ElementPositions = ElementPositions1d(opts); % obtain the element locations  
        pwapod = [];
        inputArgs = {'f_number_TX', opts.postprocess.BF.f_number_TX,... % dynamic apertures
            'f_number_RX', opts.postprocess.BF.f_number_RX,... % dynamic apertures
            'ElementPositions',  ElementPositions,... 
            'xax_sos',  opts.postprocess.BF.sos_x_axis,... % this is a coarse grid, where the distances/delays are calculated. This will be interpolated later.
            'zax_sos', opts.postprocess.BF.sos_z_axis,...
            'xax_bf', opts.postprocess.BF.x_axis,...
            'zax_bf', opts.postprocess.BF.z_axis,...
            'NC', opts.acq.Transducer.channels,...
            'cycleperiod', cycleperiod,...
            'sos', opts.postprocess.BF.sos_data,... % speed of sound map used for delay calculation
            'TXDelay', TXDelay,...
            'TXapod_emit', TXapod_emit,...
            'NTX', NTX,...
            'METHOD', opts.postprocess.sos_minus.input,...
            'max_RXchannels', max_RXchannels,...
            'psf_angles',psf_angles,...
            'noPSFalign', noPSFalign,...
            'DispComb', nan,...
            'TxNumbPsfAngCombinations', nan,...
            'pwapod',pwapod,... %% 5/2/2020 added by RR
            };

        outtemp = VariableReadGeneral('RX_Apod',opts.postprocess.general.varsavepath,opts,inputArgs{:}); % load the distance matrix, which will be used for beamforming and L-Matrix computation
        RX_Apod = outtemp{1};
        RX_Apod = reshape(full(RX_Apod),outtemp{3});
        
        for k = 1:size(RX_Apod,5)
            for m = 1:size(RX_Apod,4)
                for nc = 1:size(RX_Apod,3)
                    RX_Apod_coarse(:,:,nc,m,k) = interp2(X_fine,Z_fine,RX_Apod(:,:,nc,m,k) ,X_sos,Z_sos);
                end
            end
        end
        %%
        L_RX = zeros(NZ*NX,NZ*NX,Ncomb);
        RxCenterElement = nan(2,NZ*NX,Ncomb);
        %RXpaths = zeros(NZ*NX,NZ*NX,Ncomb,2);
        for ncomb = 1:Ncomb
            idx = opts.postprocess.BF.TxNumbPsfAngCombinations(DispComb(ncomb,:),1);
            %idx = DispComb(ncomb,:);
            if psfalign == 1
                psf_idx = find(opts.postprocess.BF.TxNumbPsfAngCombinations(DispComb(ncomb,1),2) == psf_angles);
            else
                psf_idx = 1;
            end
            rxapod_temp1 = RX_Apod_coarse(:,:,:,idx(1),psf_idx); %Commented sonia, dimension mismatch
            %rxapod_temp1 = RX_Apod_coarse(:,:, idx(1),:,psf_idx);
            rxapod_temp1 = double(rxapod_temp1(:));
            
            rxapod_temp2 = RX_Apod_coarse(:,:,:,idx(2),psf_idx);
            rxapod_temp2 = double(rxapod_temp2(:));

            if ~CENTERpathONLY
                path1 = Distances_.*rxapod_temp1';
                path2 = Distances_.*rxapod_temp2';
            end
            for nxz = 1:NZ*NX
                idx2 = (NZ*NX)*[0:(NC-1)] + nxz;
                centerch1 = round(mean(find(rxapod_temp1(idx2) > .8)));
                centerch2 = round(mean(find(rxapod_temp2(idx2) > .8)));
                if CENTERpathONLY
                    if ~isnan(centerch1) && ~isnan(centerch2)
                        L_RX(:,nxz,ncomb) = Distances_(:,idx2(centerch1)) - ...
                            Distances_(:,idx2(centerch2));
                        RxCenterElement(:,nxz,ncomb) = [centerch1 centerch2];
                    end
                else
                    sumRX1 = sum(rxapod_temp1(idx2));
                    sumRX2 = sum(rxapod_temp2(idx2));
                    if sumRX1 ~= 0 && sumRX2 ~= 0
                        L_RX(:,nxz,ncomb) = sum(full(path1(:,idx2))/sumRX1,2) -...
                            sum(full(path2(:,idx2))/sumRX2,2);
                        RxCenterElement(:,nxz,ncomb) = [centerch1 centerch2];
                    end
                end

            end
            try
                multiWaitbar('L_RX computation',ncomb/Ncomb);
                pause(.0001);
            end
        end
        try
            multiWaitbar('L_RX computation','Close');
            pause(.0001);
        end
    end
    TxCenterElement = reshape(permute(TxCenterElement,[2 3 1]),NZ,NX,Ncomb,2);
    RxCenterElement = reshape(permute(RxCenterElement,[2 3 1]),NZ,NX,Ncomb,2);
    
    assignin('base','TxCenterElement',TxCenterElement);
    assignin('base','RxCenterElement',RxCenterElement);
    assignin('base','L_TX',L_TX);
    assignin('base','L_RX',L_RX);
end


%% combine L matrix

% approxmiate the sos that was reaching a certain point in depth.
% accurately one would have to compute the mean sos fro each path, this is
% doable but takes time to copmute and the effects are probably minor given
% the noise.

c_mean_comb = cumsum(sos)./[1:NZ]';
ElementPositions = ElementPositions1d(opts);
ElementPositions(end+1) = nan; % artificially add this such that RX center elements that do not make sense can be mapped onto this
RxCenterElement(RxCenterElement==0) = numel(ElementPositions);
RxCenterElement(isnan(RxCenterElement)) = numel(ElementPositions);
deltaZ_unit = mean(diff(zax_fine));% * 1.5;
DT2delay = deltaZ_unit .* Z_sos ./ c_mean_comb .*...
    ( 1 ./ sqrt( (X_sos - mean(ElementPositions(TxCenterElement),4) ).^2 + Z_sos.^2) + ...
    1 ./ sqrt( (X_sos - mean(ElementPositions(RxCenterElement),4) ).^2 + Z_sos.^2) );
ElementPositions = ElementPositions(1:end-1);

for n = 1:size(input,3)
    data_dec(:,:,n) = interp2(X_fine,Z_fine,input(:,:,n,1),X_sos,Z_sos); % axial displacments in smaples
    CorrCoeff(:,:,n) = interp2(X_fine,Z_fine,input(:,:,n,2),X_sos,Z_sos); % this matrix gives the correlation values indicating the quality of the dispTrack
end
data_dec_pre = data_dec;
data_dec = data_dec.*DT2delay;

%% Post processing old way L matrix
Mask = ones(size(data_dec));

f_number_mask = opts.postprocess.sos_minus.f_number_mask;
rx_aperture = [Z_sos]/f_number_mask;
roll = 0.9;
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
opts.postprocess.sos_minus.f_number_mask = 0.5

load('/scratch_net/biwidl307/sonia/USImageReconstruction-Melanie/codes/evaluation/Precomp_Distance_Matrix/Pitch300mu_Depth49.8mm_Width18.9mm_all.mat')
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
L = Distances_(:,idx_rel1(mask)) - Distances_(:,idx_rel2(mask));
L = L';
c_mean = sum(Distances_.*sos(:)./sum(Distances_,1),1); % compute mean sos along paths, such that displacements later can be converted to s
c_mean_comb = zeros(NZ,NX,Ncomb);
c_mean_comb(mask) = (c_mean(idx_rel1(mask)) + c_mean(idx_rel2(mask)))/2;
c_mean_comb = reshape(c_mean_comb,NZ,NX,Ncomb); % accumulated speed of sound values along rays

data_dec_s = nan(NZ,NX,Ncomb);

data_dec_s(mask) = data_dec(mask);
tm = data_dec_s;

slow0 = 1./gather(sos(:));
t_homo_old = L*slow0;
tm(mask) = data_dec_s(mask) + t_homo_old ;
msz=[84,64];
data_dec_s_old = data_dec_s;
tm_old = tm;
mask_old = mask;
CorrCoeff_mask = CorrCoeff;
CorrCoeff_mask(~mask_old) = nan;
%% Post processing
Mask = ones(size(data_dec));
Mask(isnan(data_dec)) = 0;
if isfield(opts.postprocess.sos_minus,'limit_dtrecordings')
    Nrec = opts.postprocess.sos_minus.limit_dtrecordings;
    idxx = find(Mask == 1);
    while nnz(Mask) > Nrec
        Mask(idxx(round(rand*(numel(idxx)-1))+1)) = 0;
    end
end
mask = logical(Mask(:));

L_TX = sparse(reshape(L_TX,NZ*NX,NZ*NX*Ncomb))';
L_RX = sparse(reshape(L_RX,NZ*NX,NZ*NX*Ncomb))';
L_raw = L_TX + L_RX;
data_dec_s = nan(NZ,NX,Ncomb);
data_dec_s(mask) = data_dec(mask);

try
    dt_mean = nan(NZ,NX,Ncomb);
    dt_mean(mask) = data_dec_s(mask);
    dt_mean = nanmean(reshape(abs(dt_mean),NZ*NX,Ncomb));
    opts.postprocess.sos_minus.DisplacementError = dt_mean;
    opts.postprocess.sos_minus.NumElInput = sum(mask(:));
    fprintf('%2.0f pixels for measured displacements\\n',opts.postprocess.sos_minus.NumElInput)
end

slow0 = 1./gather(sos(:));
t_homo = L_raw*slow0;
data_dec_s = reshape(data_dec_s,[84*64*6,1]);
tm = data_dec_s + t_homo;
data_dec_s_new = data_dec_s;
tm_new = tm;