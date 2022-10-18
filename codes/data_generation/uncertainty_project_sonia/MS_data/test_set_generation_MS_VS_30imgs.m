%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate synthetic testing data
% Generates the inverse crime data for the 
% ground truth of the Virtual Source test set
% 30 images
% It also computes the L matrix, mask, linear ray measurements
% to have a file ready to use for evaluation on our network.
% Sonia Laguna - M.Sc. Thesis - ETH Zurich, adapted from Melanie Bernhardt
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load the parameters
clearvars;
close all hidden
cd '/scratch_net/biwidl307/sonia/USImageReconstruction-Sonia'
data_dir = '/scratch_net/biwidl307/sonia/data_original/test';
fname = 'IC_fromVS_toMS_30_imgs';
addpath('/scratch_net/biwidl307/sonia/USImageReconstruction-Sonia/codes/data_generation_Melanie/')
addpath('/scratch_net/biwidl307/sonia/USImageReconstruction-Sonia/codes/evaluation/')
addpath('/scratch_net/biwidl307/sonia/USImageReconstruction-Sonia/codes/evaluation/SimulationModels-melTest')
addpath(genpath(cd));
%% Set the parameters
SAVE = 'all';
PLOTS = true;
COW = false; % random cow mask instead of circles - always set to FALSE for now.

%% Set geometry parameters for L matrix and mask computation
load('Data_kwave_inclusion1520_raw.mat');
opts.acq.RX.fs = 4*8*opts.acq.TX.TXfreq;
chan_comb = [15, 25 ; 
            35, 45 ; 
            55, 65;
            75, 85;
            95, 105;
            105, 115]; % only relevant for MS
Ncomb = size(chan_comb, 1);
angles = unique(angle_comb);

% transducer element positions
pitch = 3e-4; % pitch of transducer
NC = 128; % channels of the transducer
ElementPositions = [0:NC-1].*pitch;
ElementPositions = ElementPositions-mean(ElementPositions);

% imaging region size
Depth = 50e-3; % imaging depth (For kwave)
Width = (NC-1)/2*pitch; % imaging width

% high-resolution grid for forward simulation
pixelsize_recon_hr = [1.5, 1.5] .* pitch; % axial/lateral resolution for sos recon. grid
xax_recon_hr = [-Width : pixelsize_recon_hr(2) : Width];
xax_recon_hr = xax_recon_hr-mean(xax_recon_hr);
zax_recon_hr = [0e-3:pixelsize_recon_hr(1):Depth+0.0001];
NX_hr = numel(xax_recon_hr); 
NZ_hr = numel(zax_recon_hr);
[X_hr, Z_hr] = meshgrid(xax_recon_hr, zax_recon_hr);

% low-resolution grid for reconstruction
pixelsize_recon_lr = [2, 2] .* pitch; % axial/lateral resolution for sos recon. grid 
xax_recon_lr = [-Width : pixelsize_recon_lr(2) : Width];
xax_recon_lr = xax_recon_lr-mean(xax_recon_lr);
zax_recon_lr = [0e-3:pixelsize_recon_lr(1):Depth];

NX_lr = numel(xax_recon_lr); NZ_lr = numel(zax_recon_lr);
[X_lr, Z_lr] = meshgrid(xax_recon_lr, zax_recon_lr);
% simulation grid
xax_sim = [-Width : opts.sim.grid_resolution : Width];
xax_sim = xax_sim-mean(xax_sim);
zax_sim = [0e-3:opts.sim.grid_resolution:Depth];
NX_sim = numel(xax_sim);
NZ_sim = numel(zax_sim);

opts.postprocess.sos_minus.f_number_mask = 0.5; 
opts.postprocess.sos_minus.mask_nearfield = 5e-3; % for masking the data in the nearfield (for MS usually 2e-3 is good, for PW nearfield effects are larger, e.g. chose 10e-3)
opts.postprocess.sos_minus.mask_farfield = 2e-3; % for masking the data in the farfield (2e-3 seems to work well)
opts.postprocess.sos_minus.mask_maskedge = 4e-3; % for masking the data at the edges (for PW 8e-3 is good, for MS 2e-3)

%% Construct L matrices
% FOR NOW, measurement and reconstruction grid have the same resolution
% high-resolution L matrix for forward simulation
opts.postprocess.general.DistancePrecomp_Path = '/scratch_net/biwidl307/sonia/USImageReconstruction-Sonia/codes/evaluation/Precomp_Distance_Matrix/';
xax_sos = xax_recon_hr;
zax_sos = zax_recon_hr;
if SAVE == 'all'
     try % first try to see if the Distances are already in the base workspace
            Distances_ = evalin('base','Distances_');   
            IDX = evalin('base','IDX');   
         catch
            file = sprintf('Pitch%1.0fmu_Depth%1.1fmm_Width%1.1fmm.mat', pitch*1e6, zax_recon_lr(end)*1e3, xax_recon_lr(end)*1e3);
            try % now try to relaoad it from disk
                % first check if axis are correct
		        xax_sos = xax_recon_lr;
		        zax_sos = zax_recon_lr;
                ax_loaded = load([opts.postprocess.general.DistancePrecomp_Path file],'xax_sos','zax_sos');   
                load([opts.postprocess.general.DistancePrecomp_Path file]);
            catch % if no precomputed matrix can be loaded, we need to compute it (very memory demanding)
                fprintf('\n Recomputing Distance Matrix - very memory demanding - PC might crash if not suitable');
                ElementPositions = [0:NC-1].*pitch;
                ElementPositions = ElementPositions-mean(ElementPositions);
                [Distances, IDX_lookup, IDX] = ComputeDistanceMatrix(xax_sos, zax_sos, ElementPositions);
                Distances_ = sparse(double(Distances));
                clear Distances
                save([opts.postprocess.general.DistancePrecomp_Path file],'xax_sos','zax_sos','ElementPositions','Distances_','IDX_lookup','IDX','-v7.3')
            end
        end
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
        
        [ZZ,XX] = ndgrid(1:NZ_lr,1:NX_lr);
        idx_rel1 = nan(NZ_lr*NX_lr,Ncomb);
        idx_rel2 = nan(NZ_lr*NX_lr,Ncomb);
        for n_comb = 1:Ncomb
            ch1 = chan_comb(n_comb,1);
            ch2 = chan_comb(n_comb,2);
            idx_temp = sub2ind([NZ_lr,NX_lr,NC],ZZ,XX,repmat(ch1,NZ_lr,NX_lr));
            idx_rel1(Mask_lr(:,:,n_comb),n_comb) = find(ismember(IDX',idx_temp(Mask_lr(:,:,n_comb))));
            idx_temp = sub2ind([NZ_lr,NX_lr,NC],ZZ,XX,repmat(ch2,NZ_lr,NX_lr));
            idx_rel2(Mask_lr(:,:,n_comb),n_comb) = find(ismember(IDX',idx_temp(Mask_lr(:,:,n_comb))));
        end
        idx_rel1 = reshape(idx_rel1,NZ_lr,NX_lr,Ncomb);
        idx_rel2 = reshape(idx_rel2,NZ_lr,NX_lr,Ncomb);
        L_lr = zeros(NZ_lr*NX_lr, NZ_lr*NX_lr*Ncomb,'double'); % Npixels x Nmeasurements
        L_lr(:, maskFixed_lr) = Distances_(:,idx_rel1(maskFixed_lr)) - Distances_(:,idx_rel2(maskFixed_lr));
        L_lr = L_lr';
        
        disp('LR computed')
        % HR
        % first initialize distances for all [nz,nx,nc] combinations for beamforming (and L-Matrix computation for if recon METHOD = 'MS')
        file = sprintf('Pitch%1.0fmu_Depth%1.1fmm_Width%1.1fmm_hr.mat',pitch*1e6, zax_recon_hr(end)*1e3, xax_recon_hr(end)*1e3);
        try % now try to relaoad it from disk
            % first check if axis are correct
	        xax_sos = xax_recon_hr;
	        zax_sos = zax_recon_hr;
            ax_loaded = load([opts.postprocess.general.DistancePrecomp_Path file],'xax_sos','zax_sos'); 
            load([opts.postprocess.general.DistancePrecomp_Path file]);
        catch % if no precomputed matrix can be loaded, we need to compute it (very memory demanding)
            fprintf('\n Recomputing Distance Matrix - very memory demanding - PC might crash if not suitable');
            ElementPositions = [0:NC-1].*pitch;
            ElementPositions = ElementPositions-mean(ElementPositions);
            [Distances, IDX_lookup, IDX] = ComputeDistanceMatrix(xax_sos, zax_sos, ElementPositions);
            Distances_ = sparse(double(Distances));
            clear Distances
            save([opts.postprocess.general.DistancePrecomp_Path file],'xax_sos','zax_sos','ElementPositions','Distances_','IDX_lookup','IDX','-v7.3')
        end
        
        f_number_mask = opts.postprocess.sos_minus.f_number_mask;
        rx_aperture = [Z_hr]/f_number_mask;
        roll = 0.000001;
        Apod_mask = zeros(NZ_hr,NX_hr,NC);
        for nc = 1:NC
            rx_aperture_distance = abs([X_hr]-ElementPositions(nc));
            Apod_mask(:,:,nc) =(rx_aperture_distance<(rx_aperture/2*(1-roll))) +...
                (rx_aperture_distance>(rx_aperture/2*(1-roll))).*(rx_aperture_distance<(rx_aperture/2)).* ...
                0.5.*(1+cos(2*pi/roll*(rx_aperture_distance./rx_aperture-roll/2-1/2))); % tukey apod
        end
        Apod_mask(isnan(Apod_mask)) = 0;
        for n = 1:Ncomb
		disp(n)
		disp(chan_comb(n,1))
		disp(chan_comb(n,2))
            Mask_hr(:,:,n) = (Apod_mask(:,:,chan_comb(n,1)).*Apod_mask(:,:,chan_comb(n,2)) > 0);
        end
        Mask_hr(zax_recon_hr < opts.postprocess.sos_minus.mask_nearfield,:,:) = 0;
        Mask_hr(zax_recon_hr > (zax_recon_hr(end) - opts.postprocess.sos_minus.mask_farfield),:,:) = 0;
        Mask_hr(:,xax_recon_hr < (xax_recon_hr(1)  + opts.postprocess.sos_minus.mask_maskedge),:) = 0;
        Mask_hr(:,xax_recon_hr > (xax_recon_hr(end)-opts.postprocess.sos_minus.mask_maskedge),:) = 0;
        maskFixed_hr = logical(Mask_hr(:));
        Mask_hr = gather(Mask_hr);
        
        [ZZ,XX] = ndgrid(1:NZ_hr,1:NX_hr);
        idx_rel1 = nan(NZ_hr*NX_hr,Ncomb);
        idx_rel2 = nan(NZ_hr*NX_hr,Ncomb);
        for n_comb = 1:Ncomb
            ch1 = chan_comb(n_comb,1);
            ch2 = chan_comb(n_comb,2);
            idx_temp = sub2ind([NZ_hr,NX_hr,NC],ZZ,XX,repmat(ch1,NZ_hr,NX_hr));
            idx_rel1(Mask_hr(:,:,n_comb),n_comb) = find(ismember(IDX',idx_temp(Mask_hr(:,:,n_comb))));
            idx_temp = sub2ind([NZ_hr,NX_hr,NC],ZZ,XX,repmat(ch2,NZ_hr,NX_hr));
            idx_rel2(Mask_hr(:,:,n_comb),n_comb) = find(ismember(IDX',idx_temp(Mask_hr(:,:,n_comb))));
        end
        idx_rel1 = reshape(idx_rel1,NZ_hr,NX_hr,Ncomb);
        idx_rel2 = reshape(idx_rel2,NZ_hr,NX_hr,Ncomb);
        L_hr = zeros(NZ_hr*NX_hr, NZ_hr*NX_hr*Ncomb,'double'); % Npixels x Nmeasurements
        L_hr(:, maskFixed_hr) = Distances_(:,idx_rel1(maskFixed_hr)) - Distances_(:,idx_rel2(maskFixed_hr));
        L_hr = L_hr';
        disp('HR computed')
	maskFixed = maskFixed_lr;
end

%% Simulate measurements
% generate ground-truth images
Nimgs = 30;

imsz = [NZ_hr, NX_hr];
sim_imsz = [NZ_sim, NX_sim];
rsz = [NZ_lr, NX_lr];
msz = [NZ_lr, NX_lr];
NA = Ncomb;

imgs_gt = zeros([rsz, Nimgs]);
sos_sim = zeros([sim_imsz, Nimgs]);
sos_noincl_sim = zeros([sim_imsz, Nimgs]);
measmnts = zeros([msz(1)*msz(2)*NA, Nimgs]);
depth = zeros([Nimgs, 1]);
foreground = zeros([Nimgs, 1]);
background = zeros([Nimgs, 1]);
[X_hr, Z_hr] = meshgrid(xax_recon_hr, zax_recon_hr);
[X_lr, Z_lr] = meshgrid(xax_recon_lr, zax_recon_lr);
[X_sim, Z_sim] = meshgrid(xax_sim, zax_sim);

img_idx = 1;
%%
%Loading GT imgage
 load('/scratch_net/biwidl307/sonia/data_original/VS/train_VS_15comb_IC_30.mat', 'imgs_gt')

for i = 1 : Nimgs   
    img = 1./imgs_gt(:,:,i);
    img_big = imresize(img, imsz, 'bilinear');
    img_lr = imresize(img, rsz, 'bilinear');
    if SAVE == 'all'
        d_hr = L_hr * (1./img_big(:));
        d_hr(d_hr==0) = nan;
        d_hr_reshaped = reshape(d_hr, [NZ_hr, NX_hr, Ncomb]);
        d_lr_reshaped = nan(NZ_lr, NX_lr, Ncomb);
        for iA = 1:Ncomb
            d_lr_reshaped(:,:,iA) = interp2(X_hr, Z_hr, d_hr_reshaped(:,:,iA), X_lr, Z_lr);
        end
        d_lr = d_lr_reshaped(:);
        measmnts(:,i) = d_lr;
    end
    imgs_gt(:,:, i) = 1./img_lr; % note that we convert to slowness after interpolation
end
imgs_gt = imgs_gt(:,:, 1:Nimgs);
%% Compute SVD and fixed mask
if SAVE == 'all'
    L_fact = svds(L_lr, 1);
    L = L_lr / L_fact;
    % SVD of L
    tic; [U, S, V] = svd(L, 'econ'); toc;
    % convert all quantities to single precision
    L_fact = single(L_fact); L = single(L);
    U = single(U); S = single(S); V = single(V);
    %measmnts = single(measmnts); 
end

imgs_gt = single(imgs_gt);

%% Save data
%if SAVE == 'all'
%    save(fullfile(data_dir, fname), 'imgs_gt', 'measmnts', 'L', 'L_fact', ...
%        'maskFixed', 'U', 'S', 'V','sos_sim', 'sos_noincl_sim');
%else
%    save(fullfile(data_dir, fname), 'imgs_gt', 'sos_sim', 'sos_noincl_sim', '-v7'); 
%end