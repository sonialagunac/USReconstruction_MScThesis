%% Load data
clear all
%addpath(genpath('/Users/melaniebernhardt/Desktop/MScThesis/SimulationModels'));
addpath(genpath('/scratch_net/biwidl307/sonia/SimulationModels'))
%addpath(genpath('/Users/melaniebernhardt/Desktop/MScThesis/Generic_Data_Processing_Structure-OGtests/'));
addpath(genpath('/scratch_net/biwidl307/sonia/USImageReconstruction-Melanie/code/evaluation/Generic_Data_Processing_Structure-master/'))
%save_dir = '/Volumes/MelSSD/runs/1_10l/eval-vn-120000/LBFGS_checks/';
save_dir = '/scratch_net/biwidl307/sonia/USImageReconstruction-Melanie/runs/32_mix_ideal_20l_reg/eval-vn-120000/LBFGS_checks';
%data_dir = '/Volumes/MelSSD/runs/1_10l/eval-vn-120000/' ;
data_dir = '/scratch_net/biwidl307/sonia/USImageReconstruction-Melanie/runs/32_mix_ideal_20l_reg/eval-vn-120000/';
%lbfgs_dir = '/Volumes/MelSSD/runs/lbfgs' ;
lbfgs_dir = '/scratch_net/biwidl307/sonia/USImageReconstruction-Melanie/runs/lbfgs';
% lbfgs_dir = save_dir;
mkdir(save_dir);
%%
% % filenames = ["test-syn-0.0-0.6-patchy-val_MS_6comb.mat",
% %     "test-syn-0.0-0.6-patchy-testset_ideal_MS_32_imgs.mat", % [
% %     "test-syn-0.0-0.6-test-fullpipeline_testset_6comb_32_imgs.mat"]; %    "test-ideal_time-0.0-patchy-testset_ideal_MS_32_imgs", "dm-train-60000.mat", 
% filenames = ["dm-val-full-49000.mat"];
filenames = [
%     "test-syn-0.0-0.2-patchy-val_MS_6comb.mat"
%     "test-syn-0.0-0.4-patchy-val_MS_6comb.mat",
%     "test-syn-0.0-0.6-patchy-val_MS_6comb.mat",
%     "test-syn-0.0-0.7-patchy-val_MS_6comb.mat"
%     "test-syn-0.0-0.8-patchy-val_MS_6comb.mat",
%     "test-syn-0.0-0.9-patchy-val_MS_6comb.mat",
%     "test-ideal_time-0.0-0.2-patchy-ideal_time_val.mat",
%     "test-ideal_time-0.0-0.4-patchy-ideal_time_val.mat",
%     "test-ideal_time-0.0-0.6-patchy-ideal_time_val.mat",
%     "test-ideal_time-0.0-0.7-patchy-ideal_time_val.mat",
%     "test-ideal_time-0.0-0.8-patchy-ideal_time_val.mat",
%     "test-ideal_time-0.0-0.9-patchy-ideal_time_val.mat",
%     "test-syn-0.0-0.2-patchy-testset_ideal_MS_32_imgs.mat", 
%     "test-syn-0.0-0.4-patchy-testset_ideal_MS_32_imgs.mat", 
%     "test-syn-0.0-0.6-patchy-testset_ideal_MS_32_imgs.mat",
%     "test-syn-0.0-0.7-patchy-testset_ideal_MS_32_imgs.mat",
%     "test-syn-0.0-0.8-patchy-testset_ideal_MS_32_imgs.mat",
%     "test-syn-0.0-0.9-patchy-testset_ideal_MS_32_imgs.mat",
%     "test-ideal_time-0.0-0.2-patchy-testset_ideal_MS_32_imgs.mat", 
%     "test-ideal_time-0.0-0.4-patchy-testset_ideal_MS_32_imgs.mat", 
%     "test-ideal_time-0.0-0.6-patchy-testset_ideal_MS_32_imgs.mat",
%     "test-ideal_time-0.0-0.7-patchy-testset_ideal_MS_32_imgs.mat",
%     "test-ideal_time-0.0-0.8-patchy-testset_ideal_MS_32_imgs.mat",
%     "test-ideal_time-0.0-0.9-patchy-testset_ideal_MS_32_imgs.mat",
%     "test-syn-0.05-0.7-patchy-testset_ideal_MS_32_imgs.mat",
%     "test-syn-0.1-0.7-patchy-testset_ideal_MS_32_imgs.mat",
%     "test-syn-0.15-0.7-patchy-testset_ideal_MS_32_imgs.mat",
%     "test-syn-0.2-0.7-patchy-testset_ideal_MS_32_imgs.mat",
%     "test-syn-0.5-0.7-patchy-testset_ideal_MS_32_imgs.mat",
%     "test-syn-0.0-0.2-patchy-testset_ideal_MS_2incl_8_imgs.mat",
%     "test-syn-0.0-0.4-patchy-testset_ideal_MS_2incl_8_imgs.mat",
%     "test-syn-0.0-0.6-patchy-testset_ideal_MS_2incl_8_imgs.mat",
%     "test-syn-0.0-0.7-patchy-testset_ideal_MS_2incl_8_imgs.mat",
%     "test-syn-0.0-0.8-patchy-testset_ideal_MS_2incl_8_imgs.mat",
%     "test-syn-0.0-0.9-patchy-testset_ideal_MS_2incl_8_imgs.mat",
%     "test-ideal_time-0.0-0.2-patchy-testset_ideal_MS_2incl_8_imgs.mat",
%     "test-ideal_time-0.0-0.4-patchy-testset_ideal_MS_2incl_8_imgs.mat",
%     "test-ideal_time-0.0-0.6-patchy-testset_ideal_MS_2incl_8_imgs.mat",
%     "test-ideal_time-0.0-0.7-patchy-testset_ideal_MS_2incl_8_imgs.mat",
%     "test-ideal_time-0.0-0.8-patchy-testset_ideal_MS_2incl_8_imgs.mat",
%     "test-ideal_time-0.0-0.9-patchy-testset_ideal_MS_2incl_8_imgs.mat",
"test-syn-0.1-0.7-patchy-val_MS_6comb.mat",
"test-syn-0.05-0.7-patchy-val_MS_6comb.mat",
"test-syn-0.2-0.7-patchy-val_MS_6comb.mat",
"test-syn-0.5-0.7-patchy-val_MS_6comb.mat",
    "test-syn-0.0-0.0-test-fullpipeline_val_64_6comb_64_imgs.mat",
    "test-syn-0.0-0.0-test-fullpipeline-testset_2incl_8_imgs.mat",
    "test-syn-0.0-0.0-test-fullpipeline_testset_6comb_32_imgs.mat"
    ];
PLOT = false;
        
%% Set the params for the reconstruction
chan_comb = [15, 25 ; 
            35, 45 ; 
            55, 65;
            75, 85;
            95, 105;
            105, 115];
load('Data_kwave_inclusion1520_raw.mat')
%opts.postprocess.general.DistancePrecomp_Path = '/Users/melaniebernhardt/Desktop/MScThesis/Precomp_Distance_Matrix/';
opts.postprocess.general.DistancePrecomp_Path = '/scratch_net/biwidl307/sonia/USImageReconstruction-Melanie/code/evaluation/Precomp_Distance_Matrix/';
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
opts.postprocess.sos_minus.input = 'MS'; %'MS' or 'PW' for multistatic based sos reconstruction of plane wave based reconstruction. Has to be adapted according to the data input to this processing block, change this above
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
    fprintf(fileID, sprintf('\nCheck for %s (%d images tested) \n', filename, Nimgs));
    % GET LBFGS RECONSTRUCTION OR RECOMPUTE IT.
    try 
        load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)))
    catch
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
                subplot(1,2,2);
                imagesc(gt); colorbar; title('Ground truth SoS');
            end
            recon_lbfgs(p,:,:) = outputs;
        end
        save(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)), 'recon_lbfgs');
    end
    % Compare to VN reconstruction
    [RMSEvn, RMSElbfgs] = RMSE(Nimgs, m, recon_lbfgs) ;
    [MAEvn, MAElbfgs] = MAE(Nimgs, m, recon_lbfgs) ; 
    disp(filename) ;
    fprintf(fileID, 'RMSE VN %.2f \n', mean(RMSEvn(:)));
    fprintf(fileID, 'RMSE LBFGS %.2f \n', mean(RMSElbfgs(:)));
    fprintf(fileID, 'MAE VN %.2f \n', mean(MAEvn(:)));
    fprintf(fileID, 'MAE LBFGS %.2f \n', mean(MAElbfgs(:)));
end
fclose(fileID);

%% Plot validation set ray-based usm
x = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9] ;

for i = 1:numel(x)
    u = x(i);
    filename = sprintf('test-syn-0.0-%.1f-patchy-val_MS_6comb.mat', u);
    lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
    m = load(fullfile(data_dir, filename));
    [Nimgs,nx, ny] = size(m.init_img) ;
    [RMSEvn, RMSElbfgs] = RMSE(Nimgs, m, lbfgs_m.recon_lbfgs) ;
    rmse_vn(i) = mean(RMSEvn(:));
    rmse_lbfgs(i)= mean(RMSElbfgs(:));
end
clf;
plot(x, rmse_vn, '-o', x, rmse_lbfgs, '-o'); title('RMSE on RAY-BASED validation set'); 
legend('VN', 'LBFGS'); xlabel('Undersampling rate'); ylabel('RMSE'); xticks(0:0.1:1); 
set(gca, 'FontName', 'Palatino');
saveas(gcf, fullfile(save_dir, 'validation_usr_ray.png'));

%% Plot validation set ideal time usm
x = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9] ;

for i = 1:numel(x)
    u = x(i);
    filename = sprintf('test-ideal_time-0.0-%.1f-patchy-ideal_time_val.mat', u);
    lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
    m = load(fullfile(data_dir, filename));
    [Nimgs,nx, ny] = size(m.init_img) ;
    [RMSEvn, RMSElbfgs] = RMSE(Nimgs, m, lbfgs_m.recon_lbfgs) ;
    rmse_vn(i) = mean(RMSEvn(:));
    rmse_lbfgs(i)= mean(RMSElbfgs(:));
end
clf;
plot(x, rmse_vn, '-o', x, rmse_lbfgs, '-o'); title({'RMSE on IDEAL TIME DELAYS', 'validation set'}); 
legend('VN', 'LBFGS'); xlabel('Undersampling rate'); ylabel('RMSE'); xticks(0:0.1:1); 
set(gca, 'FontName', 'Palatino');
saveas(gcf, fullfile(save_dir, 'validation_usr_ideal.png'));

%% Plot test set ray-based usm 
x = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9] ;
for i = 1:numel(x)
    u = x(i);
    filename = sprintf('test-syn-0.0-%.1f-patchy-testset_ideal_MS_32_imgs.mat', u);
    lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
    m = load(fullfile(data_dir, filename));
    [Nimgs,nx, ny] = size(m.init_img) ;
    [RMSEvn, RMSElbfgs] = RMSE(Nimgs, m, lbfgs_m.recon_lbfgs) ;
    rmse_vn(i) = mean(RMSEvn(:));
    rmse_lbfgs(i)= mean(RMSElbfgs(:));
end
clf;
plot(x, rmse_vn, '-o', x, rmse_lbfgs, '-o'); title('RMSE on RAY-BASED test set'); 
legend('VN', 'LBFGS'); xlabel('Undersampling rate'); ylabel('RMSE'); xticks(0:0.1:1); 
set(gca, 'FontName', 'Palatino');
saveas(gcf, fullfile(save_dir, 'test_usr.png'));

%% Plot testset noise
x = [0.0, 0.05, 0.10, 0.15, 0.2, 0.5] ;
for i = 1:numel(x)
    u = x(i);
    filename = sprintf('test-syn-%.1f-0.7-patchy-testset_ideal_MS_32_imgs.mat', u);
    lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
    m = load(fullfile(data_dir, filename));
    [Nimgs,nx, ny] = size(m.init_img) ;
    [RMSEvn, RMSElbfgs] = RMSE(Nimgs, m, lbfgs_m.recon_lbfgs) ;
    rmse_vn(i) = mean(RMSEvn(:));
    rmse_lbfgs(i)= mean(RMSElbfgs(:));
end
clf;
plot(x, rmse_vn, '-o', x, rmse_lbfgs, '-o'); title({'RMSE on RAY-BASED test set','undersampling rate 0.7'}); 
legend('VN', 'LBFGS'); xlabel('Noise rate'); ylabel('RMSE'); xticks(0:0.1:1); 
set(gca, 'FontName', 'Palatino');
saveas(gcf, fullfile(save_dir, 'test_noise.png'));

%% Plot testset ideal usr 
x = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9] ;
for i = 1:numel(x)
    u = x(i);
    filename = sprintf('test-ideal_time-0.0-%.1f-patchy-testset_ideal_MS_32_imgs.mat', u);
    lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
    m = load(fullfile(data_dir, filename));
    [Nimgs,nx, ny] = size(m.init_img) ;
    [RMSEvn, RMSElbfgs] = RMSE(Nimgs, m, lbfgs_m.recon_lbfgs) ;
    rmse_vn(i) = mean(RMSEvn(:));
    rmse_lbfgs(i)= mean(RMSElbfgs(:));
end
clf;
plot(x, rmse_vn, '-o', x, rmse_lbfgs, '-o'); title('RMSE on IDEAL TIME DELAYS test set'); 
legend('VN', 'LBFGS'); xlabel('Undersampling rate'); ylabel('RMSE'); xticks(0:0.1:1); 
set(gca, 'FontName', 'Palatino');
saveas(gcf, fullfile(save_dir, 'ideal_test_usr.png'));


%% Functions
function [RMSEvn, RMSElbfgs] = RMSE(Nimgs, m, recon_lbfgs)
    for p=1:Nimgs
        try
            xtrue = m.gt_sos(p,:,:);
        catch 
            xtrue = 1./m.gt_slowness(p,:,:);
        end
        recon = squeeze(m.recon(p,:,:));	
        recon_l = squeeze(recon_lbfgs(p,:,:))' ;
        RMSEvn(p) = sqrt(mean(power(recon(:) - xtrue(:), 2)));
        RMSElbfgs(p) = sqrt(mean(power(recon_l(:) - xtrue(:), 2)));
    end
end


function [MAEvn, MAElbfgs] = MAE(Nimgs, m, recon_lbfgs)
    for p=1:Nimgs
        try
            xtrue = m.gt_sos(p,:,:);
        catch 
            xtrue = 1./m.gt_slowness(p,:,:);
        end
        recon = squeeze(m.recon(p,:,:));	
        recon_l = squeeze(recon_lbfgs(p,:,:))' ;
        MAEvn(p) = mean(abs(recon(:) - xtrue(:)));
        MAElbfgs(p) = mean(abs(recon_l(:) - xtrue(:)));
    end
end
