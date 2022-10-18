clear
% addpath(genpath('~/Desktop/MScThesis/USImageReconstruction/runs/2angles_20_med_syn/eval/'))
% addpath(genpath('~/Desktop/MScThesis/USImageReconstruction/runs/2angles_4_med_syn_orig/eval/'))
% data_dir = '~/Desktop/MScThesis/USImageReconstruction/runs/2angles_4_med_syn_orig/eval/'

data_dir = '/Volumes/MelanieDisk/runs/3_F_noS_noU_noT_noLr/eval-vn-30000-1510/';
METHOD = 'MS';
cd '/Users/melaniebernhardt/Desktop/MScThesis/SimulationModels' ;
addpath('~/Desktop/MScThesis/SimulationModels');
addpath(genpath('/Users/melaniebernhardt/Desktop/MScThesis/Generic_Data_Processing_Structure-OGtests/'));
addpath(genpath('~/Desktop/MScThesis/USImageReconstruction/code/evaluation'));
save_dir = 'eval_SoSdiff';
c = fullfile(data_dir, save_dir);
chan_comb = [15, 25 ; 
            35, 45 ; 
            55, 65;
            75, 85;
            95, 105;
            105, 115];
angle_comb = [-8, -4; -4, 0; 0, 4 ; 8, 4];
NA = 6;
mkdir(c)
Nimgs=32;
%% TEST FULL PIPELINE
filename = 'fullpipeline_test_1450_32_imgs.mat';
sos_array = [1450, 1480, 1500, 1510, 1530]; 
fileID = fopen(fullfile(data_dir, save_dir, 'CR_metrics_full.txt'),'w')
for i=1:numel(sos_array)
    sos = sos_array(i) ; 
        
        name = sprintf('test-syn-0.0-test-fullpipeline_test_%d_32_imgs.mat', sos);
        val = load(fullfile(data_dir, name));
        try 
            load(fullfile(data_dir, sprintf('lbfgs-%s', name)));
        catch
            if strcmp(METHOD,'PW')==true
                lbfgs_val = getLBFGS_PW(val, angle_comb);
            else
                lbfgs_val = getLBFGS_MS(val, chan_comb, 32);
            end
            save(fullfile(data_dir, sprintf('lbfgs-%s', name)), 'lbfgs_val', '-v7');
        end
        plot_depth_two(val, lbfgs_val, sprintf('Depth %d', sos)); 
        tmp = sprintf('depth_full_%d.png', sos);
        saveas(gcf,fullfile(data_dir, save_dir, tmp)); clf;
        plot_contrast_two(val, lbfgs_val, sprintf('Contrast - assumed SoS %d m/s', sos)); 
        tmp = sprintf('constrast_full_%d.png', sos);
        saveas(gcf,fullfile(data_dir, save_dir, tmp)); clf;
        plot_variation_two(val, lbfgs_val, sprintf('Variation - assumed SoS %d m/s', sos)); 
        tmp = sprintf('variation_full_%d.png', sos);
        saveas(gcf,fullfile(data_dir, save_dir, tmp)); clf;
        plot_size_two(val, lbfgs_val, sprintf('Size - assumed SoS %d m/s', sos)); 
        tmp = sprintf('size_full_%d.png', sos);
        saveas(gcf,fullfile(data_dir, save_dir, tmp)); clf;
        plot_smoothing_two(val, lbfgs_val, sprintf('Smoothing - assumed SoS %d m/s', sos)); 
        tmp = sprintf('smoothing_full_%d.png', sos);
        saveas(gcf, fullfile(data_dir, save_dir, tmp)); clf;
        plot_rectangle_two(val, lbfgs_val, sprintf('Rectangle - assumed SoS %d m/s', sos)); 
        tmp = sprintf('rectangle_full_%d.png', sos);
        saveas(gcf, fullfile(data_dir, save_dir, tmp)); clf;
        [CR1,  CR1lbfgs, CR_truth, RMSE1, RMSE2, PSNR1, PSNR2, MAE1, MAE2] = compute_metrics(val, lbfgs_val, Nimgs);
        CR_f1 = CR1./CR_truth;
        CR_l1 = CR1lbfgs./CR_truth;
        fprintf(fileID, sprintf('\n Assumed SoS %d 005\n', sos));
        fprintf(fileID, 'Mean CR VN %f std %f\n', mean(CR_f1(:)), std(CR_f1(:)));
        fprintf(fileID, 'Mean CR LBFGS %f std %f\n', mean(CR_l1(:)), std(CR_l1(:)));
        fprintf(fileID, 'Mean RMSE VN %f std %f\n', mean(RMSE1(:)), std(RMSE1(:)));
        fprintf(fileID, 'Mean RMSE LBFGS %f std %f\n', mean(RMSE2(:)), std(RMSE2(:)));
        fprintf(fileID, 'Mean SAD VN %f std %f\n', mean(MAE1(:)), std(MAE1(:)));
        fprintf(fileID, 'Mean SAD LBFGS %f std %f\n', mean(MAE2(:)), std(MAE2(:)));
        fprintf(fileID, 'Mean PSNR VN %f std %f\n', mean(PSNR1(:)), std(PSNR1(:)));
        fprintf(fileID, 'Mean PSNR LBFGS %f std %f\n', mean(PSNR2(:)), std(PSNR2(:)));
end
fclose(fileID);

function plot_all_both(i, m_syn, m_ideal, NA)
    clf;
    tmp = reshape(m_syn.dgt(:,i), [84,64, NA]);
    xg1 = reshape(m_syn.gt_sos(i,:,:), [64,84]);
    rec1 = reshape(m_syn.recon(i,:,:), [64,84]);
    subplot(2, NA+2, 1);
    imagesc(xg1'); colorbar; caxis([min(xg1(:)), max(xg1(:))]);
    title({'Ground truth', 'SoS map'});
    subplot(2, NA+2, 2);
    imagesc(rec1'); colorbar; caxis([min(xg1(:)), max(xg1(:))]);
    title({'Reconstructed', 'SoS map', 'LINEAR' }); 
    for a=1:NA
    subplot(2,NA+2,a+2);
    imagesc(tmp(:,:,a)); colorbar;
    title({'Undersampled measurement', 'mean inpainted', sprintf('angle combination %d',a),'LINEAR' });
    end
    tmp = reshape(m_ideal.dgt(:,i), [84,64, NA]);
    xg1 = reshape(m_ideal.gt_sos(i,:,:), [64,84]);
    rec1 = reshape(m_ideal.recon(i,:,:), [64,84]);
    subplot(2, NA+2, NA+3);
    imagesc(xg1'); colorbar; caxis([min(xg1(:)), max(xg1(:))]);
    title({'Ground truth', 'SoS map'});
    subplot(2, NA+2, NA+4);
    imagesc(rec1'); colorbar; caxis([min(xg1(:)), max(xg1(:))]);
    title({'Reconstructed', 'SoS map', 'IDEAL DELAYS'}); 
    for a=1:NA
    subplot(2,NA+2,a+NA+4);
    imagesc(tmp(:,:,a)); colorbar;
    title({'Undersampled measurement', 'mean inpainted', sprintf('angle combination %d',a),'IDEAL DELAYS' });
    end
end

function plot_all_one(i, m_syn, data_type, NA)
    clf;
    din = reshape(m_syn.din(:,i), [84,64, NA]);
    xinit = reshape(m_syn.xinit(1,i, :,:), [64,84]);
    xnorm = reshape(m_syn.xinit_norm(1, i, :,:), [64,84]);
    xg1 = reshape(m_syn.gt_sos(i,:,:), [64,84]);
    rec1 = reshape(m_syn.recon(i,:,:), [64,84]);
    subplot(2, NA, 1);
    imagesc(xg1'); colorbar; caxis([min(xg1(:))-20, max(xg1(:))+10]);
    title({'Ground truth', 'SoS map'});
    subplot(2, NA, 2);
    imagesc(rec1'); colorbar; caxis([min(xg1(:))-20, max(xg1(:))+10]);
    title({'Reconstructed', 'SoS map', data_type }); 
    subplot(2, NA, 3);
    imagesc(xinit'); colorbar;
    title({'X init'});  
    subplot(2, NA, 4);
    imagesc(xnorm'); colorbar; 
    title({'X init (normalized)'});  
    for a=1:NA
        subplot(2,NA,a+NA);
        imagesc(din(:,:,a)); colorbar;
        title({'Undersampled measurement', sprintf('Combination %d',a)});
    end
end

function plot_depth(syn, ideal, mat_lbfgs_syn, mat_lbfgs_ideal, title_str)
set(gcf, 'Position',  [1, 1, 1500, 1500]); 
for i = 1:5
    xg = reshape(syn.gt_sos(i,:,:), [64,84]);
    rec1 = reshape(syn.recon(i,:,:), [64,84]);
    rec2 = reshape(ideal.recon(i,:,:), [64,84]);
    lbfgs_syn = squeeze(mat_lbfgs_syn(i,:,:));
    lbfgs_ideal = squeeze(mat_lbfgs_ideal(i,:,:));
    subplot(5,5,i);
    imagesc(xg'); colorbar; title({'Ground truth SoS map'});
    subplot(5,5,i+5);
    imagesc(rec1'); colorbar; title({'VN Reconstructed','RAY BASED' }); caxis([min(xg(:)), max(xg(:))]);
    subplot(5,5,i+10);
    imagesc(lbfgs_syn); colorbar; title({'LBFGS Reconstructed','RAY BASED' }); %caxis([min(xg(:)), max(xg(:))]);
    subplot(5,5,i+15);    
    imagesc(rec2'); colorbar; title({'Reconstructed','IDEAL TIME' }); caxis([min(xg(:)), max(xg(:))]); 
    subplot(5,5,i+20);
    imagesc(lbfgs_ideal); colorbar; title({'LBFGS Reconstructed','IDEAL TIME' }); %caxis([min(xg(:)), max(xg(:))]);
end
suptitle(title_str)
end


function plot_depth_one(syn, title_str)
set(gcf, 'Position',  [1, 1, 1200, 400]); 
for i = 1:5
    xg = reshape(syn.gt_sos(i,:,:), [64,84]);
    rec1 = reshape(syn.recon(i,:,:), [64,84]);
    subplot(2,5,i);
    imagesc(xg'); colorbar; title({'Ground truth SoS map'});
    subplot(2,5,i+5);
    imagesc(rec1'); colorbar; title({'VN Reconstructed'}); %caxis([min(xg(:)), max(xg(:))]);
end
suptitle(title_str)
end

function plot_depth_two(syn, mat2, title_str)
set(gcf, 'Position',  [1, 1, 1600, 1000]); 
for i = 1:5
    xg = reshape(syn.gt_sos(i,:,:), [64,84]);
    rec1 = reshape(syn.recon(i,:,:), [64,84]);
    rec2 = reshape(mat2(i,:,:), [84,64]);
    subplot(3,5,i);
    imagesc(xg'); colorbar; title({'Ground truth SoS map'}); caxis([min(xg(:))-20, max(xg(:))+10]);
    subplot(3,5,i+5);
    imagesc(rec1'); colorbar; title({'VN Reconstructed'}); caxis([min(xg(:))-20, max(xg(:))+10]);
    subplot(3,5,i+10);
    imagesc(rec2); colorbar; title({'LBFGS Reconstructed'}); caxis([min(xg(:))-20, max(xg(:))+10]);
end
suptitle(title_str)
end


function plot_size(syn, ideal, mat_lbfgs_syn, mat_lbfgs_ideal, title_str)
set(gcf, 'Position',  [1, 1, 1500, 1600]);
for i = 6:9
    xg = reshape(syn.gt_sos(i,:,:), [64,84]);
    rec1 = reshape(syn.recon(i,:,:), [64,84]);
    rec2 = reshape(ideal.recon(i,:,:), [64,84]);
    lbfgs_syn = squeeze(mat_lbfgs_syn(i,:,:));
    lbfgs_ideal = squeeze(mat_lbfgs_ideal(i,:,:));
    subplot(5,4,i-5)
    imagesc(xg'); colorbar; title({'Ground truth SoS map'});
    subplot(5,4,i-5+4)
    imagesc(rec1'); colorbar; title({'VN Reconstructed','RAY BASED'}); caxis([min(xg(:))-20, max(xg(:))+10]);
    subplot(5,4,i-5+8); 
    imagesc(lbfgs_syn); colorbar; title({'LBFGS Reconstructed','RAY BASED' }); caxis([min(xg(:))-20, max(xg(:))+10]);
    subplot(5,4,i-5+12)
    imagesc(rec2'); colorbar; title({'Reconstructed','IDEAL TIME'}); caxis([min(xg(:))-20, max(xg(:))+10]); 
    subplot(5,4,i-5+16)
    imagesc(lbfgs_ideal); colorbar; title({'LBFGS  Reconstructed','IDEAL TIME'}); caxis([min(xg(:))-20, max(xg(:))+10]); 
end
suptitle(title_str)
end

function plot_size_one(syn, title_str)
set(gcf, 'Position',  [1, 1, 1500, 300]);
for i = 6:9
    xg = reshape(syn.gt_sos(i,:,:), [64,84]);
    rec1 = reshape(syn.recon(i,:,:), [64,84]);
    subplot(2,4,i-5)
    imagesc(xg'); colorbar; title({'Ground truth SoS map'});
    subplot(2,4,i-5+4)
    imagesc(rec1'); colorbar; title({'VN Reconstructed'}); caxis([min(xg(:)), max(xg(:))]);
end
suptitle(title_str)
end

function plot_size_two(syn, lb, title_str)
set(gcf, 'Position',  [1, 1, 1500, 1500]);
for i = 6:9
    xg = reshape(syn.gt_sos(i,:,:), [64,84]);
    rec1 = reshape(syn.recon(i,:,:), [64,84]);
    rec2 = reshape(lb(i,:,:), [84,64]);
    subplot(3,4,i-5)
    imagesc(xg'); colorbar; title({'Ground truth SoS map'}); caxis([min(xg(:))-20, max(xg(:))+10]);
    subplot(3,4,i-5+4)
    imagesc(rec1'); colorbar; title({'VN Reconstructed'}); caxis([min(xg(:))-20, max(xg(:))+10]);
    subplot(3, 4,i-5+8)
    imagesc(rec2); colorbar; title({'LBGFS Reconstructed'}); caxis([min(xg(:))-20, max(xg(:))+10]);
end
suptitle(title_str)
end

function plot_contrast(syn, ideal, mat_lbfgs_syn, mat_lbfgs_ideal, title_str)
set(gcf, 'Position',  [1, 1, 1800, 500]);
for i = 10:15
    xg = reshape(syn.gt_sos(i,:,:), [64,84]);
    rec1 = reshape(syn.recon(i,:,:), [64,84]);
    rec2 = reshape(ideal.recon(i,:,:), [64,84]);
    lbfgs_syn = squeeze(mat_lbfgs_syn(i,:,:));
    lbfgs_ideal = squeeze(mat_lbfgs_ideal(i,:,:));
    subplot(5,6,i-9)
    imagesc(xg'); colorbar; title({'Ground truth SoS map'}); caxis([1400 1500]);
    subplot(5,6,i-9+6)
    imagesc(rec1'); colorbar; title({'Reconstructed','RAY BASED' }); caxis([1400 1500]);
    subplot(5,6,i-9+12); 
    imagesc(lbfgs_syn); colorbar; title({'LBFGS Reconstructed','RAY BASED' }); caxis([1400 1500]);
    subplot(5,6,i-9+18);
    imagesc(rec2'); colorbar; title({'Reconstructed','IDEAL TIME' }); caxis([1400 1500]);
    subplot(5,6,i-9+24); 
    imagesc(lbfgs_ideal); colorbar; title({'LBFGS Reconstructed','IDEAL TIME' }); caxis([1400 1500]);
end
suptitle(title_str)
end

function plot_contrast_one(syn, title_str)
set(gcf, 'Position',  [1, 1, 1600, 400]);
for i = 10:15
    xg = reshape(syn.gt_sos(i,:,:), [64,84]);
    rec1 = reshape(syn.recon(i,:,:), [64,84]);
    subplot(2,6,i-9)
    imagesc(xg'); colorbar; title({'Ground truth SoS map'}); caxis([1400 1520]);
    subplot(2,6,i-9+6)
    imagesc(rec1'); colorbar; title({'Reconstructed'}); caxis([1400 1520]);
end
suptitle(title_str)
end

function plot_contrast_two(syn, lb, title_str)
set(gcf, 'Position',  [1, 1, 1600, 800]);
for i = 10:15
    xg = reshape(syn.gt_sos(i,:,:), [64,84]);
    rec1 = reshape(syn.recon(i,:,:), [64,84]);
    rec2 = reshape(lb(i,:,:), [84, 64]);
    subplot(3,6,i-9)
    imagesc(xg'); colorbar; title({'Ground truth SoS map'}); caxis([1400 1520]);
    subplot(3,6,i-9+6)
    imagesc(rec1'); colorbar; title({'VN Reconstructed'}); caxis([1400 1520]);
    subplot(3,6,i-9+12)
    imagesc(rec2); colorbar; title({'LBFGS Reconstructed'}); caxis([1400 1520]);
end
suptitle(title_str)
end

function plot_variation(syn, ideal, mat_lbfgs_syn, mat_lbfgs_ideal, title_str)
set(gcf, 'Position',  [1, 1, 2400, 800]);
t = syn.gt_sos(:,:,22);
for i= 16:22
    xg = reshape(syn.gt_sos(i,:,:), [64,84]);
    rec1 = reshape(syn.recon(i,:,:), [64,84]);
    rec2 = reshape(ideal.recon(i,:,:), [64,84]);
    lbfgs_syn = squeeze(mat_lbfgs_syn(i,:,:));
    lbfgs_ideal = squeeze(mat_lbfgs_ideal(i,:,:));
    subplot(5,8,i-15)
    imagesc(xg'); colorbar; title({'Ground truth SoS map'}); caxis([min(t(:)) max(t(:))]); 
    subplot(5,8,i-15+8);
    imagesc(rec1'); colorbar; title({'Reconstructed','RAY BASED' }); caxis([min(t(:)) max(t(:))]); 
    subplot(5,8,i-15+16); 
    imagesc(lbfgs_syn); colorbar; title({'LBFGS Reconstructed','RAY BASED' }); caxis([min(t(:)) max(t(:))]); 
    subplot(5,8,i-15+24);
    imagesc(rec2'); colorbar; title({'Reconstructed','IDEAL TIME' }); caxis([min(t(:)) max(t(:))]); 
    subplot(5,8,i-15+32); 
    imagesc(lbfgs_ideal); colorbar; title({'LBFGS Reconstructed','IDEAL TIME' }); caxis([min(t(:)) max(t(:))]); 
end
suptitle(title_str)
end

function plot_variation_one(syn, title_str)
set(gcf, 'Position',  [1, 1, 1600, 400]);
t = syn.gt_sos(:,:,22);
for i= 16:22
    xg = reshape(syn.gt_sos(i,:,:), [64,84]); caxis([1480, 1550]);
    rec1 = reshape(syn.recon(i,:,:), [64,84]);
    subplot(2,8,i-15)
    imagesc(xg'); colorbar; title({'Ground truth SoS map'}); caxis([1480, 1550]);
    subplot(2,8,i-15+8);
    imagesc(rec1'); colorbar; title({'Reconstructed'}); caxis([1480, 1550]);
end
suptitle(title_str)
end


function plot_variation_two(syn, lb, title_str)
set(gcf, 'Position',  [1, 1, 2000, 800]);
t = syn.gt_sos(:,:,22);
for i= 16:22
    xg = reshape(syn.gt_sos(i,:,:), [64,84]);
    rec1 = reshape(syn.recon(i,:,:), [64,84]);
    rec2 = reshape(lb(i,:,:), [84,64]);
    subplot(3,8,i-15)
    imagesc(xg'); colorbar; title({'Ground truth SoS map'}); caxis([1480, 1550]); 
    subplot(3,8,i-15+8);
    imagesc(rec1'); colorbar; title({'VN Reconstructed'}); caxis([1480, 1550]);
    subplot(3,8,i-15+16);
    imagesc(rec2); colorbar; title({'LBFGS Reconstructed'}); caxis([1480, 1550]);
end
suptitle(title_str)
end


function plot_smoothing(syn, ideal, mat_lbfgs_syn, mat_lbfgs_ideal, title_str)
set(gcf, 'Position',  [1, 1, 1500, 1500]);
for i = 23:27
    xg = reshape(syn.gt_sos(i,:,:), [64,84]);
    rec1 = reshape(syn.recon(i,:,:), [64,84]);
    rec2 = reshape(ideal.recon(i,:,:), [64,84]);
    lbfgs_syn = squeeze(mat_lbfgs_syn(i,:,:));
    lbfgs_ideal = squeeze(mat_lbfgs_ideal(i,:,:));
    subplot(5,5,i-22)
    imagesc(xg'); colorbar; title({'Ground truth SoS map'}); caxis([min(xg(:)), max(xg(:))]);
    subplot(5,5,i-22+5)
    imagesc(rec1'); colorbar; title({'Reconstructed','RAY BASED' }); caxis([min(xg(:)), max(xg(:))]);
    subplot(5,5,i-22+10); 
    imagesc(lbfgs_syn); colorbar; title({'LBFGS Reconstructed','RAY BASED'}); caxis([min(xg(:)), max(xg(:))]);
    subplot(5,5,i-22+15)
    imagesc(rec2'); colorbar; title({'Reconstructed','IDEAL TIME' }); caxis([min(xg(:)), max(xg(:))]);
    subplot(5,5,i-22+20); 
    imagesc(lbfgs_ideal); colorbar; title({'LBFGS Reconstructed','IDEAL TIME' }); caxis([min(xg(:)), max(xg(:))]);
end
suptitle(title_str)
end

function plot_smoothing_one(syn, title_str)
set(gcf, 'Position',  [1, 1, 1200, 400]);
for i = 23:27
    xg = reshape(syn.gt_sos(i,:,:), [64,84]);
    rec1 = reshape(syn.recon(i,:,:), [64,84]);
    subplot(2,5,i-22)
    imagesc(xg'); colorbar; title({'Ground truth SoS map'}); caxis([min(xg(:))-10, max(xg(:))+10]);
    subplot(2,5,i-22+5)
    imagesc(rec1'); colorbar; title({'Reconstructed'});caxis([min(xg(:))-10, max(xg(:))+10]);
end
suptitle(title_str)
end

function plot_smoothing_two(syn, lb, title_str)
set(gcf, 'Position',  [1, 1, 1200, 800]);
for i = 23:27
    xg = reshape(syn.gt_sos(i,:,:), [64,84]);
    rec1 = reshape(syn.recon(i,:,:), [64,84]);
    rec2 = reshape(lb(i,:,:), [84,64]);
    subplot(3,5,i-22)
    imagesc(xg'); colorbar; title({'Ground truth SoS map'}); caxis([min(xg(:))-20, max(xg(:))+10]);
    subplot(3,5,i-22+5)
    imagesc(rec1'); colorbar; title({'VN Reconstructed'}); caxis([min(xg(:))-20, max(xg(:))+10]);
    subplot(3,5,i-22+10)
    imagesc(rec2); colorbar; title({'LBFGS Reconstructed'}); caxis([min(xg(:))-20, max(xg(:))+10]);
end
suptitle(title_str)
end


function plot_rectangle(syn, ideal, mat_lbfgs_syn, mat_lbfgs_ideal, title_str)
    set(gcf, 'Position',  [1, 1, 1500, 1500]);
    for i = 28:32
        xg = reshape(syn.gt_sos(i,:,:), [64,84]);
        rec1 = reshape(syn.recon(i,:,:), [64,84]);
        rec2 = reshape(ideal.recon(i,:,:), [64,84]);
    lbfgs_syn = squeeze(mat_lbfgs_syn(i,:,:));
    lbfgs_ideal = squeeze(mat_lbfgs_ideal(i,:,:));
        subplot(5,5,i-27)
        imagesc(xg'); colorbar; title({'Ground truth SoS map'}); caxis([min(xg(:)), max(xg(:))]);
        subplot(5,5,i-27+5);
        imagesc(rec1'); colorbar; title({'Reconstructed','RAY BASED' }); caxis([min(xg(:)), max(xg(:))]);
        subplot(5,5,i-27+10); 
        imagesc(lbfgs_syn); colorbar; title({'LBFGS Reconstructed','RAY BASED' }); caxis([min(xg(:)), max(xg(:))]);
        subplot(5,5,i-27+15);
        imagesc(rec2'); colorbar; title({'Reconstructed','IDEAL TIME' });  caxis([min(xg(:)), max(xg(:))]);
        subplot(5,5,i-27+20); 
        imagesc(lbfgs_ideal); colorbar; title({'LBFGS Reconstructed','IDEAL TIME' }); caxis([min(xg(:)), max(xg(:))]);
    end
    suptitle(title_str)
end

function plot_rectangle_two(syn, lb, title_str)
    set(gcf, 'Position',  [1, 1, 1000, 500]);
    for i = 28:32
        xg = reshape(syn.gt_sos(i,:,:), [64,84]);
        rec1 = reshape(syn.recon(i,:,:), [64,84]);
        rec2 = reshape(lb(i,:,:), [84,64]);
        subplot(3,5,i-27)
        imagesc(xg'); colorbar; title({'Ground truth SoS map'}); caxis([min(xg(:))-20, max(xg(:))+10]);
        subplot(3,5,i-27+5);
        imagesc(rec1'); colorbar; title({'Reconstructed VN'}); caxis([min(xg(:))-20, max(xg(:))+10]);
        subplot(3,5,i-27+10);
        imagesc(rec2); colorbar; title({'Reconstructed LBFGS'}); caxis([min(xg(:))-20, max(xg(:))+10]);
    end
    suptitle(title_str)
end

function [CR1,  CR1lbfgs, CR_truth, RMSE1, RMSE2, PSNR1, PSNR2, MAE1, MAE2] = compute_metrics(m_syn, mat_lbfgs_syn, Nimgs)
% imaging region size
pitch = 3e-4;
Depth = 50e-3; % imaging depth (For kwave)
Width = (128-1)/2*pitch; % imaging widt
% high-resolution grid for forward simulation
pixelsize_recon_lr = [2, 2] .* pitch; % axial/lateral resolution for sos recon. grid
xax_recon_lr = [-Width : pixelsize_recon_lr(2) : Width];
xax_recon_lr = xax_recon_lr-mean(xax_recon_lr);
zax_recon_lr = [0e-3:pixelsize_recon_lr(1):Depth+0.0001];
NX_lr = numel(xax_recon_lr); 
NZ_lr = numel(zax_recon_lr);
[X_grid, Z_grid] = meshgrid(xax_recon_lr, zax_recon_lr);
CR1 = zeros(Nimgs,1);
RMSE1 = zeros(Nimgs,1);
PSNR1 = zeros(Nimgs,1);
MAE1 = zeros(Nimgs,1);
CR1lbfgs = zeros(Nimgs,1);
RMSE2 = zeros(Nimgs,1);
PSNR2 = zeros(Nimgs,1);
MAE2 = zeros(Nimgs,1);
CR_truth = zeros(Nimgs,1);
for i = 1 : Nimgs
    % inclusion definition
        rect = false;
        H = Depth/4;
        W = Width/3;
        % init rectangle
        x1 = 0-2*W/3;
        y1 = Depth/2 - H/3;
        x2 = x1 + W;
        y2 = y1;
        x3 = x2;
        y3 = y1+H;
        x4 = x1;
        y4 = y3;
        x = [x1, x2, x3, x4];
        y = [y1, y2, y3, y4];
        switch i
            % 1 to 5 depth 
            case 2
                x0 = 0;  y0 = 2*Depth/5 ;  r = Width/5; 
            case 1
                x0 = 0;  y0 = Depth/3 ;  r = Width/5; 
            case {3, 10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27}
                x0 = 0;  y0 = Depth/2 ;  r = Width/5;  
            case 4
                x0 = 0;  y0 = 2*Depth/3 ;  r = Width/5;
            case 5
                x0 = 0;  y0 = 4*Depth/5 ;  r = Width/5;
            % 6 to 9 size effect
            case 6
                x0 = 0;  y0 = Depth/2 ;  r = Width/3;
            case 7
                x0 = 0;  y0 = Depth/2 ;  r = Width/4; 
            case 8
                x0 = 0;  y0 = Depth/2 ;  r = Width/5;
            case 9
                x0 = 0;  y0 = Depth/2 ;  r = Width/10;
           % rectagnle
            case 28
                rect = true;
                angle = 0;
            case 29
                rect = true;
                angle = -25;
            case 30
                rect = true;
                angle = -45;
            case 31
                rect = true;
                angle = -65;
            case 32
                rect = true;
                angle = -90;
        end
        if rect==false
            inc_mask = (X_grid-x0).^2 + (Z_grid-y0).^2 < r^2; 
        else
            inc_mask = draw_rectangle(angle, x, y, X_grid, Z_grid);
        end
        xg = squeeze(m_syn.gt_sos(i,:,:))';
        rec1 = squeeze(m_syn.recon(i,:,:))';
        lbfgs_syn = squeeze(mat_lbfgs_syn(i,:,:));
        max_truth = max(xg(:));
        MAE1(i) = mean(abs(xg(:)-rec1(:)));
        mse1 = mean((xg(:)-rec1(:)).^2);
        RMSE1(i) = sqrt(mse1);
        PSNR1(i) = 10*log10((max_truth.^2)/mse1);
        mse2 = mean((xg(:)-lbfgs_syn(:)).^2);
        MAE2(i) = mean(abs(xg(:)-lbfgs_syn(:)));
        RMSE2(i) = sqrt(mse2);
        PSNR2(i) = 10*log10((max_truth.^2)/mse2);
        mu_inc1 = mean(rec1(inc_mask==1));
        mu_bg1 = mean(rec1(inc_mask==0));
        mu_bg = mean(xg(inc_mask==0));
        mu_inc = mean(xg(inc_mask==1));
        mu_l_inc1 = mean(lbfgs_syn(inc_mask==1));
        mu_l_bg1 = mean(lbfgs_syn(inc_mask==0));      
        CR1(i) = abs(mu_inc1-mu_bg1)/(mu_bg1);
        CR_truth(i) = abs(mu_inc-mu_bg)/(mu_bg) ;
        CR1lbfgs(i) = abs(mu_l_inc1-mu_l_bg1)/(mu_l_bg1);     
end
end

function [CR1, CR_truth] = compute_contrast_ratio_one(m_syn, Nimgs)
% imaging region size
pitch = 3e-4;
Depth = 50e-3; % imaging depth (For kwave)
Width = (128-1)/2*pitch; % imaging widt
% high-resolution grid for forward simulation
pixelsize_recon_lr = [2, 2] .* pitch; % axial/lateral resolution for sos recon. grid
xax_recon_lr = [-Width : pixelsize_recon_lr(2) : Width];
xax_recon_lr = xax_recon_lr-mean(xax_recon_lr);
zax_recon_lr = [0e-3:pixelsize_recon_lr(1):Depth+0.0001];
NX_lr = numel(xax_recon_lr); 
NZ_lr = numel(zax_recon_lr);
[X_grid, Z_grid] = meshgrid(xax_recon_lr, zax_recon_lr);
CR1 = zeros(Nimgs,1);
CR2 = zeros(Nimgs,1);
CR1lbfgs = zeros(Nimgs,1);
CR2lbfgs = zeros(Nimgs,1);
CR2 = zeros(Nimgs,1);
CR_truth = zeros(Nimgs,1);
for i = 1 : Nimgs
    % inclusion definition
        rect = false;
        H = Depth/4;
        W = Width/3;
        % init rectangle
        x1 = 0-2*W/3;
        y1 = Depth/2 - H/3;
        x2 = x1 + W;
        y2 = y1;
        x3 = x2;
        y3 = y1+H;
        x4 = x1;
        y4 = y3;
        x = [x1, x2, x3, x4];
        y = [y1, y2, y3, y4];
        switch i
            % 1 to 5 depth 
            case 2
                x0 = 0;  y0 = 2*Depth/5 ;  r = Width/5; 
            case 1
                x0 = 0;  y0 = Depth/3 ;  r = Width/5; 
            case {3, 10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27}
                x0 = 0;  y0 = Depth/2 ;  r = Width/5;  
            case 4
                x0 = 0;  y0 = 2*Depth/3 ;  r = Width/5;
            case 5
                x0 = 0;  y0 = 4*Depth/5 ;  r = Width/5;
            % 6 to 9 size effect
            case 6
                x0 = 0;  y0 = Depth/2 ;  r = Width/3;
            case 7
                x0 = 0;  y0 = Depth/2 ;  r = Width/4; 
            case 8
                x0 = 0;  y0 = Depth/2 ;  r = Width/5;
            case 9
                x0 = 0;  y0 = Depth/2 ;  r = Width/10;
           % rectagnle
            case 28
                rect = true;
                angle = 0;
            case 29
                rect = true;
                angle = -25;
            case 30
                rect = true;
                angle = -45;
            case 31
                rect = true;
                angle = -65;
            case 32
                rect = true;
                angle = -90;
        end
        if rect==false
            inc_mask = (X_grid-x0).^2 + (Z_grid-y0).^2 < r^2; 
        else
            inc_mask = draw_rectangle(angle, x, y, X_grid, Z_grid);
        end
        xg = squeeze(m_syn.gt_sos(i,:,:))';
        rec1 = squeeze(m_syn.recon(i,:,:))';
        mu_inc1 = mean(rec1(inc_mask==1));
        mu_bg1 = mean(rec1(inc_mask==0));
        mu_bg = mean(xg(inc_mask==0));
        mu_inc = mean(xg(inc_mask==1));
        CR1(i) = abs(mu_inc1-mu_bg1)/(mu_bg1);
        CR_truth(i) = abs(mu_inc-mu_bg)/(mu_bg) ;        
end
end   