%% Load data
clear all
%addpath(genpath('/Users/melaniebernhardt/Desktop/MScThesis/SimulationModels'));
addpath(genpath('/scratch_net/biwidl307/sonia/SimulationModels'))
%addpath(genpath('/Users/melaniebernhardt/Desktop/MScThesis/Generic_Data_Processing_Structure-OGtests/'));
addpath(genpath('/scratch_net/biwidl307/sonia/USImageReconstruction-Melanie/code/evaluation/Generic_Data_Processing_Structure-master/'))
%save_dir = '/Volumes/MelSSD/runs/32_mix_ideal_20l_reg/' ; 
save_dir = '/scratch_net/biwidl307/sonia/USImageReconstruction-Melanie/runs/';
lbfgs_dir = '/Volumes/MelSSD/runs/lbfgs' ;
lbfgs_dir =  '/scratch_net/biwidl307/sonia/USImageReconstruction-Melanie/runs/lbfgs'
mkdir(save_dir);
set(gcf, 'Position', [440, 286, 423, 512]);

%% Experiment 1 - number of layer
%data_dir_10 = '/Volumes/MelSSD/runs/1_10l/eval-vn-120000/' ;
data_dir_10 = '/scratch_net/biwidl307/sonia/USImageReconstruction-Melanie/runs/32_mix_ideal_20l_reg/eval-vn-120000/';
%data_dir_20 = '/Volumes/MelSSD/runs/2_20l/eval-vn-120000/' ;
data_dir_20 = '/scratch_net/biwidl307/sonia/USImageReconstruction-Melanie/runs/32_mix_ideal_20l_reg/eval-vn-120000/';
%data_dir_30 = '/Volumes/MelSSD/runs/3_30l/eval-vn-120000/' ;
data_dir_30 = '/scratch_net/biwidl307/sonia/USImageReconstruction-Melanie/runs/32_mix_ideal_20l_reg/eval-vn-120000/';
% Plot fullpipeline validation
filename = 'test-syn-0.0-0.0-test-fullpipeline_val_64_6comb_64_imgs.mat' ;
lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
m_10 = load(fullfile(data_dir_10, filename));
m_20 = load(fullfile(data_dir_20, filename));
m_30 = load(fullfile(data_dir_30, filename));
[Nimgs,nx, ny] = size(m_10.init_img) ;
[RMSEvn_10, RMSElbfgs] = RMSE(Nimgs, m_10, lbfgs_m.recon_lbfgs) ;
[RMSEvn_20, RMSElbfgs] = RMSE(Nimgs, m_20, lbfgs_m.recon_lbfgs) ;
[RMSEvn_30, RMSElbfgs] = RMSE(Nimgs, m_30, lbfgs_m.recon_lbfgs) ;
x = categorical({'VN 10 layers', 'VN 20 layers', 'VN 30 layers', 'LBFGS'}) ;
boxplot([RMSEvn_10(:), RMSEvn_20(:), RMSEvn_30(:), RMSElbfgs(:)], x, 'Colors', 'k', 'Symbol', 'k+', 'DataLim', [0, 300], 'Jitter', 0.1)
ylabel('RMSE'); %title({'RMSE on FULLPIPELINE validation set'}); 
set(gca, 'FontName', 'Palatino', 'FontSize', 13);
saveas(gcf, fullfile(save_dir, 'exp1_validation_full.png'));

% Plot fullpipeline test
filename = 'test-syn-0.0-0.0-test-fullpipeline_testset_6comb_32_imgs.mat' ;
lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
m_10 = load(fullfile(data_dir_10, filename));
m_20 = load(fullfile(data_dir_20, filename));
m_30 = load(fullfile(data_dir_30, filename));
[Nimgs,nx, ny] = size(m_10.init_img) ;
[RMSEvn_10, RMSElbfgs] = RMSE(Nimgs, m_10, lbfgs_m.recon_lbfgs) ;
[RMSEvn_20, RMSElbfgs] = RMSE(Nimgs, m_20, lbfgs_m.recon_lbfgs) ;
[RMSEvn_30, RMSElbfgs] = RMSE(Nimgs, m_30, lbfgs_m.recon_lbfgs) ;
x = categorical({'VN 10 layers', 'VN 20 layers', 'VN 30 layers', 'LBFGS'}) ;
boxplot([RMSEvn_10(:), RMSEvn_20(:), RMSEvn_30(:), RMSElbfgs(:)], x, 'Colors', 'k', 'Symbol', 'k+', 'DataLim', [0, 300], 'Jitter', 0.1)
ylabel('RMSE'); %title({'RMSE on FULLPIPELINE validation set'});
set(gca, 'FontName', 'Palatino', 'FontSize', 13);
saveas(gcf, fullfile(save_dir, 'exp1_test_full.png'));

% Plot validation set ray-based usm
x = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9] ;
for i = 1:numel(x)
    u = x(i);
    filename = sprintf('test-syn-0.0-%.1f-patchy-val_MS_6comb.mat', u);
    lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
    m_10 = load(fullfile(data_dir_10, filename));
    m_20 = load(fullfile(data_dir_20, filename));
    m_30 = load(fullfile(data_dir_30, filename));
    [Nimgs,nx, ny] = size(m_10.init_img) ;
    [RMSEvn_10, RMSElbfgs] = RMSE(Nimgs, m_10, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_20, RMSElbfgs] = RMSE(Nimgs, m_20, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_30, RMSElbfgs] = RMSE(Nimgs, m_30, lbfgs_m.recon_lbfgs) ;
    rmse_vn_10(i) = median(RMSEvn_10(:));
    rmse_vn_20(i) = median(RMSEvn_20(:));
    rmse_vn_30(i) = median(RMSEvn_30(:));
    rmse_lbfgs(i)= median(RMSElbfgs(:)); 
end
clf;
plot(x, rmse_vn_10, '-o', x, rmse_vn_20, '-o', x, rmse_vn_30, '-o', x, rmse_lbfgs, 'k--o'); %title('RMSE on RAY-BASED validation set'); 
legend('VN 10 layers', 'VN 20 layers', 'VN 30 layers', 'LBFGS', 'Location','northwest'); xlabel('Undersampling rate'); ylabel('Median RMSE'); xticks(0:0.1:1); 
set(gca, 'FontName', 'Palatino', 'FontSize', 13);
saveas(gcf, fullfile(save_dir, 'exp1_validation_usr_ray.png'));

% Plot validation set ideal time usm
x = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9] ;
for i = 1:numel(x)
    u = x(i);
    filename = sprintf('test-ideal_time-0.0-%.1f-patchy-ideal_time_val.mat', u);
    lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
    m_10 = load(fullfile(data_dir_10, filename));
    m_20 = load(fullfile(data_dir_20, filename));
    m_30 = load(fullfile(data_dir_30, filename));
    [Nimgs,nx, ny] = size(m_10.init_img) ;
    [RMSEvn_10, RMSElbfgs] = RMSE(Nimgs, m_10, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_20, RMSElbfgs] = RMSE(Nimgs, m_20, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_30, RMSElbfgs] = RMSE(Nimgs, m_30, lbfgs_m.recon_lbfgs) ;
    rmse_vn_10(i) = median(RMSEvn_10(:));
    rmse_vn_20(i) = median(RMSEvn_20(:));
    rmse_vn_30(i) = median(RMSEvn_30(:));
    rmse_lbfgs(i)= median(RMSElbfgs(:));
end
clf;
plot(x, rmse_vn_10, '-o', x, rmse_vn_20, '-o',  x, rmse_vn_30, '-o', x, rmse_lbfgs, 'k--o'); %title('RMSE on IDEAL TIME DELAYS validation set'); 
legend('VN 10 layers', 'VN 20 layers', 'VN 30 layers', 'LBFGS', 'Location','northwest'); xlabel('Undersampling rate'); ylabel('Median RMSE'); xticks(0:0.1:1); 
set(gca, 'FontName', 'Palatino', 'FontSize', 13);
saveas(gcf, fullfile(save_dir, 'exp1_validation_usr_ideal.png'));

% Plot test set ray based usm
x = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9] ;
for i = 1:numel(x)
    u = x(i);
    filename = sprintf('test-syn-0.0-%.1f-patchy-testset_ideal_MS_32_imgs.mat', u);
    lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
    m_10 = load(fullfile(data_dir_10, filename));
    m_20 = load(fullfile(data_dir_20, filename));
    m_30 = load(fullfile(data_dir_30, filename));
    [Nimgs,nx, ny] = size(m_10.init_img) ;
    [RMSEvn_10, RMSElbfgs] = RMSE(Nimgs, m_10, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_20, RMSElbfgs] = RMSE(Nimgs, m_20, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_30, RMSElbfgs] = RMSE(Nimgs, m_30, lbfgs_m.recon_lbfgs) ;
    rmse_vn_10(i) = median(RMSEvn_10(:));
    rmse_vn_20(i) = median(RMSEvn_20(:));
    rmse_vn_30(i) = median(RMSEvn_30(:));
    rmse_lbfgs(i)= median(RMSElbfgs(:));
end
clf;
plot(x, rmse_vn_10, '-o', x, rmse_vn_20, '-o', x, rmse_vn_30, '-o', x, rmse_lbfgs, 'k--o'); %title('RMSE on RAY-BASED test set'); 
legend('VN 10 layers', 'VN 20 layers', 'VN 30 layers', 'LBFGS', 'Location','northwest'); xlabel('Undersampling rate'); ylabel('Median RMSE'); xticks(0:0.1:1); 
set(gca, 'FontName', 'Palatino', 'FontSize', 13);
saveas(gcf, fullfile(save_dir, 'exp1_test_usr_ray.png'));

% Plot testset noise ray
x = [0.0, 0.05, 0.10, 0.15, 0.2, 0.5] ;
for i = 1:numel(x)
    u = x(i);
    filename = sprintf('test-syn-%.1f-0.7-patchy-testset_ideal_MS_32_imgs.mat', u);
    lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
    m_10 = load(fullfile(data_dir_10, filename));
    m_20 = load(fullfile(data_dir_20, filename));
    m_30 = load(fullfile(data_dir_30, filename));
    [Nimgs,nx, ny] = size(m_10.init_img) ;
    [RMSEvn_10, RMSElbfgs] = RMSE(Nimgs, m_10, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_20, RMSElbfgs] = RMSE(Nimgs, m_20, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_30, RMSElbfgs] = RMSE(Nimgs, m_30, lbfgs_m.recon_lbfgs) ;
    rmse_vn_10(i) = median(RMSEvn_10(:));
    rmse_vn_20(i) = median(RMSEvn_20(:));
    rmse_vn_30(i) = median(RMSEvn_30(:));
    rmse_lbfgs(i)= median(RMSElbfgs(:));
end
clf;
plot(x, rmse_vn_10, '-o', x, rmse_vn_20, '-o',  x, rmse_vn_30, '-o', x, rmse_lbfgs, 'k--o'); %title({'RMSE on RAY-BASED test set','undersampling rate 0.7'});
legend('VN 10 layers', 'VN 20 layers', 'VN 30 layers', 'LBFGS', 'Location','northwest'); xlabel('Noise rate'); ylabel('Median RMSE'); xticks(0:0.1:1); 
set(gca, 'FontName', 'Palatino', 'FontSize', 13);
saveas(gcf, fullfile(save_dir, 'exp1_test_noise.png'));
%%
% Plot validation set noise ray
x = [0.0, 0.05, 0.10, 0.2, 0.5] ;
clear('rmse_vn_10', 'rmse_vn_20', 'rmse_vn_30', 'rmse_lbfgs') 
for i = 1:numel(x)
    u = x(i);
    filename = sprintf('test-syn-%.1f-0.7-patchy-val_MS_6comb.mat', u);
    lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
    m_10 = load(fullfile(data_dir_10, filename));
    m_20 = load(fullfile(data_dir_20, filename));
    m_30 = load(fullfile(data_dir_30, filename));
    [Nimgs,nx, ny] = size(m_10.init_img) ;
    [RMSEvn_10, RMSElbfgs] = RMSE(Nimgs, m_10, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_20, RMSElbfgs] = RMSE(Nimgs, m_20, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_30, RMSElbfgs] = RMSE(Nimgs, m_30, lbfgs_m.recon_lbfgs) ;
    rmse_vn_10(i) = median(RMSEvn_10(:));
    rmse_vn_20(i) = median(RMSEvn_20(:));
    rmse_vn_30(i) = median(RMSEvn_30(:));
    rmse_lbfgs(i)= median(RMSElbfgs(:));
end
clf;
plot(x, rmse_vn_10, '-o', x, rmse_vn_20, '-o',  x, rmse_vn_30, '-o', x, rmse_lbfgs, 'k--o'); %title({'RMSE on RAY-BASED test set','undersampling rate 0.7'});
legend('VN 10 layers', 'VN 20 layers', 'VN 30 layers', 'LBFGS', 'Location','northwest'); xlabel('Noise rate'); ylabel('Median RMSE'); xticks(0:0.1:1); 
set(gca, 'FontName', 'Palatino', 'FontSize', 13);
saveas(gcf, fullfile(save_dir, 'exp1_val_noise.png'));
%%


% Plot testset ideal usr 
x = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9] ;
for i = 1:numel(x)
    u = x(i);
    filename = sprintf('test-ideal_time-0.0-%.1f-patchy-testset_ideal_MS_32_imgs.mat', u);
    lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
    m_10 = load(fullfile(data_dir_10, filename));
    m_20 = load(fullfile(data_dir_20, filename));
    m_30 = load(fullfile(data_dir_30, filename));
    [Nimgs,nx, ny] = size(m_10.init_img) ;
    [RMSEvn_10, RMSElbfgs] = RMSE(Nimgs, m_10, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_20, RMSElbfgs] = RMSE(Nimgs, m_20, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_30, RMSElbfgs] = RMSE(Nimgs, m_30, lbfgs_m.recon_lbfgs) ;
    rmse_vn_10(i) = median(RMSEvn_10(:));
    rmse_vn_20(i) = median(RMSEvn_20(:));
    rmse_vn_30(i) = median(RMSEvn_30(:));
    rmse_lbfgs(i)= median(RMSElbfgs(:));
end
clf;
plot(x, rmse_vn_10, '-o', x, rmse_vn_20, '-o',  x, rmse_vn_30, '-o', x, rmse_lbfgs, 'k--o'); %title('RMSE on IDEAL TIME DELAYS test set'); 
legend('VN 10 layers', 'VN 20 layers', 'VN 30 layers', 'LBFGS', 'Location','northwest'); xlabel('Undersampling rate'); ylabel('Median RMSE'); xticks(0:0.1:1); 
set(gca, 'FontName', 'Palatino', 'FontSize', 13);
saveas(gcf, fullfile(save_dir, 'exp1_test_usr_ideal.png'));


% Plot 2 testset ray based
x = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9] ;
for i = 1:numel(x)
    u = x(i);
    filename = sprintf('test-syn-0.0-%.1f-patchy-testset_ideal_MS_2incl_8_imgs.mat', u);
    lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
    m_10 = load(fullfile(data_dir_10, filename));
    m_20 = load(fullfile(data_dir_20, filename));
    m_30 = load(fullfile(data_dir_30, filename));
    [Nimgs,nx, ny] = size(m_10.init_img) ;
    [RMSEvn_10, RMSElbfgs] = RMSE(Nimgs, m_10, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_20, RMSElbfgs] = RMSE(Nimgs, m_20, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_30, RMSElbfgs] = RMSE(Nimgs, m_30, lbfgs_m.recon_lbfgs) ;
    rmse_vn_10(i) = median(RMSEvn_10(:));
    rmse_vn_20(i) = median(RMSEvn_20(:));
    rmse_vn_30(i) = median(RMSEvn_30(:));
    rmse_lbfgs(i)= median(RMSElbfgs(:));
end
clf;
plot(x, rmse_vn_10, '-o', x, rmse_vn_20, '-o',  x, rmse_vn_30, '-o', x, rmse_lbfgs, 'k--o'); %title({'RMSE on RAY-BASED', '2 inclusions test set'}); 
legend('VN 10 layers', 'VN 20 layers', 'VN 30 layers', 'LBFGS', 'Location','northwest'); xlabel('Undersampling rate'); ylabel('Median RMSE'); xticks(0:0.1:1); 
set(gca, 'FontName', 'Palatino', 'FontSize', 13);
saveas(gcf, fullfile(save_dir, 'exp1_test2_usr_ray.png'));


% Plot 2 testset ray based
x = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9] ;
for i = 1:numel(x)
    u = x(i);
    filename = sprintf('test-ideal_time-0.0-%.1f-patchy-testset_ideal_MS_2incl_8_imgs.mat', u);
    lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
    m_10 = load(fullfile(data_dir_10, filename));
    m_20 = load(fullfile(data_dir_20, filename));
    m_30 = load(fullfile(data_dir_30, filename));
    [Nimgs,nx, ny] = size(m_10.init_img) ;
    [RMSEvn_10, RMSElbfgs] = RMSE(Nimgs, m_10, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_20, RMSElbfgs] = RMSE(Nimgs, m_20, lbfgs_m.recon_lbfgs) ;
    rmse_vn_10(i) = median(RMSEvn_10(:));
    rmse_vn_20(i) = median(RMSEvn_20(:));
    rmse_vn_30(i) = median(RMSEvn_30(:));
    rmse_lbfgs(i)= median(RMSElbfgs(:));
end
clf;
plot(x, rmse_vn_10, '-o', x, rmse_vn_20, '-o',  x, rmse_vn_30, '-o', x, rmse_lbfgs, 'k--o'); %title({'RMSE on IDEAL TIME DELAYS', '2 inclusions test set'}); 
legend('VN 10 layers', 'VN 20 layers', 'VN 30 layers', 'LBFGS', 'Location','northwest'); xlabel('Undersampling rate'); ylabel('Median RMSE'); xticks(0:0.1:1); 
set(gca, 'FontName', 'Palatino', 'FontSize', 13);
saveas(gcf, fullfile(save_dir, 'exp1_test2_usr_ideal.png'));




%% Experiment 2 - constant against Lt
data_dir_10 = '/Volumes/MelSSD/runs/2_20l/eval-vn-120000/' ;
data_dir_20 = '/Volumes/MelSSD/runs/4_cst/eval-vn-120000/' ;

% Plot fullpipeline validation
filename = 'test-syn-0.0-0.0-test-fullpipeline_val_64_6comb_64_imgs.mat' ;
lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
m_10 = load(fullfile(data_dir_10, filename));
m_20 = load(fullfile(data_dir_20, filename));
[Nimgs,nx, ny] = size(m_10.init_img) ;
[RMSEvn_10, RMSElbfgs] = RMSE(Nimgs, m_10, lbfgs_m.recon_lbfgs) ;
[RMSEvn_20, RMSElbfgs] = RMSE(Nimgs, m_20, lbfgs_m.recon_lbfgs) ;
x = categorical({'VN Lt init.', 'VN constant init.', 'LBFGS'}) ;
boxplot([RMSEvn_10(:), RMSEvn_20(:), RMSElbfgs(:)], x, 'Colors', 'k', 'Symbol', 'k+')
ylabel('RMSE'); %title({'RMSE on FULLPIPELINE validation set'});
set(gca, 'FontName', 'Palatino', 'FontSize', 13);
 saveas(gcf, fullfile(save_dir, 'exp2_validation_full.png'));  
 
% Fullpipeline test set
filename = 'test-syn-0.0-0.0-test-fullpipeline_testset_6comb_32_imgs.mat' ;
lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
m_10 = load(fullfile(data_dir_10, filename));
m_20 = load(fullfile(data_dir_20, filename));
[Nimgs,nx, ny] = size(m_10.init_img) ;
[RMSEvn_10, RMSElbfgs] = RMSE(Nimgs, m_10, lbfgs_m.recon_lbfgs) ;
[RMSEvn_20, RMSElbfgs] = RMSE(Nimgs, m_20, lbfgs_m.recon_lbfgs) ;
x = categorical({'VN Lt init.', 'VN constant init.', 'LBFGS'}) ;
boxplot([RMSEvn_10(:), RMSEvn_20(:), RMSElbfgs(:)], x, 'Colors', 'k', 'Symbol', 'k+')
ylabel('RMSE'); %title({'RMSE on FULLPIPELINE validation set'});
set(gca, 'FontName', 'Palatino', 'FontSize', 13);
saveas(gcf, fullfile(save_dir, 'exp2_test_full.png'));

% Plot validation set ray-based usm
x = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9] ;
for i = 1:numel(x)
    u = x(i);
    filename = sprintf('test-syn-0.0-%.1f-patchy-val_MS_6comb.mat', u);
    lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
    m_10 = load(fullfile(data_dir_10, filename));
    m_20 = load(fullfile(data_dir_20, filename));
    [Nimgs,nx, ny] = size(m_10.init_img) ;
    [RMSEvn_10, RMSElbfgs] = RMSE(Nimgs, m_10, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_20, RMSElbfgs] = RMSE(Nimgs, m_20, lbfgs_m.recon_lbfgs) ;
    rmse_vn_10(i) = median(RMSEvn_10(:));
    rmse_vn_20(i) = median(RMSEvn_20(:));
    rmse_lbfgs(i)= median(RMSElbfgs(:)); 
end
clf;
plot(x, rmse_vn_10, '-o', x, rmse_vn_20, '-o', x, rmse_lbfgs, 'k--o'); %title('RMSE on RAY-BASED validation set'); 
legend('VN Lt initialization', 'VN constant initialization', 'LBFGS', 'Location', 'northwest'); xlabel('Undersampling rate'); ylabel('Median RMSE'); xticks(0:0.1:1); 
set(gca, 'FontName', 'Palatino', 'FontSize', 13);
saveas(gcf, fullfile(save_dir, 'exp2_validation_usr_ray.png'));

% Plot validation set ideal time usm
x = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9] ;
for i = 1:numel(x)
    u = x(i);
    filename = sprintf('test-ideal_time-0.0-%.1f-patchy-ideal_time_val.mat', u);
    lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
    m_10 = load(fullfile(data_dir_10, filename));
    m_20 = load(fullfile(data_dir_20, filename));
    [Nimgs,nx, ny] = size(m_10.init_img) ;
    [RMSEvn_10, RMSElbfgs] = RMSE(Nimgs, m_10, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_20, RMSElbfgs] = RMSE(Nimgs, m_20, lbfgs_m.recon_lbfgs) ;
    rmse_vn_10(i) = median(RMSEvn_10(:));
    rmse_vn_20(i) = median(RMSEvn_20(:));
    rmse_lbfgs(i)= median(RMSElbfgs(:));
end
clf;
plot(x, rmse_vn_10, '-o', x, rmse_vn_20, '-o',x, rmse_lbfgs, 'k--o'); %title('RMSE on IDEAL TIME DELAYS validation set'); 
legend('VN Lt initialization', 'VN constant initialization', 'LBFGS', 'Location', 'northwest'); xlabel('Undersampling rate'); ylabel('Median RMSE'); xticks(0:0.1:1); 
set(gca, 'FontName', 'Palatino', 'FontSize', 13);
saveas(gcf, fullfile(save_dir, 'exp2_validation_usr_ideal.png'));


% Plot test set ray based usm
x = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9] ;
for i = 1:numel(x)
    u = x(i);
    filename = sprintf('test-syn-0.0-%.1f-patchy-testset_ideal_MS_32_imgs.mat', u);
    lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
    m_10 = load(fullfile(data_dir_10, filename));
    m_20 = load(fullfile(data_dir_20, filename));
    [Nimgs,nx, ny] = size(m_10.init_img) ;
    [RMSEvn_10, RMSElbfgs] = RMSE(Nimgs, m_10, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_20, RMSElbfgs] = RMSE(Nimgs, m_20, lbfgs_m.recon_lbfgs) ;
    rmse_vn_10(i) = median(RMSEvn_10(:));
    rmse_vn_20(i) = median(RMSEvn_20(:));
    rmse_lbfgs(i)= median(RMSElbfgs(:));
end
clf;
plot(x, rmse_vn_10, '-o', x, rmse_vn_20, '-o',x, rmse_lbfgs, 'k--o'); %title('RMSE on RAY-BASED test set'); 
legend('VN Lt initialization', 'VN constant initialization', 'LBFGS', 'Location', 'northwest'); xlabel('Undersampling rate'); ylabel('Median RMSE'); xticks(0:0.1:1); 
set(gca, 'FontName', 'Palatino', 'FontSize', 13);
saveas(gcf, fullfile(save_dir, 'exp2_test_usr_ray.png'));

% Plot testset noise ray
x = [0.0, 0.05, 0.10, 0.15, 0.2, 0.5] ;
for i = 1:numel(x)
    u = x(i);
    filename = sprintf('test-syn-%.1f-0.7-patchy-testset_ideal_MS_32_imgs.mat', u);
    lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
    m_10 = load(fullfile(data_dir_10, filename));
    m_20 = load(fullfile(data_dir_20, filename));
    [Nimgs,nx, ny] = size(m_10.init_img) ;
    [RMSEvn_10, RMSElbfgs] = RMSE(Nimgs, m_10, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_20, RMSElbfgs] = RMSE(Nimgs, m_20, lbfgs_m.recon_lbfgs) ;
    rmse_vn_10(i) = median(RMSEvn_10(:));
    rmse_vn_20(i) = median(RMSEvn_20(:));
    rmse_lbfgs(i)= median(RMSElbfgs(:));
end
clf;
plot(x, rmse_vn_10, '-o', x, rmse_vn_20, '-o',x, rmse_lbfgs, 'k--o'); %title({'RMSE on RAY-BASED test set','undersampling rate 0.7'});
legend('VN Lt initialization', 'VN constant initialization', 'LBFGS', 'Location', 'northwest');xlabel('Noise rate'); ylabel('Median RMSE'); xticks(0:0.1:1); 
set(gca, 'FontName', 'Palatino', 'FontSize', 13);
saveas(gcf, fullfile(save_dir, 'exp2_test_noise.png'));


% Plot testset ideal usr 
x = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9] ;
for i = 1:numel(x)
    u = x(i);
    filename = sprintf('test-ideal_time-0.0-%.1f-patchy-testset_ideal_MS_32_imgs.mat', u);
    lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
    m_10 = load(fullfile(data_dir_10, filename));
    m_20 = load(fullfile(data_dir_20, filename));
    [Nimgs,nx, ny] = size(m_10.init_img) ;
    [RMSEvn_10, RMSElbfgs] = RMSE(Nimgs, m_10, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_20, RMSElbfgs] = RMSE(Nimgs, m_20, lbfgs_m.recon_lbfgs) ;
    rmse_vn_10(i) = median(RMSEvn_10(:));
    rmse_vn_20(i) = median(RMSEvn_20(:));
    rmse_lbfgs(i)= median(RMSElbfgs(:));
end
clf;
plot(x, rmse_vn_10, '-o', x, rmse_vn_20, '-o', x, rmse_lbfgs, 'k--o'); %title('RMSE on IDEAL TIME DELAYS test set'); 
legend('VN Lt initialization', 'VN constant initialization', 'LBFGS', 'Location', 'northwest'); xlabel('Undersampling rate'); ylabel('Median RMSE'); xticks(0:0.1:1); 
set(gca, 'FontName', 'Palatino', 'FontSize', 13);
saveas(gcf, fullfile(save_dir, 'exp2_test_usr_ideal.png'));


% Plot 2 testset ray based
x = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9] ;
for i = 1:numel(x)
    u = x(i);
    filename = sprintf('test-syn-0.0-%.1f-patchy-testset_ideal_MS_2incl_8_imgs.mat', u);
    lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
    m_10 = load(fullfile(data_dir_10, filename));
    m_20 = load(fullfile(data_dir_20, filename));
    [Nimgs,nx, ny] = size(m_10.init_img) ;
    [RMSEvn_10, RMSElbfgs] = RMSE(Nimgs, m_10, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_20, RMSElbfgs] = RMSE(Nimgs, m_20, lbfgs_m.recon_lbfgs) ;
    rmse_vn_10(i) = median(RMSEvn_10(:));
    rmse_vn_20(i) = median(RMSEvn_20(:));
    rmse_lbfgs(i)= median(RMSElbfgs(:));
end
clf;
plot(x, rmse_vn_10, '-o', x, rmse_vn_20, '-o',x, rmse_lbfgs, 'k--o'); %title({'RMSE on RAY-BASED', '2 inclusions test set'}); 
legend('VN Lt initialization', 'VN constant initialization', 'LBFGS', 'Location', 'northwest'); xlabel('Undersampling rate'); ylabel('Median RMSE'); xticks(0:0.1:1); 
set(gca, 'FontName', 'Palatino', 'FontSize', 13);
saveas(gcf, fullfile(save_dir, 'exp2_test2_usr_ray.png'));


% Plot 2 testset ray based
x = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9] ;
for i = 1:numel(x)
    u = x(i);
    filename = sprintf('test-ideal_time-0.0-%.1f-patchy-testset_ideal_MS_2incl_8_imgs.mat', u);
    lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
    m_10 = load(fullfile(data_dir_10, filename));
    m_20 = load(fullfile(data_dir_20, filename));
    [Nimgs,nx, ny] = size(m_10.init_img) ;
    [RMSEvn_10, RMSElbfgs] = RMSE(Nimgs, m_10, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_20, RMSElbfgs] = RMSE(Nimgs, m_20, lbfgs_m.recon_lbfgs) ;
    rmse_vn_10(i) = median(RMSEvn_10(:));
    rmse_vn_20(i) = median(RMSEvn_20(:));
    rmse_lbfgs(i)= median(RMSElbfgs(:));
end
clf;
plot(x, rmse_vn_10, '-o', x, rmse_vn_20, '-o', x, rmse_lbfgs, 'k--o'); %title({'RMSE on IDEAL TIME DELAYS', '2 inclusions test set'}); 
legend('VN Lt initialization', 'VN constant initialization', 'LBFGS', 'Location', 'northwest'); xlabel('Undersampling rate'); ylabel('Median RMSE'); xticks(0:0.1:1); 
set(gca, 'FontName', 'Palatino', 'FontSize', 13);
saveas(gcf, fullfile(save_dir, 'exp2_test2_usr_ideal.png'));


%% Experiment 3 - number of filters
data_dir_1 = '/Volumes/MelSSD/runs/5_1_filter_20/eval-vn-120000/' ;
data_dir_3 = '/Volumes/MelSSD/runs/6_3_filter_20/eval-vn-120000/' ;
data_dir_10 = '/Volumes/MelSSD/runs/7_10_filters_20/eval-vn-120000/';
data_dir_32 = '/Volumes/MelSSD/runs/2_20l/eval-vn-120000/';

% Plot fullpipeline validation
filename = 'test-syn-0.0-0.0-test-fullpipeline_val_64_6comb_64_imgs.mat' ;
lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
m_1 = load(fullfile(data_dir_1, filename));
m_3 = load(fullfile(data_dir_3, filename));
m_10 = load(fullfile(data_dir_10, filename));
m_32 = load(fullfile(data_dir_32, filename));
[Nimgs,nx, ny] = size(m_1.init_img) ;
[RMSEvn_1, RMSElbfgs] = RMSE(Nimgs, m_1, lbfgs_m.recon_lbfgs) ;
[RMSEvn_3, RMSElbfgs] = RMSE(Nimgs, m_3, lbfgs_m.recon_lbfgs) ;
[RMSEvn_10, RMSElbfgs] = RMSE(Nimgs, m_10, lbfgs_m.recon_lbfgs) ;
[RMSEvn_32, RMSElbfgs] = RMSE(Nimgs, m_32, lbfgs_m.recon_lbfgs) ;
x = categorical(1:5, 1:5, {'VN 1 filter', 'VN 3 filters', 'VN 10 filters', 'VN 32 filters',  'LBFGS'}, 'Ordinal' , true) ;
boxplot([RMSEvn_1(:), RMSEvn_3(:), RMSEvn_10(:), RMSEvn_32(:), RMSElbfgs(:)], x, 'Colors', 'k', 'Symbol', 'k+', 'DataLim', [0, 300], 'Jitter', 0.1)
ylabel('RMSE'); %title({'RMSE on FULLPIPELINE validation set'});
set(gca, 'FontName', 'Palatino', 'FontSize', 13);
saveas(gcf, fullfile(save_dir, 'exp3_validation_full.png'));

% Plot fullpipeline test
filename = 'test-syn-0.0-0.0-test-fullpipeline_testset_6comb_32_imgs.mat' ;
lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
m_1 = load(fullfile(data_dir_1, filename));
m_3 = load(fullfile(data_dir_3, filename));
m_10 = load(fullfile(data_dir_10, filename));
m_32 = load(fullfile(data_dir_32, filename));
[Nimgs,nx, ny] = size(m_1.init_img) ;
[RMSEvn_1, RMSElbfgs] = RMSE(Nimgs, m_1, lbfgs_m.recon_lbfgs) ;
[RMSEvn_3, RMSElbfgs] = RMSE(Nimgs, m_3, lbfgs_m.recon_lbfgs) ;
[RMSEvn_10, RMSElbfgs] = RMSE(Nimgs, m_10, lbfgs_m.recon_lbfgs) ;
[RMSEvn_32, RMSElbfgs] = RMSE(Nimgs, m_32, lbfgs_m.recon_lbfgs) ;
x = categorical(1:5, 1:5, {'VN 1 filter', 'VN 3 filters', 'VN 10 filters', 'VN 32 filters',  'LBFGS'}, 'Ordinal' , true) ;
boxplot([RMSEvn_1(:), RMSEvn_3(:), RMSEvn_10(:), RMSEvn_32(:), RMSElbfgs(:)], x, 'Colors', 'k', 'Symbol', 'k+', 'DataLim', [0, 300], 'Jitter', 0.1)
ylabel('RMSE'); %title({'RMSE on FULLPIPELINE validation set'});
set(gca, 'FontName', 'Palatino', 'FontSize', 13);
saveas(gcf, fullfile(save_dir, 'exp3_test_full.png'));

% Plot validation set ray-based usm
x = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9] ;
for i = 1:numel(x)
    u = x(i);
    filename = sprintf('test-syn-0.0-%.1f-patchy-val_MS_6comb.mat', u);
    lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
    m_1 = load(fullfile(data_dir_1, filename));
    m_3 = load(fullfile(data_dir_3, filename));
	m_10 = load(fullfile(data_dir_10, filename));
	m_32 = load(fullfile(data_dir_32, filename));
    [Nimgs,nx, ny] = size(m_1.init_img) ;
    [RMSEvn_1, RMSElbfgs] = RMSE(Nimgs, m_1, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_3, RMSElbfgs] = RMSE(Nimgs, m_3, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_10, RMSElbfgs] = RMSE(Nimgs, m_10, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_32, RMSElbfgs] = RMSE(Nimgs, m_32, lbfgs_m.recon_lbfgs) ;
    rmse_vn_1(i) = median(RMSEvn_1(:));
    rmse_vn_3(i) = median(RMSEvn_3(:));
    rmse_vn_10(i) = median(RMSEvn_10(:));
    rmse_vn_32(i) = median(RMSEvn_32(:));
    rmse_lbfgs(i)= median(RMSElbfgs(:));
end
clf;
plot(x, rmse_vn_1, '-o', x, rmse_vn_3, '-o', x, rmse_vn_10, '-o', x, rmse_vn_32, '-o', x, rmse_lbfgs, 'k--o'); %title('RMSE on RAY-BASED validation set'); 
legend('VN 1 filter', 'VN 3 filters', 'VN 10 filters', 'VN 32 filters',  'LBFGS', 'Location', 'northwest') ; xlabel('Undersampling rate'); ylabel('Median RMSE'); xticks(0:0.1:1); 
set(gca, 'FontName', 'Palatino', 'FontSize', 13);
saveas(gcf, fullfile(save_dir, 'exp3_validation_usr_ray.png'));

% Plot validation set ideal time usm
x = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9] ;
for i = 1:numel(x)
    u = x(i);
    filename = sprintf('test-ideal_time-0.0-%.1f-patchy-ideal_time_val.mat', u);
    lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
    m_1 = load(fullfile(data_dir_1, filename));
    m_3 = load(fullfile(data_dir_3, filename));
	m_10 = load(fullfile(data_dir_10, filename));
	m_32 = load(fullfile(data_dir_32, filename));
    [Nimgs,nx, ny] = size(m_1.init_img) ;
    [RMSEvn_1, RMSElbfgs] = RMSE(Nimgs, m_1, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_3, RMSElbfgs] = RMSE(Nimgs, m_3, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_10, RMSElbfgs] = RMSE(Nimgs, m_10, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_32, RMSElbfgs] = RMSE(Nimgs, m_32, lbfgs_m.recon_lbfgs) ;
    rmse_vn_1(i) = median(RMSEvn_1(:));
    rmse_vn_3(i) = median(RMSEvn_3(:));
    rmse_vn_10(i) = median(RMSEvn_10(:));
    rmse_vn_32(i) = median(RMSEvn_32(:));
    rmse_lbfgs(i)= median(RMSElbfgs(:));
end
clf;
plot(x, rmse_vn_1, '-o', x, rmse_vn_3, '-o', x, rmse_vn_10, '-o', x, rmse_vn_32, '-o', x, rmse_lbfgs, 'k--o'); %title('RMSE on IDEAL TIME DELAYS validation set'); 
legend('VN 1 filter', 'VN 3 filters', 'VN 10 filters', 'VN 32 filters',  'LBFGS', 'Location', 'northwest') ; xlabel('Undersampling rate'); ylabel('Median RMSE'); xticks(0:0.1:1); 
set(gca, 'FontName', 'Palatino', 'FontSize', 13);
saveas(gcf, fullfile(save_dir, 'exp3_validation_usr_ideal.png'));


% Plot test set ray based usm
x = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9] ;
for i = 1:numel(x)
    u = x(i);
    filename = sprintf('test-syn-0.0-%.1f-patchy-testset_ideal_MS_32_imgs.mat', u);
    lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
    m_1 = load(fullfile(data_dir_1, filename));
    m_3 = load(fullfile(data_dir_3, filename));
	m_10 = load(fullfile(data_dir_10, filename));
	m_32 = load(fullfile(data_dir_32, filename));
    [Nimgs,nx, ny] = size(m_1.init_img) ;
    [RMSEvn_1, RMSElbfgs] = RMSE(Nimgs, m_1, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_3, RMSElbfgs] = RMSE(Nimgs, m_3, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_10, RMSElbfgs] = RMSE(Nimgs, m_10, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_32, RMSElbfgs] = RMSE(Nimgs, m_32, lbfgs_m.recon_lbfgs) ;
    rmse_vn_1(i) = median(RMSEvn_1(:));
    rmse_vn_3(i) = median(RMSEvn_3(:));
    rmse_vn_10(i) = median(RMSEvn_10(:));
    rmse_vn_32(i) = median(RMSEvn_32(:));
    rmse_lbfgs(i)= median(RMSElbfgs(:));
end
clf;
plot(x, rmse_vn_1, '-o', x, rmse_vn_3, '-o', x, rmse_vn_10, '-o', x, rmse_vn_32, '-o', x, rmse_lbfgs, 'k--o'); %title('RMSE on RAY-BASED testset set'); 
legend('VN 1 filter', 'VN 3 filters', 'VN 10 filters', 'VN 32 filters', 'LBFGS', 'Location', 'northwest'); xlabel('Undersampling rate'); ylabel('Median RMSE'); xticks(0:0.1:1); 
set(gca, 'FontName', 'Palatino', 'FontSize', 13);
saveas(gcf, fullfile(save_dir, 'exp3_test_usr_ray.png'));


% Plot testset noise ray
x = [0.0, 0.05, 0.10, 0.15, 0.2, 0.5] ;
for i = 1:numel(x)
    u = x(i);
    filename = sprintf('test-syn-%.1f-0.7-patchy-testset_ideal_MS_32_imgs.mat', u);
    lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
    m_1 = load(fullfile(data_dir_1, filename));
    m_3 = load(fullfile(data_dir_3, filename));
	m_10 = load(fullfile(data_dir_10, filename));
	m_32 = load(fullfile(data_dir_32, filename));
    [Nimgs,nx, ny] = size(m_1.init_img) ;
    [RMSEvn_1, RMSElbfgs] = RMSE(Nimgs, m_1, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_3, RMSElbfgs] = RMSE(Nimgs, m_3, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_10, RMSElbfgs] = RMSE(Nimgs, m_10, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_32, RMSElbfgs] = RMSE(Nimgs, m_32, lbfgs_m.recon_lbfgs) ;
    rmse_vn_1(i) = median(RMSEvn_1(:));
    rmse_vn_3(i) = median(RMSEvn_3(:));
    rmse_vn_10(i) = median(RMSEvn_10(:));
    rmse_vn_32(i) = median(RMSEvn_32(:));
    rmse_lbfgs(i)= median(RMSElbfgs(:));
end
clf;
plot(x, rmse_vn_1, '-o', x, rmse_vn_3, '-o', x, rmse_vn_10, '-o', x, rmse_vn_32, '-o', x, rmse_lbfgs, 'k--o'); %title('RMSE on RAY-BASED testset set'); 
legend('VN 1 filter', 'VN 3 filters', 'VN 10 filters', 'VN 32 filters', 'LBFGS', 'Location', 'northwest'); xlabel('Noise rate'); ylabel('Median RMSE'); xticks(0:0.1:1); 
set(gca, 'FontName', 'Palatino', 'FontSize', 13);
saveas(gcf, fullfile(save_dir, 'exp3_test_noise_ray.png'));


% Plot testset ideal usr 
x = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9] ;
for i = 1:numel(x)
    u = x(i);
    filename = sprintf('test-ideal_time-0.0-%.1f-patchy-testset_ideal_MS_32_imgs.mat', u);
    lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
    m_1 = load(fullfile(data_dir_1, filename));
    m_3 = load(fullfile(data_dir_3, filename));
	m_10 = load(fullfile(data_dir_10, filename));
	m_32 = load(fullfile(data_dir_32, filename));
    [Nimgs,nx, ny] = size(m_1.init_img) ;
    [RMSEvn_1, RMSElbfgs] = RMSE(Nimgs, m_1, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_3, RMSElbfgs] = RMSE(Nimgs, m_3, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_10, RMSElbfgs] = RMSE(Nimgs, m_10, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_32, RMSElbfgs] = RMSE(Nimgs, m_32, lbfgs_m.recon_lbfgs) ;
    rmse_vn_1(i) = median(RMSEvn_1(:));
    rmse_vn_3(i) = median(RMSEvn_3(:));
    rmse_vn_10(i) = median(RMSEvn_10(:));
    rmse_vn_32(i) = median(RMSEvn_32(:));
    rmse_lbfgs(i)= median(RMSElbfgs(:));
end
clf;
plot(x, rmse_vn_1, '-o', x, rmse_vn_3, '-o', x, rmse_vn_10, '-o', x, rmse_vn_32, '-o', x, rmse_lbfgs, 'k--o'); %title('RMSE on IDEAL TIME DELAYS testset set'); 
legend('VN 1 filter', 'VN 3 filters', 'VN 10 filters', 'VN 32 filters', 'LBFGS', 'Location', 'northwest'); xlabel('Undersampling rate'); ylabel('Median RMSE'); xticks(0:0.1:1); 
set(gca, 'FontName', 'Palatino', 'FontSize', 13);
saveas(gcf, fullfile(save_dir, 'exp3_test_usr_ideal.png'));


% Plot 2 testset ray based
x = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9] ;
for i = 1:numel(x)
    u = x(i);
    filename = sprintf('test-syn-0.0-%.1f-patchy-testset_ideal_MS_2incl_8_imgs.mat', u);
    lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
    m_1 = load(fullfile(data_dir_1, filename));
    m_3 = load(fullfile(data_dir_3, filename));
	m_10 = load(fullfile(data_dir_10, filename));
	m_32 = load(fullfile(data_dir_32, filename));
    [Nimgs,nx, ny] = size(m_1.init_img) ;
    [RMSEvn_1, RMSElbfgs] = RMSE(Nimgs, m_1, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_3, RMSElbfgs] = RMSE(Nimgs, m_3, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_10, RMSElbfgs] = RMSE(Nimgs, m_10, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_32, RMSElbfgs] = RMSE(Nimgs, m_32, lbfgs_m.recon_lbfgs) ;
    rmse_vn_1(i) = median(RMSEvn_1(:));
    rmse_vn_3(i) = median(RMSEvn_3(:));
    rmse_vn_10(i) = median(RMSEvn_10(:));
    rmse_vn_32(i) = median(RMSEvn_32(:));
    rmse_lbfgs(i)= median(RMSElbfgs(:));
end
clf;
plot(x, rmse_vn_1, '-o', x, rmse_vn_3, '-o', x, rmse_vn_10, '-o', x, rmse_vn_32, '-o', x, rmse_lbfgs, 'k--o'); %title({'RMSE on RAY-BASED', '2 inclusions test set'}); 
legend('VN 1 filter', 'VN 3 filters', 'VN 10 filters', 'VN 32 filters', 'LBFGS', 'Location', 'northwest'); xlabel('Undersampling rate'); ylabel('Median RMSE'); xticks(0:0.1:1); 
set(gca, 'FontName', 'Palatino', 'FontSize', 13);
saveas(gcf, fullfile(save_dir, 'exp3_test2_usr_ray.png'));


% Plot 2 testset ideal based
x = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9] ;
for i = 1:numel(x)
    u = x(i);
    filename = sprintf('test-ideal_time-0.0-%.1f-patchy-testset_ideal_MS_2incl_8_imgs.mat', u);
    lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
    m_1 = load(fullfile(data_dir_1, filename));
    m_3 = load(fullfile(data_dir_3, filename));
	m_10 = load(fullfile(data_dir_10, filename));
	m_32 = load(fullfile(data_dir_32, filename));
    [Nimgs,nx, ny] = size(m_1.init_img) ;
    [RMSEvn_1, RMSElbfgs] = RMSE(Nimgs, m_1, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_3, RMSElbfgs] = RMSE(Nimgs, m_3, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_10, RMSElbfgs] = RMSE(Nimgs, m_10, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_32, RMSElbfgs] = RMSE(Nimgs, m_32, lbfgs_m.recon_lbfgs) ;
    rmse_vn_1(i) = median(RMSEvn_1(:));
    rmse_vn_3(i) = median(RMSEvn_3(:));
    rmse_vn_10(i) = median(RMSEvn_10(:));
    rmse_vn_32(i) = median(RMSEvn_32(:));
    rmse_lbfgs(i)= median(RMSElbfgs(:));
end
clf;
plot(x, rmse_vn_1, '-o', x, rmse_vn_3, '-o', x, rmse_vn_10, '-o', x, rmse_vn_32, '-o', x, rmse_lbfgs, 'k--o'); %title({'RMSE on IDEAL TIME DELAYS', '2 inclusions test set'}); 
legend('VN 1 filter', 'VN 3 filters', 'VN 10 filters', 'VN 32 filters', 'LBFGS', 'Location', 'northwest'); xlabel('Undersampling rate'); ylabel('Median RMSE'); xticks(0:0.1:1); 
set(gca, 'FontName', 'Palatino', 'FontSize', 13);
saveas(gcf, fullfile(save_dir, 'exp3_test2_usr_ideal.png'));


%% Experiment 4 - shared filters
data_dir_1 = '/Volumes/MelSSD/runs/13_20l_reg_0_1e5/eval-vn-120000/' ;
data_dir_2 = '/Volumes/MelSSD/runs/base_shared_reg/eval-vn-120000/' ;

% Plot fullpipeline validation
filename = 'test-syn-0.0-0.0-test-fullpipeline_val_64_6comb_64_imgs.mat' ;
lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
m_1 = load(fullfile(data_dir_1, filename));
m_2 = load(fullfile(data_dir_2, filename));
[Nimgs,nx, ny] = size(m_1.init_img) ;
[RMSEvn_1, RMSElbfgs] = RMSE(Nimgs, m_1, lbfgs_m.recon_lbfgs) ;
[RMSEvn_2, RMSElbfgs] = RMSE(Nimgs, m_2, lbfgs_m.recon_lbfgs) ;
x = categorical({'VN base', 'VN shared', 'LBFGS'}) ;
boxplot([RMSEvn_1(:), RMSEvn_2(:), RMSElbfgs(:)], x, 'Colors', 'k', 'Symbol', 'k+', 'DataLim', [0, 300], 'Jitter', 0.1)
ylabel('RMSE'); %title({'RMSE on FULLPIPELINE validation set'});
set(gca, 'FontName', 'Palatino', 'FontSize', 13);
saveas(gcf, fullfile(save_dir, 'exp4_validation_full.png'));

% Plot fullpipeline test
filename = 'test-syn-0.0-0.0-test-fullpipeline_testset_6comb_32_imgs.mat' ;
lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
m_1 = load(fullfile(data_dir_1, filename));
m_2 = load(fullfile(data_dir_2, filename));
[Nimgs,nx, ny] = size(m_1.init_img) ;
[RMSEvn_1, RMSElbfgs] = RMSE(Nimgs, m_1, lbfgs_m.recon_lbfgs) ;
[RMSEvn_2, RMSElbfgs] = RMSE(Nimgs, m_2, lbfgs_m.recon_lbfgs) ;
x = categorical({'VN base', 'VN shared', 'LBFGS'}) ;
boxplot([RMSEvn_1(:), RMSEvn_2(:), RMSElbfgs(:)], x, 'Colors', 'k', 'Symbol', 'k+', 'DataLim', [0, 300], 'Jitter', 0.1)
ylabel('RMSE'); %title({'RMSE on FULLPIPELINE validation set'});
set(gca, 'FontName', 'Palatino', 'FontSize', 13);
saveas(gcf, fullfile(save_dir, 'exp4_test_full.png'));


% Plot validation set ray-based usm
x = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9] ;
for i = 1:numel(x)
    u = x(i);
    filename = sprintf('test-syn-0.0-%.1f-patchy-val_MS_6comb.mat', u);
    lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
    m_1 = load(fullfile(data_dir_1, filename));
    m_2 = load(fullfile(data_dir_2, filename));
    [Nimgs,nx, ny] = size(m_1.init_img) ;
    [RMSEvn_1, RMSElbfgs] = RMSE(Nimgs, m_1, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_2, RMSElbfgs] = RMSE(Nimgs, m_2, lbfgs_m.recon_lbfgs) ;
    rmse_vn_1(i) = median(RMSEvn_1(:));
    rmse_vn_2(i) = median(RMSEvn_2(:));
    rmse_lbfgs(i)= median(RMSElbfgs(:));
end
clf;
plot(x, rmse_vn_1, '-o', x, rmse_vn_2, '-o', x, rmse_lbfgs, 'k--o'); %title('RMSE on RAY-BASED validation set'); 
legend('VN base', 'VN shared', 'LBFGS', 'Location', 'northwest'); xlabel('Undersampling rate'); ylabel('Median RMSE'); xticks(0:0.1:1); 
set(gca, 'FontName', 'Palatino', 'FontSize', 13);
saveas(gcf, fullfile(save_dir, 'exp4_validation_usr_ray.png'));

% Plot validation set ideal time usm
x = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9] ;
for i = 1:numel(x)
    u = x(i);
    filename = sprintf('test-ideal_time-0.0-%.1f-patchy-ideal_time_val.mat', u);
        lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
    m_1 = load(fullfile(data_dir_1, filename));
    m_2 = load(fullfile(data_dir_2, filename));
    [Nimgs,nx, ny] = size(m_1.init_img) ;
    [RMSEvn_1, RMSElbfgs] = RMSE(Nimgs, m_1, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_2, RMSElbfgs] = RMSE(Nimgs, m_2, lbfgs_m.recon_lbfgs) ;
    rmse_vn_1(i) = median(RMSEvn_1(:));
    rmse_vn_2(i) = median(RMSEvn_2(:));
    rmse_lbfgs(i)= median(RMSElbfgs(:));
end
clf;
plot(x, rmse_vn_1, '-o', x, rmse_vn_2, '-o', x, rmse_lbfgs, 'k--o'); %title('RMSE on RAY-BASED validation set'); 
legend('VN base', 'VN shared', 'LBFGS', 'Location', 'northwest'); xlabel('Undersampling rate'); ylabel('Median RMSE'); xticks(0:0.1:1); 
set(gca, 'FontName', 'Palatino', 'FontSize', 13);
saveas(gcf, fullfile(save_dir, 'exp4_validation_usr_ideal.png'));


% Plot test set ray based usm
x = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9] ;
for i = 1:numel(x)
    u = x(i);
    filename = sprintf('test-syn-0.0-%.1f-patchy-testset_ideal_MS_32_imgs.mat', u);
        lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
    m_1 = load(fullfile(data_dir_1, filename));
    m_2 = load(fullfile(data_dir_2, filename));
    [Nimgs,nx, ny] = size(m_1.init_img) ;
    [RMSEvn_1, RMSElbfgs] = RMSE(Nimgs, m_1, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_2, RMSElbfgs] = RMSE(Nimgs, m_2, lbfgs_m.recon_lbfgs) ;
    rmse_vn_1(i) = median(RMSEvn_1(:));
    rmse_vn_2(i) = median(RMSEvn_2(:));
    rmse_lbfgs(i)= median(RMSElbfgs(:));
end
clf;
plot(x, rmse_vn_1, '-o', x, rmse_vn_2, '-o', x, rmse_lbfgs, 'k--o'); %title('RMSE on RAY-BASED validation set'); 
legend('VN base', 'VN shared', 'LBFGS', 'Location', 'northwest'); xlabel('Undersampling rate'); ylabel('Median RMSE'); xticks(0:0.1:1); 
set(gca, 'FontName', 'Palatino', 'FontSize', 13);
saveas(gcf, fullfile(save_dir, 'exp4_test_usr_ray.png'));


% Plot testset ideal usr 
x = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9] ;
for i = 1:numel(x)
    u = x(i);
    filename = sprintf('test-ideal_time-0.0-%.1f-patchy-testset_ideal_MS_32_imgs.mat', u);
        lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
    m_1 = load(fullfile(data_dir_1, filename));
    m_2 = load(fullfile(data_dir_2, filename));
    [Nimgs,nx, ny] = size(m_1.init_img) ;
    [RMSEvn_1, RMSElbfgs] = RMSE(Nimgs, m_1, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_2, RMSElbfgs] = RMSE(Nimgs, m_2, lbfgs_m.recon_lbfgs) ;
    rmse_vn_1(i) = median(RMSEvn_1(:));
    rmse_vn_2(i) = median(RMSEvn_2(:));
    rmse_lbfgs(i)= median(RMSElbfgs(:));
end
clf;
plot(x, rmse_vn_1, '-o', x, rmse_vn_2, '-o', x, rmse_lbfgs, 'k--o'); %title('RMSE on RAY-BASED validation set'); 
legend('VN base', 'VN shared', 'LBFGS', 'Location', 'northwest'); xlabel('Undersampling rate'); ylabel('Median RMSE'); xticks(0:0.1:1); 
set(gca, 'FontName', 'Palatino', 'FontSize', 13);
saveas(gcf, fullfile(save_dir, 'exp4_test_usr_ideal.png'));






%% Experiment 5 investigating exponential weight
data_dir_1 = '/Volumes/MelSSD/runs/nexp/eval-vn-120000/' ;
data_dir_3 = '/Volumes/MelSSD/runs/fixedT/eval-vn-120000/' ;
data_dir_10 = '/Volumes/MelSSD/runs/2_20l/eval-vn-120000/';
%%
% Plot fullpipeline validation
filename = 'test-syn-0.0-0.0-test-fullpipeline_val_64_6comb_64_imgs.mat' ;
lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
m_1 = load(fullfile(data_dir_1, filename));
m_3 = load(fullfile(data_dir_3, filename));
m_10 = load(fullfile(data_dir_10, filename));
[Nimgs,nx, ny] = size(m_1.init_img) ;
[RMSEvn_1, RMSElbfgs] = RMSE(Nimgs, m_1, lbfgs_m.recon_lbfgs) ;
[RMSEvn_3, RMSElbfgs] = RMSE(Nimgs, m_3, lbfgs_m.recon_lbfgs) ;
[RMSEvn_10, RMSElbfgs] = RMSE(Nimgs, m_10, lbfgs_m.recon_lbfgs) ;
x = categorical({'VN $$\tau=\infty$$', 'VN $$\tau=5$$', 'VN $$\tau=0 \rightarrow \infty$$', 'LBFGS'}) ;
boxplot([RMSEvn_1(:), RMSEvn_3(:), RMSEvn_10(:), RMSElbfgs(:)], x, 'Colors', 'k', 'Symbol', 'k+', 'DataLim', [0, 300], 'Jitter', 0.1)
ylabel('RMSE'); %title({'RMSE on FULLPIPELINE validation set'});
set(gca, 'FontName', 'Palatino', 'FontSize', 13);
bp = gca;
bp.XAxis.TickLabelInterpreter = 'latex';
saveas(gcf, fullfile(save_dir, 'exp5_validation_full.png'));

% Plot fullpipeline test
filename = 'test-syn-0.0-0.0-test-fullpipeline_testset_6comb_32_imgs.mat' ;
lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
m_1 = load(fullfile(data_dir_1, filename));
m_3 = load(fullfile(data_dir_3, filename));
m_10 = load(fullfile(data_dir_10, filename));
[Nimgs,nx, ny] = size(m_1.init_img) ;
[RMSEvn_1, RMSElbfgs] = RMSE(Nimgs, m_1, lbfgs_m.recon_lbfgs) ;
[RMSEvn_3, RMSElbfgs] = RMSE(Nimgs, m_3, lbfgs_m.recon_lbfgs) ;
[RMSEvn_10, RMSElbfgs] = RMSE(Nimgs, m_10, lbfgs_m.recon_lbfgs) ;
x = categorical({'VN $$\tau=\infty$$', 'VN $$\tau=5$$', 'VN $$\tau=0 \rightarrow \infty$$', 'LBFGS'}) ;
boxplot([RMSEvn_1(:), RMSEvn_3(:), RMSEvn_10(:), RMSElbfgs(:)], x, 'Colors', 'k', 'Symbol', 'k+', 'DataLim', [0, 300], 'Jitter', 0.1)
ylabel('RMSE'); %title({'RMSE on FULLPIPELINE validation set'});
set(gca, 'FontName', 'Palatino', 'FontSize', 13);
bp = gca;
bp.XAxis.TickLabelInterpreter = 'latex';
saveas(gcf, fullfile(save_dir, 'exp5_test_full.png'));


% Plot validation set ray-based usm
x = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9] ;
for i = 1:numel(x)
    u = x(i);
    filename = sprintf('test-syn-0.0-%.1f-patchy-val_MS_6comb.mat', u);
    lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
    m_1 = load(fullfile(data_dir_1, filename));
    m_3 = load(fullfile(data_dir_3, filename));
	m_10 = load(fullfile(data_dir_10, filename));
    [Nimgs,nx, ny] = size(m_1.init_img) ;
    [RMSEvn_1, RMSElbfgs] = RMSE(Nimgs, m_1, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_3, RMSElbfgs] = RMSE(Nimgs, m_3, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_10, RMSElbfgs] = RMSE(Nimgs, m_10, lbfgs_m.recon_lbfgs) ;
    rmse_vn_1(i) = median(RMSEvn_1(:));
    rmse_vn_3(i) = median(RMSEvn_3(:));
    rmse_vn_10(i) = median(RMSEvn_10(:));
    rmse_lbfgs(i)= median(RMSElbfgs(:));
end
clf;
plot(x, rmse_vn_1, '-o', x, rmse_vn_3, '-o', x, rmse_vn_10, '-o', x, rmse_lbfgs, 'k--o'); %title('RMSE on RAY-BASED validation set'); 
legend('VN \tau=\infty', 'VN \tau=5', 'VN \tau=0 \rightarrow \infty', 'LBFGS', 'Location', 'northwest'); xlabel('Undersampling rate'); ylabel('Median RMSE'); xticks(0:0.1:1); 
set(gca, 'FontName', 'Palatino', 'FontSize', 13);
saveas(gcf, fullfile(save_dir, 'exp5_validation_usr_ray.png'));

% Plot validation set ideal time usm
x = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9] ;
for i = 1:numel(x)
    u = x(i);
    filename = sprintf('test-ideal_time-0.0-%.1f-patchy-ideal_time_val.mat', u);
    lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
    m_1 = load(fullfile(data_dir_1, filename));
    m_3 = load(fullfile(data_dir_3, filename));
	m_10 = load(fullfile(data_dir_10, filename));
    [Nimgs,nx, ny] = size(m_1.init_img) ;
    [RMSEvn_1, RMSElbfgs] = RMSE(Nimgs, m_1, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_3, RMSElbfgs] = RMSE(Nimgs, m_3, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_10, RMSElbfgs] = RMSE(Nimgs, m_10, lbfgs_m.recon_lbfgs) ;
    rmse_vn_1(i) = median(RMSEvn_1(:));
    rmse_vn_3(i) = median(RMSEvn_3(:));
    rmse_vn_10(i) = median(RMSEvn_10(:));
    rmse_lbfgs(i)= median(RMSElbfgs(:));
end
clf;
plot(x, rmse_vn_1, '-o', x, rmse_vn_3, '-o', x, rmse_vn_10, '-o', x, rmse_lbfgs, 'k--o'); %title('RMSE on IDEAL TIME DELAYS validation set'); 
legend('VN \tau=\infty', 'VN \tau=5', 'VN \tau=0\rightarrow\infty', 'LBFGS', 'Location', 'northwest');xlabel('Undersampling rate'); ylabel('Median RMSE'); xticks(0:0.1:1); 
set(gca, 'FontName', 'Palatino', 'FontSize', 13);
saveas(gcf, fullfile(save_dir, 'exp5_validation_usr_ideal.png'));


% Plot test set ray based usm
x = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9] ;
for i = 1:numel(x)
    u = x(i);
    filename = sprintf('test-syn-0.0-%.1f-patchy-testset_ideal_MS_32_imgs.mat', u);
    lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
	m_1 = load(fullfile(data_dir_1, filename));
    m_3 = load(fullfile(data_dir_3, filename));
	m_10 = load(fullfile(data_dir_10, filename));
    [Nimgs,nx, ny] = size(m_1.init_img) ;
    [RMSEvn_1, RMSElbfgs] = RMSE(Nimgs, m_1, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_3, RMSElbfgs] = RMSE(Nimgs, m_3, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_10, RMSElbfgs] = RMSE(Nimgs, m_10, lbfgs_m.recon_lbfgs) ;
    rmse_vn_1(i) = median(RMSEvn_1(:));
    rmse_vn_3(i) = median(RMSEvn_3(:));
    rmse_vn_10(i) = median(RMSEvn_10(:));
    rmse_lbfgs(i)= median(RMSElbfgs(:));
end
clf;
plot(x, rmse_vn_1, '-o', x, rmse_vn_3, '-o', x, rmse_vn_10, '-o', x, rmse_lbfgs, 'k--o'); %title('RMSE on RAY-BASED testset set'); 
legend('VN \tau=\infty', 'VN \tau=5', 'VN \tau=0 \rightarrow \infty', 'LBFGS', 'Location', 'northwest'); xlabel('Undersampling rate'); ylabel('Median RMSE'); xticks(0:0.1:1); 
set(gca, 'FontName', 'Palatino', 'FontSize', 13);
saveas(gcf, fullfile(save_dir, 'exp5_test_usr_ray.png'));
%%

% % Plot testset noise ray
% x = [0.0, 0.05, 0.10, 0.15, 0.2, 0.5] ;
% for i = 1:numel(x)
%     u = x(i);
%     filename = sprintf('test-syn-%.1f-0.7-patchy-testset_ideal_MS_32_imgs.mat', u);
%     lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
%     m_1 = load(fullfile(data_dir_1, filename));
%     m_3 = load(fullfile(data_dir_3, filename));
% 	m_10 = load(fullfile(data_dir_10, filename));
%     [Nimgs,nx, ny] = size(m_1.init_img) ;
%     [RMSEvn_1, RMSElbfgs] = RMSE(Nimgs, m_1, lbfgs_m.recon_lbfgs) ;
%     [RMSEvn_3, RMSElbfgs] = RMSE(Nimgs, m_3, lbfgs_m.recon_lbfgs) ;
%     [RMSEvn_10, RMSElbfgs] = RMSE(Nimgs, m_10, lbfgs_m.recon_lbfgs) ;
%     rmse_vn_1(i) = median(RMSEvn_1(:));
%     rmse_vn_3(i) = median(RMSEvn_3(:));
%     rmse_vn_10(i) = median(RMSEvn_10(:));
%     rmse_lbfgs(i)= median(RMSElbfgs(:));
% end
% clf;
% plot(x, rmse_vn_1, '-o', x, rmse_vn_3, '-o', x, rmse_vn_10, '-o', x, rmse_lbfgs, 'k--o'); %title('RMSE on RAY-BASED testset set'); 
% legend('VN \tau=\infty', 'VN \tau=5', 'VN \tau=0\to\infty', 'LBFGS', 'Location', 'northwest'); xlabel('Noise rate'); ylabel('Median RMSE'); xticks(0:0.1:1); 
% set(gca, 'FontName', 'Palatino', 'FontSize', 13);
% saveas(gcf, fullfile(save_dir, 'exp5_test_noise_ray.png'));


% Plot testset ideal usr 
x = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9] ;
for i = 1:numel(x)
    u = x(i);
    filename = sprintf('test-ideal_time-0.0-%.1f-patchy-testset_ideal_MS_32_imgs.mat', u);
    lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
    m_1 = load(fullfile(data_dir_1, filename));
    m_3 = load(fullfile(data_dir_3, filename));
	m_10 = load(fullfile(data_dir_10, filename));
    [Nimgs,nx, ny] = size(m_1.init_img) ;
    [RMSEvn_1, RMSElbfgs] = RMSE(Nimgs, m_1, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_3, RMSElbfgs] = RMSE(Nimgs, m_3, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_10, RMSElbfgs] = RMSE(Nimgs, m_10, lbfgs_m.recon_lbfgs) ;
    rmse_vn_1(i) = median(RMSEvn_1(:));
    rmse_vn_3(i) = median(RMSEvn_3(:));
    rmse_vn_10(i) = median(RMSEvn_10(:));
    rmse_lbfgs(i)= median(RMSElbfgs(:));
end
clf;
plot(x, rmse_vn_1, '-o', x, rmse_vn_3, '-o', x, rmse_vn_10, '-o', x, rmse_lbfgs, 'k--o'); %title('RMSE on IDEAL TIME DELAYS testset set'); 
legend('VN \tau=\infty', 'VN \tau=5', 'VN \tau=0\rightarrow\infty', 'LBFGS', 'Location', 'northwest'); xlabel('Undersampling rate'); ylabel('Median RMSE'); xticks(0:0.1:1); 
set(gca, 'FontName', 'Palatino', 'FontSize', 13);
saveas(gcf, fullfile(save_dir, 'exp5_test_usr_ideal.png'));


% Plot 2 testset ray based
x = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9] ;
for i = 1:numel(x)
    u = x(i);
    filename = sprintf('test-syn-0.0-%.1f-patchy-testset_ideal_MS_2incl_8_imgs.mat', u);
    lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
    m_1 = load(fullfile(data_dir_1, filename));
    m_3 = load(fullfile(data_dir_3, filename));
	m_10 = load(fullfile(data_dir_10, filename));
    [Nimgs,nx, ny] = size(m_1.init_img) ;
    [RMSEvn_1, RMSElbfgs] = RMSE(Nimgs, m_1, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_3, RMSElbfgs] = RMSE(Nimgs, m_3, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_10, RMSElbfgs] = RMSE(Nimgs, m_10, lbfgs_m.recon_lbfgs) ;
    rmse_vn_1(i) = median(RMSEvn_1(:));
    rmse_vn_3(i) = median(RMSEvn_3(:));
    rmse_vn_10(i) = median(RMSEvn_10(:));
    rmse_lbfgs(i)= median(RMSElbfgs(:));
end
clf;
plot(x, rmse_vn_1, '-o', x, rmse_vn_3, '-o', x, rmse_vn_10, '-o', x, rmse_lbfgs, 'k--o'); %title({'RMSE on RAY-BASED', '2 inclusions test set'}); 
legend('VN \tau=\infty', 'VN \tau=5', 'VN \tau=0\rightarrow\infty', 'LBFGS', 'Location', 'northwest'); xlabel('Undersampling rate'); ylabel('Median RMSE'); xticks(0:0.1:1); 
set(gca, 'FontName', 'Palatino', 'FontSize', 13);
saveas(gcf, fullfile(save_dir, 'exp5_test2_usr_ray.png'));


% Plot 2 testset ideal based
x = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9] ;
for i = 1:numel(x)
    u = x(i);
    filename = sprintf('test-ideal_time-0.0-%.1f-patchy-testset_ideal_MS_2incl_8_imgs.mat', u);
    lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
    m_1 = load(fullfile(data_dir_1, filename));
    m_3 = load(fullfile(data_dir_3, filename));
	m_10 = load(fullfile(data_dir_10, filename));
    [Nimgs,nx, ny] = size(m_1.init_img) ;
    [RMSEvn_1, RMSElbfgs] = RMSE(Nimgs, m_1, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_3, RMSElbfgs] = RMSE(Nimgs, m_3, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_10, RMSElbfgs] = RMSE(Nimgs, m_10, lbfgs_m.recon_lbfgs) ;
    rmse_vn_1(i) = median(RMSEvn_1(:));
    rmse_vn_3(i) = median(RMSEvn_3(:));
    rmse_vn_10(i) = median(RMSEvn_10(:));
    rmse_lbfgs(i)= median(RMSElbfgs(:));
end
clf;
plot(x, rmse_vn_1, '-o', x, rmse_vn_3, '-o', x, rmse_vn_10, '-o', x, rmse_lbfgs, 'k--o'); %title({'RMSE on IDEAL TIME DELAYS', '2 inclusions test set'}); 
legend('VN \tau=\infty', 'VN \tau=5', 'VN \tau=0\rightarrow\infty', 'LBFGS', 'Location', 'northwest'); xlabel('Undersampling rate'); ylabel('Median RMSE'); xticks(0:0.1:1); 
set(gca, 'FontName', 'Palatino', 'FontSize', 13);
saveas(gcf, fullfile(save_dir, 'exp5_test2_usr_ideal.png'));

%%
% opt plots
o1 = load(fullfile(data_dir_1, 'opt-eval/residuals.mat'));
o3 = load(fullfile(data_dir_3, 'opt-eval/residuals.mat'));
o10 = load(fullfile(data_dir_10, 'opt-eval/residuals.mat'));

plot(0:20, mean(o1.data_res, 1), 0:20, mean(o3.data_res, 1), 0:20, median(o10.data_res, 1) ) ;
xlabel('Layer'); ylabel('Mean data residual'); set(gca, 'FontName', 'Palatino', 'FontSize', 13); 
legend('VN \tau=\infty', 'VN \tau=5', 'VN \tau=0\rightarrow\infty', 'Location', 'northwest');
saveas(gcf, fullfile(save_dir, 'exp5_data_res.png'));

plot(0:20, mean(o1.img_res, 1), 0:20, mean(o3.img_res, 1), 0:20, mean(o10.img_res, 1) ) ;
xlabel('Layer'); 
ylabel('Mean image residual'); 
set(gca, 'FontName', 'Palatino', 'FontSize', 13); 
legend('VN \tau=\infty', 'VN \tau=5', 'VN \tau=0\rightarrow\infty', 'Location', 'northwest');
saveas(gcf, fullfile(save_dir, 'exp5_img_res.png'));

plot(0:20, mean(sum(abs(o1.d_grad), [3,4]), 1), 0:20, mean(sum(abs(o3.d_grad), [3,4]), 1), 0:20, mean(sum(abs(o10.d_grad), [3,4]), 1)) ; xlabel('Layer') ; ylabel('||\nabla_d^{(layer)}||_1'); set(gca, 'FontName', 'Palatino', 'FontSize', 13); legend('VN \tau=\infty', 'VN \tau=5', 'VN \tau=0\rightarrow\infty', 'Location', 'southwest');
saveas(gcf, fullfile(save_dir, 'exp5_d_grad.png'));

plot(0:20, mean(sum(abs(o1.r_grad), [3, 4]), 1), 0:20, mean(sum(abs(o3.r_grad), [3, 4]), 1), 0:20, mean(sum(abs(o10.r_grad), [3, 4]), 1)); xlabel('Layer') ; ylabel('||\nabla_r^{(layer)}||_1'); set(gca, 'FontName', 'Palatino', 'FontSize', 13); legend('VN \tau=\infty', 'VN \tau=5', 'VN \tau=0\rightarrow\infty', 'Location', 'southwest');
saveas(gcf, fullfile(save_dir, 'exp5_r_grad.png'));

plot(0:20, mean(o1.corr_r_d,1), 0:20, mean(o3.corr_r_d,1), 0:20, mean(o10.corr_r_d,1)); xlabel('Layer') ; ylabel('Correlation coefficient'); set(gca, 'FontName', 'Palatino', 'FontSize', 13); legend('VN \tau=\infty', 'VN \tau=5', 'VN \tau=0\rightarrow\infty', 'Location', 'southwest');
saveas(gcf, fullfile(save_dir, 'exp5_corr.png'));




%% Exp 6 activation function regularization
data_dir_1 = '/Volumes/MelSSD/runs/12_20l_reg_1e5_1e5/eval-vn-120000/' ;
data_dir_2 = '/Volumes/MelSSD/runs/13_20l_reg_0_1e5/eval-vn-120000/' ;
data_dir_3 = '/Volumes/MelSSD/runs/14_20l_reg_1e5_0/eval-vn-120000/';
data_dir_4 = '/Volumes/MelSSD/runs/base/eval-vn-120000/'; 
% data_dir_5 = '/Volumes/MelSSD/runs/17_30l_reg_1e3_1e4/eval-vn-120000/';
% data_dir_6 = '/Volumes/MelSSD/runs/18_30l_reg_1e6_1e6/eval-vn-120000/';
% data_dir_7 = '/Volumes/MelSSD/runs/19_30l_reg_1e3_1e3/eval-vn-120000/';
% data_dir_8 = '/Volumes/MelSSD/runs/3_30l/eval-vn-120000/';
set(gcf, 'Position', [539, 1071, 837, 665]) ;
% Plot fullpipeline validation
filename = 'test-syn-0.0-0.0-test-fullpipeline_val_64_6comb_64_imgs.mat' ;
lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
m_1 = load(fullfile(data_dir_1, filename));
m_2 = load(fullfile(data_dir_2, filename));
m_3 = load(fullfile(data_dir_3, filename));
m_4 = load(fullfile(data_dir_4, filename));
%m_5 = load(fullfile(data_dir_5, filename));
%m_6 = load(fullfile(data_dir_6, filename));
%m_7 = load(fullfile(data_dir_7, filename));
%m_8 = load(fullfile(data_dir_8, filename));
[Nimgs,nx, ny] = size(m_1.init_img) ;
[RMSEvn_1, RMSElbfgs] = RMSE(Nimgs, m_1, lbfgs_m.recon_lbfgs) ;
[RMSEvn_2, RMSElbfgs] = RMSE(Nimgs, m_2, lbfgs_m.recon_lbfgs) ;
[RMSEvn_3, RMSElbfgs] = RMSE(Nimgs, m_3, lbfgs_m.recon_lbfgs) ;
[RMSEvn_4, RMSElbfgs] = RMSE(Nimgs, m_4, lbfgs_m.recon_lbfgs) ;
%[RMSEvn_5, RMSElbfgs] = RMSE(Nimgs, m_5, lbfgs_m.recon_lbfgs) ;
%[RMSEvn_6, RMSElbfgs] = RMSE(Nimgs, m_6, lbfgs_m.recon_lbfgs) ;
%[RMSEvn_7, RMSElbfgs] = RMSE(Nimgs, m_7, lbfgs_m.recon_lbfgs) ;
%[RMSEvn_8, RMSElbfgs] = RMSE(Nimgs, m_8, lbfgs_m.recon_lbfgs) ;
x = categorical({'VN $$\lambda_{d}: 1e5, \lambda_{r}: 1e5$$', 'VN $$\lambda_{d}: 0, \lambda_{r}: 1e5$$', 'VN $$\lambda_{d}: 1e5, \lambda_{r}: 0$$', 'No regularization', 'LBFGS'}) ;
boxplot([RMSEvn_1(:), RMSEvn_2(:), RMSEvn_3(:), RMSEvn_4(:), RMSElbfgs(:)], x, 'Colors', 'k', 'Symbol', 'k+', 'DataLim', [0, 300], 'Jitter', 0.1)
ylabel('RMSE'); %title({'RMSE on FULLPIPELINE validation set'});
set(gca, 'FontName', 'Palatino', 'FontSize', 13);
bp = gca;
bp.XAxis.TickLabelInterpreter = 'latex';
saveas(gcf, fullfile(save_dir, 'exp6_validation_full.png'));

% Plot fullpipeline test
filename = 'test-syn-0.0-0.0-test-fullpipeline_testset_6comb_32_imgs.mat' ;
lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
m_1 = load(fullfile(data_dir_1, filename));
m_2 = load(fullfile(data_dir_2, filename));
m_3 = load(fullfile(data_dir_3, filename));
m_4 = load(fullfile(data_dir_4, filename));
%m_5 = load(fullfile(data_dir_5, filename));
%m_6 = load(fullfile(data_dir_6, filename));
%m_7 = load(fullfile(data_dir_7, filename));
%m_8 = load(fullfile(data_dir_8, filename));
[Nimgs,nx, ny] = size(m_1.init_img) ;
[RMSEvn_1, RMSElbfgs] = RMSE(Nimgs, m_1, lbfgs_m.recon_lbfgs) ;
[RMSEvn_2, RMSElbfgs] = RMSE(Nimgs, m_2, lbfgs_m.recon_lbfgs) ;
[RMSEvn_3, RMSElbfgs] = RMSE(Nimgs, m_3, lbfgs_m.recon_lbfgs) ;
[RMSEvn_4, RMSElbfgs] = RMSE(Nimgs, m_4, lbfgs_m.recon_lbfgs) ;
%[RMSEvn_5, RMSElbfgs] = RMSE(Nimgs, m_5, lbfgs_m.recon_lbfgs) ;
%[RMSEvn_6, RMSElbfgs] = RMSE(Nimgs, m_6, lbfgs_m.recon_lbfgs) ;
%[RMSEvn_7, RMSElbfgs] = RMSE(Nimgs, m_7, lbfgs_m.recon_lbfgs) ;
%[RMSEvn_8, RMSElbfgs] = RMSE(Nimgs, m_8, lbfgs_m.recon_lbfgs) ;
x = categorical({'VN $$\lambda_{d}: 10^5, \lambda_{r}: 10^5$$', 'VN $$\lambda_{d}: 0, \lambda_{r}: 10^5$$', 'VN $$\lambda_{d}: 10^5, \lambda_{r}: 0$$', 'No regularization', 'LBFGS'}) ;
boxplot([RMSEvn_1(:), RMSEvn_2(:), RMSEvn_3(:), RMSEvn_4(:), RMSElbfgs(:)], x, 'Colors', 'k', 'Symbol', 'k+', 'DataLim', [0, 300], 'Jitter', 0.1)
ylabel('RMSE'); %title({'RMSE on FULLPIPELINE validation set'});
set(gca, 'FontName', 'Palatino', 'FontSize', 13);
bp = gca;
bp.XAxis.TickLabelInterpreter = 'latex';
saveas(gcf, fullfile(save_dir, 'exp6_test_full.png'));

set(gcf, 'Position', [440, 286, 423, 512]);
% Plot validation set ray-based usm
x = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9] ;
for i = 1:numel(x)
    u = x(i);
    filename = sprintf('test-syn-0.0-%.1f-patchy-val_MS_6comb.mat', u);
    lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
    m_1 = load(fullfile(data_dir_1, filename));
    m_2 = load(fullfile(data_dir_2, filename));
	m_3 = load(fullfile(data_dir_3, filename));
	m_4 = load(fullfile(data_dir_4, filename));
	%m_5 = load(fullfile(data_dir_5, filename));
	%m_6 = load(fullfile(data_dir_6, filename));
	%m_7 = load(fullfile(data_dir_7, filename));
    %m_8 = load(fullfile(data_dir_8, filename));
    [Nimgs,nx, ny] = size(m_1.init_img) ;
    [RMSEvn_1, RMSElbfgs] = RMSE(Nimgs, m_1, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_2, RMSElbfgs] = RMSE(Nimgs, m_2, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_3, RMSElbfgs] = RMSE(Nimgs, m_3, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_4, RMSElbfgs] = RMSE(Nimgs, m_4, lbfgs_m.recon_lbfgs) ;
    %[RMSEvn_5, RMSElbfgs] = RMSE(Nimgs, m_5, lbfgs_m.recon_lbfgs) ;
    %[RMSEvn_6, RMSElbfgs] = RMSE(Nimgs, m_6, lbfgs_m.recon_lbfgs) ;
    %[RMSEvn_7, RMSElbfgs] = RMSE(Nimgs, m_7, lbfgs_m.recon_lbfgs) ;
    %[RMSEvn_8, RMSElbfgs] = RMSE(Nimgs, m_8, lbfgs_m.recon_lbfgs) ;
    rmse_vn_1(i) = median(RMSEvn_1(:));
    rmse_vn_2(i) = median(RMSEvn_2(:));
    rmse_vn_3(i) = median(RMSEvn_3(:));
    rmse_vn_4(i) = median(RMSEvn_4(:));
    %rmse_vn_5(i) = median(RMSEvn_5(:));
    %rmse_vn_6(i) = median(RMSEvn_6(:));
    %rmse_vn_7(i) = median(RMSEvn_7(:));
    %rmse_vn_8(i) = median(RMSEvn_8(:));
    rmse_lbfgs(i)= median(RMSElbfgs(:));
end
clf;
plot(x, rmse_vn_1, '-o', x, rmse_vn_2, '-o', x, rmse_vn_3, '-o', x, rmse_vn_4, '-o', x, rmse_lbfgs, 'k--o'); %title('RMSE on RAY-BASED validation set'); 
legend('VN \lambda_{d}: 10^5, \lambda_{r}: 10^5', 'VN \lambda_{d}: 0, \lambda_{r}: 10^5', 'VN \lambda_{d}: 10^5, \lambda_{r}: 0', 'No regularization', 'LBFGS', 'Location', 'northwest');
xlabel('Undersampling rate'); ylabel('Median RMSE'); xticks(0:0.1:1); 
set(gca, 'FontName', 'Palatino', 'FontSize', 13);
saveas(gcf, fullfile(save_dir, 'exp6_validation_usr_ray.png'));

% Plot validation set ideal time usm
x = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9] ;

for i = 1:numel(x)
    u = x(i);
    filename = sprintf('test-ideal_time-0.0-%.1f-patchy-ideal_time_val.mat', u);
    lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
    m_1 = load(fullfile(data_dir_1, filename));
    m_2 = load(fullfile(data_dir_2, filename));
	m_3 = load(fullfile(data_dir_3, filename));
	m_4 = load(fullfile(data_dir_4, filename));
	%m_5 = load(fullfile(data_dir_5, filename));
	%m_6 = load(fullfile(data_dir_6, filename));
	%m_7 = load(fullfile(data_dir_7, filename));
    %m_8 = load(fullfile(data_dir_8, filename));
    [Nimgs,nx, ny] = size(m_1.init_img) ;
    [RMSEvn_1, RMSElbfgs] = RMSE(Nimgs, m_1, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_2, RMSElbfgs] = RMSE(Nimgs, m_2, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_3, RMSElbfgs] = RMSE(Nimgs, m_3, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_4, RMSElbfgs] = RMSE(Nimgs, m_4, lbfgs_m.recon_lbfgs) ;
    %[RMSEvn_5, RMSElbfgs] = RMSE(Nimgs, m_5, lbfgs_m.recon_lbfgs) ;
    %[RMSEvn_6, RMSElbfgs] = RMSE(Nimgs, m_6, lbfgs_m.recon_lbfgs) ;
    %[RMSEvn_7, RMSElbfgs] = RMSE(Nimgs, m_7, lbfgs_m.recon_lbfgs) ;
    %[RMSEvn_8, RMSElbfgs] = RMSE(Nimgs, m_8, lbfgs_m.recon_lbfgs) ;
    rmse_vn_1(i) = median(RMSEvn_1(:));
    rmse_vn_2(i) = median(RMSEvn_2(:));
    rmse_vn_3(i) = median(RMSEvn_3(:));
    rmse_vn_4(i) = median(RMSEvn_4(:));
    %rmse_vn_5(i) = median(RMSEvn_5(:));
    %rmse_vn_6(i) = median(RMSEvn_6(:));
    %rmse_vn_7(i) = median(RMSEvn_7(:));
    %rmse_vn_8(i) = median(RMSEvn_8(:));
    rmse_lbfgs(i)= median(RMSElbfgs(:));
end
clf;
plot(x, rmse_vn_1, '-o', x, rmse_vn_2, '-o', x, rmse_vn_3, '-o', x, rmse_vn_4, '-o', x, rmse_lbfgs, 'k--o'); %title('RMSE on IDEAL TIME DELAYS validation set'); 
legend('VN \lambda_{d}: 10^5, \lambda_{r}: 10^5', 'VN \lambda_{d}: 0, \lambda_{r}: 10^5', 'VN \lambda_{d}: 10^5, \lambda_{r}: 0', 'No regularization', 'LBFGS', 'Location', 'northwest');
xlabel('Undersampling rate'); ylabel('Median RMSE'); xticks(0:0.1:1); 
set(gca, 'FontName', 'Palatino', 'FontSize', 13);
saveas(gcf, fullfile(save_dir, 'exp6_validation_usr_ideal.png'));


% Plot test set ray based usm
x = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9] ;
for i = 1:numel(x)
    u = x(i);
    filename = sprintf('test-syn-0.0-%.1f-patchy-testset_ideal_MS_32_imgs.mat', u);
    lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
    m_1 = load(fullfile(data_dir_1, filename));
    m_2 = load(fullfile(data_dir_2, filename));
	m_3 = load(fullfile(data_dir_3, filename));
	m_4 = load(fullfile(data_dir_4, filename));
	%m_5 = load(fullfile(data_dir_5, filename));
	%m_6 = load(fullfile(data_dir_6, filename));
	%m_7 = load(fullfile(data_dir_7, filename));
    %m_8 = load(fullfile(data_dir_8, filename));
    [Nimgs,nx, ny] = size(m_1.init_img) ;
    [RMSEvn_1, RMSElbfgs] = RMSE(Nimgs, m_1, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_2, RMSElbfgs] = RMSE(Nimgs, m_2, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_3, RMSElbfgs] = RMSE(Nimgs, m_3, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_4, RMSElbfgs] = RMSE(Nimgs, m_4, lbfgs_m.recon_lbfgs) ;
    rmse_vn_1(i) = median(RMSEvn_1(:));
    rmse_vn_2(i) = median(RMSEvn_2(:));
    rmse_vn_3(i) = median(RMSEvn_3(:));
    rmse_vn_4(i) = median(RMSEvn_4(:));
    rmse_lbfgs(i)= median(RMSElbfgs(:));
end
clf;
plot(x, rmse_vn_1, '-o', x, rmse_vn_2, '-o', x, rmse_vn_3, '-o', x, rmse_vn_4, '-o', x, rmse_lbfgs, 'k--o'); %title('RMSE on RAY-BASED testset set'); 
legend('VN \lambda_{d}: 10^5, \lambda_{r}: 10^5', 'VN \lambda_{d}: 0, \lambda_{r}: 10^5', 'VN \lambda_{d}: 10^5, \lambda_{r}: 0', 'No regularization', 'LBFGS', 'Location', 'northwest');
xlabel('Undersampling rate'); ylabel('Median RMSE'); xticks(0:0.1:1); 
set(gca, 'FontName', 'Palatino', 'FontSize', 13);
saveas(gcf, fullfile(save_dir, 'exp6_test_usr_ray.png'));


% % Plot testset noise ray
% x = [0.0, 0.05, 0.10, 0.15, 0.2, 0.5] ;
% for i = 1:numel(x)
%     u = x(i);
%     filename = sprintf('test-syn-%.1f-0.7-patchy-testset_ideal_MS_32_imgs.mat', u);
%     lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
%     m_1 = load(fullfile(data_dir_1, filename));
%     m_2 = load(fullfile(data_dir_2, filename));
% 	m_3 = load(fullfile(data_dir_3, filename));
% 	m_4 = load(fullfile(data_dir_4, filename));
%     [Nimgs,nx, ny] = size(m_1.init_img) ;
%     [RMSEvn_1, RMSElbfgs] = RMSE(Nimgs, m_1, lbfgs_m.recon_lbfgs) ;
%     [RMSEvn_2, RMSElbfgs] = RMSE(Nimgs, m_2, lbfgs_m.recon_lbfgs) ;
%     [RMSEvn_3, RMSElbfgs] = RMSE(Nimgs, m_3, lbfgs_m.recon_lbfgs) ;
%     [RMSEvn_4, RMSElbfgs] = RMSE(Nimgs, m_4, lbfgs_m.recon_lbfgs) ;
%     rmse_vn_1(i) = median(RMSEvn_1(:));
%     rmse_vn_2(i) = median(RMSEvn_2(:));
%     rmse_vn_3(i) = median(RMSEvn_3(:));
%     rmse_vn_4(i) = median(RMSEvn_4(:));
%     rmse_lbfgs(i)= median(RMSElbfgs(:));
% end
% clf;
% plot(x, rmse_vn_1, '-o', x, rmse_vn_2, '-o', x, rmse_vn_3, '-o', x, rmse_vn_4, '-o', x, rmse_lbfgs, 'k--o'); %title('RMSE on RAY-BASED testset set'); 
% legend('VN $\lambda_{d}: 1e5, \lambda_{r}: 1e5$', 'VN $\lambda_{d}: 0, \lambda_{r}: 1e5$', 'VN $\lambda_{d}: 1e5, \lambda_{r}: 0$', 'No regularization', 'LBFGS', 'Location', 'northwest');
% xlabel('Undersampling rate'); ylabel('Median RMSE'); xticks(0:0.1:1); 
% set(gca, 'FontName', 'Palatino', 'FontSize', 13);
% saveas(gcf, fullfile(save_dir, 'exp6_test_noise_ray.png'));


% Plot testset ideal usr 
x = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9] ;
for i = 1:numel(x)
    u = x(i);
    filename = sprintf('test-ideal_time-0.0-%.1f-patchy-testset_ideal_MS_32_imgs.mat', u);
    lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
    m_1 = load(fullfile(data_dir_1, filename));
    m_2 = load(fullfile(data_dir_2, filename));
	m_3 = load(fullfile(data_dir_3, filename));
	m_4 = load(fullfile(data_dir_4, filename));
    [Nimgs,nx, ny] = size(m_1.init_img) ;
    [RMSEvn_1, RMSElbfgs] = RMSE(Nimgs, m_1, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_2, RMSElbfgs] = RMSE(Nimgs, m_2, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_3, RMSElbfgs] = RMSE(Nimgs, m_3, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_4, RMSElbfgs] = RMSE(Nimgs, m_4, lbfgs_m.recon_lbfgs) ;
    rmse_vn_1(i) = median(RMSEvn_1(:));
    rmse_vn_2(i) = median(RMSEvn_2(:));
    rmse_vn_3(i) = median(RMSEvn_3(:));
    rmse_vn_4(i) = median(RMSEvn_4(:));
    rmse_lbfgs(i)= median(RMSElbfgs(:));
end
clf;
plot(x, rmse_vn_1, '-o', x, rmse_vn_2, '-o', x, rmse_vn_3, '-o', x, rmse_vn_4, '-o', x, rmse_lbfgs, 'k--o'); %title('RMSE on IDEAL TIME DELAYS testset set'); 
legend('VN \lambda_{d}: 10^5, \lambda_{r}: 10^5', 'VN \lambda_{d}: 0, \lambda_{r}: 10^5', 'VN \lambda_{d}: 10^5, \lambda_{r}: 0', 'No regularization', 'LBFGS', 'Location', 'northwest');
xlabel('Undersampling rate'); ylabel('Median RMSE'); xticks(0:0.1:1); 
set(gca, 'FontName', 'Palatino', 'FontSize', 13);
saveas(gcf, fullfile(save_dir, 'exp6_test_usr_ideal.png'));


% % Plot 2 testset ray based
% x = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9] ;
% for i = 1:numel(x)
%     u = x(i);
%     filename = sprintf('test-syn-0.0-%.1f-patchy-testset_ideal_MS_2incl_8_imgs.mat', u);
%     lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
%     m_1 = load(fullfile(data_dir_1, filename));
%     m_2 = load(fullfile(data_dir_2, filename));
% 	m_3 = load(fullfile(data_dir_3, filename));
% 	m_4 = load(fullfile(data_dir_4, filename));
%     [Nimgs,nx, ny] = size(m_1.init_img) ;
%     [RMSEvn_1, RMSElbfgs] = RMSE(Nimgs, m_1, lbfgs_m.recon_lbfgs) ;
%     [RMSEvn_2, RMSElbfgs] = RMSE(Nimgs, m_2, lbfgs_m.recon_lbfgs) ;
%     [RMSEvn_3, RMSElbfgs] = RMSE(Nimgs, m_3, lbfgs_m.recon_lbfgs) ;
%     [RMSEvn_4, RMSElbfgs] = RMSE(Nimgs, m_4, lbfgs_m.recon_lbfgs) ;
%     rmse_vn_1(i) = median(RMSEvn_1(:));
%     rmse_vn_2(i) = median(RMSEvn_2(:));
%     rmse_vn_3(i) = median(RMSEvn_3(:));
%     rmse_vn_4(i) = median(RMSEvn_4(:));
%     rmse_lbfgs(i)= median(RMSElbfgs(:));
% end
% clf;
% plot(x, rmse_vn_1, '-o', x, rmse_vn_2, '-o', x, rmse_vn_3, '-o', x, rmse_vn_4, '-o', x,  x, rmse_lbfgs, 'k--o'); %title({'RMSE on RAY-BASED', '2 inclusions test set'}); 
% legend('VN \lambda_{d}: 10^5, \lambda_{r}: 10^5', 'VN \lambda_{d}: 0, \lambda_{r}: 10^5', 'VN \lambda_{d}: 10^5, \lambda_{r}: 0', 'No regularization', 'LBFGS', 'Location', 'northwest');
% xlabel('Undersampling rate'); ylabel('Median RMSE'); xticks(0:0.1:1);  
% set(gca, 'FontName', 'Palatino', 'FontSize', 13);
% saveas(gcf, fullfile(save_dir, 'exp6_test2_usr_ray.png'));
% 
% 
% % Plot 2 testset ideal based
% x = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9] ;
% for i = 1:numel(x)
%     u = x(i);
%     filename = sprintf('test-ideal_time-0.0-%.1f-patchy-testset_ideal_MS_2incl_8_imgs.mat', u);
%     lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
%     m_1 = load(fullfile(data_dir_1, filename));
%     m_2 = load(fullfile(data_dir_2, filename));
% 	m_3 = load(fullfile(data_dir_3, filename));
% 	m_4 = load(fullfile(data_dir_4, filename));
%     [Nimgs,nx, ny] = size(m_1.init_img) ;
%     [RMSEvn_1, RMSElbfgs] = RMSE(Nimgs, m_1, lbfgs_m.recon_lbfgs) ;
%     [RMSEvn_2, RMSElbfgs] = RMSE(Nimgs, m_2, lbfgs_m.recon_lbfgs) ;
%     [RMSEvn_3, RMSElbfgs] = RMSE(Nimgs, m_3, lbfgs_m.recon_lbfgs) ;
%     [RMSEvn_4, RMSElbfgs] = RMSE(Nimgs, m_4, lbfgs_m.recon_lbfgs) ;
%     rmse_vn_1(i) = median(RMSEvn_1(:));
%     rmse_vn_2(i) = median(RMSEvn_2(:));
%     rmse_vn_3(i) = median(RMSEvn_3(:));
%     rmse_vn_4(i) = median(RMSEvn_4(:));
%     rmse_lbfgs(i)= median(RMSElbfgs(:));
% end
% clf;
% plot(x, rmse_vn_1, '-o', x, rmse_vn_2, '-o', x, rmse_vn_3, '-o', x, rmse_vn_4, '-o', x, rmse_lbfgs, 'k--o'); %title({'RMSE on IDEAL TIME DELAYS', '2 inclusions test set'}); 
% legend('VN \lambda_{d}: 10^5, \lambda_{r}: 10^5', 'VN \lambda_{d}: 0, \lambda_{r}: 10^5', 'VN \lambda_{d}: 10^5, \lambda_{r}: 0', 'No regularization', 'LBFGS', 'Location', 'northwest');
% xlabel('Undersampling rate'); ylabel('Median RMSE'); xticks(0:0.1:1);  
% set(gca, 'FontName', 'Palatino', 'FontSize', 13);
% saveas(gcf, fullfile(save_dir, 'exp6_test2_usr_ideal.png'));


%% Experiment 7 mixed training 

data_dir_1 = '/Volumes/MelSSD/runs/2_20l/eval-vn-120000/' ;
data_dir_2 = '/Volumes/MelSSD/runs/20_mix_ideal_20l/eval-vn-120000/' ;
data_dir_3 = '/Volumes/MelSSD/runs/21_mix_full_20l/eval-vn-120000/';
data_dir_4 = '/Volumes/MelSSD/runs/22_mix_triple_20l/eval-vn-120000/';

% Plot fullpipeline validation
filename = 'test-syn-0.0-0.0-test-fullpipeline_val_64_6comb_64_imgs.mat' ;
lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
m_1 = load(fullfile(data_dir_1, filename));
m_2 = load(fullfile(data_dir_2, filename));
m_3 = load(fullfile(data_dir_3, filename));
m_4 = load(fullfile(data_dir_4, filename));
[Nimgs,nx, ny] = size(m_1.init_img) ;
[RMSEvn_1, RMSElbfgs] = RMSE(Nimgs, m_1, lbfgs_m.recon_lbfgs) ;
[RMSEvn_2, RMSElbfgs] = RMSE(Nimgs, m_2, lbfgs_m.recon_lbfgs) ;
[RMSEvn_3, RMSElbfgs] = RMSE(Nimgs, m_3, lbfgs_m.recon_lbfgs) ;
[RMSEvn_4, RMSElbfgs] = RMSE(Nimgs, m_4, lbfgs_m.recon_lbfgs) ;
x = categorical({'VN Ray', 'VN RayIdeal', 'VN RayFull', 'VN RayIdealFull', 'LBFGS'}) ;
boxplot([RMSEvn_1(:), RMSEvn_2(:), RMSEvn_3(:), RMSEvn_4(:), RMSElbfgs(:)], x, 'Colors', 'k', 'Symbol', 'k+', 'DataLim', [0, 300], 'Jitter', 0.1)
ylabel('RMSE'); %title({'RMSE on FULLPIPELINE validation set'});
set(gca, 'FontName', 'Palatino', 'FontSize', 13);
saveas(gcf, fullfile(save_dir, 'exp7_validation_full.png'));

% Plot fullpipeline test
filename = 'test-syn-0.0-0.0-test-fullpipeline_testset_6comb_32_imgs.mat' ;
lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
m_1 = load(fullfile(data_dir_1, filename));
m_2 = load(fullfile(data_dir_2, filename));
m_3 = load(fullfile(data_dir_3, filename));
m_4 = load(fullfile(data_dir_4, filename));
[Nimgs,nx, ny] = size(m_1.init_img) ;
[RMSEvn_1, RMSElbfgs] = RMSE(Nimgs, m_1, lbfgs_m.recon_lbfgs) ;
[RMSEvn_2, RMSElbfgs] = RMSE(Nimgs, m_2, lbfgs_m.recon_lbfgs) ;
[RMSEvn_3, RMSElbfgs] = RMSE(Nimgs, m_3, lbfgs_m.recon_lbfgs) ;
[RMSEvn_4, RMSElbfgs] = RMSE(Nimgs, m_4, lbfgs_m.recon_lbfgs) ;
x = categorical({'VN Ray', 'VN RayIdeal', 'VN RayFull', 'VN RayIdealFull', 'LBFGS'}) ;
boxplot([RMSEvn_1(:), RMSEvn_2(:), RMSEvn_3(:), RMSEvn_4(:), RMSElbfgs(:)], x, 'Colors', 'k', 'Symbol', 'k+', 'DataLim', [0, 300], 'Jitter', 0.1)
ylabel('RMSE'); %title({'RMSE on FULLPIPELINE validation set'});
set(gca, 'FontName', 'Palatino', 'FontSize', 13);
saveas(gcf, fullfile(save_dir, 'exp7_test_full.png'));

% Plot validation set ray-based usm
x = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9] ;
for i = 1:numel(x)
    u = x(i);
    filename = sprintf('test-syn-0.0-%.1f-patchy-val_MS_6comb.mat', u);
    lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
    m_1 = load(fullfile(data_dir_1, filename));
    m_2 = load(fullfile(data_dir_2, filename));
	m_3 = load(fullfile(data_dir_3, filename));
    m_4 = load(fullfile(data_dir_4, filename));
    [Nimgs,nx, ny] = size(m_1.init_img) ;
    [RMSEvn_1, RMSElbfgs] = RMSE(Nimgs, m_1, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_2, RMSElbfgs] = RMSE(Nimgs, m_2, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_3, RMSElbfgs] = RMSE(Nimgs, m_3, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_4, RMSElbfgs] = RMSE(Nimgs, m_4, lbfgs_m.recon_lbfgs) ;
    rmse_vn_1(i) = median(RMSEvn_1(:));
    rmse_vn_2(i) = median(RMSEvn_2(:));
    rmse_vn_3(i) = median(RMSEvn_3(:));
    rmse_vn_4(i) = median(RMSEvn_4(:));
    rmse_lbfgs(i)= median(RMSElbfgs(:));
end
clf;
plot(x, rmse_vn_1, '-o', x, rmse_vn_2, '-o', x, rmse_vn_3, '-o', x, rmse_vn_4, '-o', x, rmse_lbfgs, 'k--o'); %title('RMSE on RAY-BASED validation set'); 
legend('VN ray-based only', 'VN ray-based + ideal', 'VN ray-based + fullpipeline', 'VN ray + ideal + full', 'LBFGS', 'Location', 'northwest'); xlabel('Undersampling rate'); ylabel('Median RMSE'); xticks(0:0.1:1); 
set(gca, 'FontName', 'Palatino', 'FontSize', 13);
saveas(gcf, fullfile(save_dir, 'exp7_validation_usr_ray.png'));

% Plot validation set ideal time usm
x = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9] ;
for i = 1:numel(x)
    u = x(i);
    filename = sprintf('test-ideal_time-0.0-%.1f-patchy-ideal_time_val.mat', u);
    lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
    m_1 = load(fullfile(data_dir_1, filename));
    m_2 = load(fullfile(data_dir_2, filename));
	m_3 = load(fullfile(data_dir_3, filename));
    m_4 = load(fullfile(data_dir_4, filename));
    [Nimgs,nx, ny] = size(m_1.init_img) ;
    [RMSEvn_1, RMSElbfgs] = RMSE(Nimgs, m_1, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_2, RMSElbfgs] = RMSE(Nimgs, m_2, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_3, RMSElbfgs] = RMSE(Nimgs, m_3, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_4, RMSElbfgs] = RMSE(Nimgs, m_4, lbfgs_m.recon_lbfgs) ;
    rmse_vn_1(i) = median(RMSEvn_1(:));
    rmse_vn_2(i) = median(RMSEvn_2(:));
    rmse_vn_3(i) = median(RMSEvn_3(:));
    rmse_vn_4(i) = median(RMSEvn_4(:));
    rmse_lbfgs(i)= median(RMSElbfgs(:));
end
clf;
plot(x, rmse_vn_1, '-o', x, rmse_vn_2, '-o', x, rmse_vn_3, '-o', x, rmse_vn_4, '-o', x, rmse_lbfgs, 'k--o'); %title('RMSE on IDEAL TIME DELAYS validation set'); 
legend('VN ray-based only', 'VN ray-based + ideal', 'VN ray-based + fullpipeline', 'VN ray + ideal + full', 'LBFGS', 'Location', 'northwest'); xlabel('Undersampling rate'); ylabel('Median RMSE'); xticks(0:0.1:1); 
set(gca, 'FontName', 'Palatino', 'FontSize', 13);
saveas(gcf, fullfile(save_dir, 'exp7_validation_usr_ideal.png'));


% Plot test set ray based usm
x = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9] ;
for i = 1:numel(x)
    u = x(i);
    filename = sprintf('test-syn-0.0-%.1f-patchy-testset_ideal_MS_32_imgs.mat', u);
    lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
    m_1 = load(fullfile(data_dir_1, filename));
    m_2 = load(fullfile(data_dir_2, filename));
	m_3 = load(fullfile(data_dir_3, filename));
    m_4 = load(fullfile(data_dir_4, filename));
    [Nimgs,nx, ny] = size(m_1.init_img) ;
    [RMSEvn_1, RMSElbfgs] = RMSE(Nimgs, m_1, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_2, RMSElbfgs] = RMSE(Nimgs, m_2, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_3, RMSElbfgs] = RMSE(Nimgs, m_3, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_4, RMSElbfgs] = RMSE(Nimgs, m_4, lbfgs_m.recon_lbfgs) ;
    rmse_vn_1(i) = median(RMSEvn_1(:));
    rmse_vn_2(i) = median(RMSEvn_2(:));
    rmse_vn_3(i) = median(RMSEvn_3(:));
    rmse_vn_4(i) = median(RMSEvn_4(:));
    rmse_lbfgs(i)= median(RMSElbfgs(:));
end
clf;
plot(x, rmse_vn_1, '-o', x, rmse_vn_2, '-o', x, rmse_vn_3, '-o', x, rmse_vn_4, '-o', x, rmse_lbfgs, 'k--o'); %title('RMSE on RAY-BASED testset set'); 
legend('VN ray-based only', 'VN ray-based + ideal', 'VN ray-based + fullpipeline', 'VN ray + ideal + full', 'LBFGS', 'Location', 'northwest'); xlabel('Undersampling rate'); ylabel('Median RMSE'); xticks(0:0.1:1); 
set(gca, 'FontName', 'Palatino', 'FontSize', 13);
saveas(gcf, fullfile(save_dir, 'exp7_test_usr_ray.png'));



% Plot testset ideal usr 
x = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9] ;
for i = 1:numel(x)
    u = x(i);
    filename = sprintf('test-ideal_time-0.0-%.1f-patchy-testset_ideal_MS_32_imgs.mat', u);
    lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
    m_1 = load(fullfile(data_dir_1, filename));
    m_2 = load(fullfile(data_dir_2, filename));
	m_3 = load(fullfile(data_dir_3, filename));
    m_4 = load(fullfile(data_dir_4, filename));
    [Nimgs,nx, ny] = size(m_1.init_img) ;
    [RMSEvn_1, RMSElbfgs] = RMSE(Nimgs, m_1, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_2, RMSElbfgs] = RMSE(Nimgs, m_2, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_3, RMSElbfgs] = RMSE(Nimgs, m_3, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_4, RMSElbfgs] = RMSE(Nimgs, m_4, lbfgs_m.recon_lbfgs) ;
    rmse_vn_1(i) = median(RMSEvn_1(:));
    rmse_vn_2(i) = median(RMSEvn_2(:));
    rmse_vn_3(i) = median(RMSEvn_3(:));
    rmse_vn_4(i) = median(RMSEvn_4(:));
    rmse_lbfgs(i)= median(RMSElbfgs(:));
end
clf;
plot(x, rmse_vn_1, '-o', x, rmse_vn_2, '-o', x, rmse_vn_3, '-o', x, rmse_vn_4, '-o', x, rmse_lbfgs, 'k--o'); %title('RMSE on IDEAL TIME DELAYS testset set'); 
legend('VN ray-based only', 'VN ray-based + ideal', 'VN ray-based + fullpipeline', 'VN ray + ideal + full', 'LBFGS', 'Location', 'northwest'); xlabel('Undersampling rate'); ylabel('Median RMSE'); xticks(0:0.1:1); 
set(gca, 'FontName', 'Palatino', 'FontSize', 13);
saveas(gcf, fullfile(save_dir, 'exp7_test_usr_ideal.png'));


% Plot 2 testset ray based
x = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9] ;
for i = 1:numel(x)
    u = x(i);
    filename = sprintf('test-syn-0.0-%.1f-patchy-testset_ideal_MS_2incl_8_imgs.mat', u);
    lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
    m_1 = load(fullfile(data_dir_1, filename));
    m_2 = load(fullfile(data_dir_2, filename));
	m_3 = load(fullfile(data_dir_3, filename));
    m_4 = load(fullfile(data_dir_4, filename));
    [Nimgs,nx, ny] = size(m_1.init_img) ;
    [RMSEvn_1, RMSElbfgs] = RMSE(Nimgs, m_1, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_2, RMSElbfgs] = RMSE(Nimgs, m_2, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_3, RMSElbfgs] = RMSE(Nimgs, m_3, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_4, RMSElbfgs] = RMSE(Nimgs, m_4, lbfgs_m.recon_lbfgs) ;
    rmse_vn_1(i) = median(RMSEvn_1(:));
    rmse_vn_2(i) = median(RMSEvn_2(:));
    rmse_vn_3(i) = median(RMSEvn_3(:));
    rmse_vn_4(i) = median(RMSEvn_4(:));
    rmse_lbfgs(i)= median(RMSElbfgs(:));
end
clf;
plot(x, rmse_vn_1, '-o', x, rmse_vn_2, '-o', x, rmse_vn_3, '-o', x, rmse_vn_4, '-o', x, rmse_lbfgs, 'k--o'); %title({'RMSE on RAY-BASED', '2 inclusions test set'}); 
legend('VN ray-based only', 'VN ray-based + ideal', 'VN ray-based + fullpipeline', 'VN ray + ideal + full', 'LBFGS', 'Location', 'northwest'); xlabel('Undersampling rate'); ylabel('Median RMSE'); xticks(0:0.1:1); 
set(gca, 'FontName', 'Palatino', 'FontSize', 13);
saveas(gcf, fullfile(save_dir, 'exp7_test2_usr_ray.png'));


% Plot 2 testset ideal based
x = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9] ;
for i = 1:numel(x)
    u = x(i);
    filename = sprintf('test-ideal_time-0.0-%.1f-patchy-testset_ideal_MS_2incl_8_imgs.mat', u);
    lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
    m_1 = load(fullfile(data_dir_1, filename));
    m_2 = load(fullfile(data_dir_2, filename));
	m_3 = load(fullfile(data_dir_3, filename));
    m_4 = load(fullfile(data_dir_4, filename));
    [Nimgs,nx, ny] = size(m_1.init_img) ;
    [RMSEvn_1, RMSElbfgs] = RMSE(Nimgs, m_1, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_2, RMSElbfgs] = RMSE(Nimgs, m_2, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_3, RMSElbfgs] = RMSE(Nimgs, m_3, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_4, RMSElbfgs] = RMSE(Nimgs, m_4, lbfgs_m.recon_lbfgs) ;
    rmse_vn_1(i) = median(RMSEvn_1(:));
    rmse_vn_2(i) = median(RMSEvn_2(:));
    rmse_vn_3(i) = median(RMSEvn_3(:));
    rmse_vn_4(i) = median(RMSEvn_4(:));
    rmse_lbfgs(i)= median(RMSElbfgs(:));
end
clf;
plot(x, rmse_vn_1, '-o', x, rmse_vn_2, '-o', x, rmse_vn_3, '-o', x, rmse_vn_4, '-o', x, rmse_lbfgs, 'k--o'); %title({'RMSE on IDEAL TIME DELAYS', '2 inclusions test set'}); 
legend('VN ray-based only', 'VN ray-based + ideal', 'VN ray-based + fullpipeline', 'VN ray + ideal + full', 'LBFGS', 'Location', 'northwest'); xlabel('Undersampling rate'); ylabel('Median RMSE'); xticks(0:0.1:1); 
set(gca, 'FontName', 'Palatino', 'FontSize', 13);
saveas(gcf, fullfile(save_dir, 'exp7_test2_usr_ideal.png'));


%% Experiment 8 mixed training - reg

data_dir_1 = '/Volumes/MelSSD/runs/13_20l_reg_0_1e5/eval-vn-120000/' ;
data_dir_2 = '/Volumes/MelSSD/runs/32_mix_ideal_20l_reg/eval-vn-120000/' ;
data_dir_3 = '/Volumes/MelSSD/runs/34_mix_full_20l_reg/eval-vn-120000/';
data_dir_4 = '/Volumes/MelSSD/runs/33_mix_triple_20l_reg/eval-vn-120000/';

% Plot fullpipeline validation
filename = 'test-syn-0.0-0.0-test-fullpipeline_val_64_6comb_64_imgs.mat' ;
lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
m_1 = load(fullfile(data_dir_1, filename));
m_2 = load(fullfile(data_dir_2, filename));
m_3 = load(fullfile(data_dir_3, filename));
m_4 = load(fullfile(data_dir_4, filename));
[Nimgs,nx, ny] = size(m_1.init_img) ;
[RMSEvn_1, RMSElbfgs] = RMSE(Nimgs, m_1, lbfgs_m.recon_lbfgs) ;
[RMSEvn_2, RMSElbfgs] = RMSE(Nimgs, m_2, lbfgs_m.recon_lbfgs) ;
[RMSEvn_3, RMSElbfgs] = RMSE(Nimgs, m_3, lbfgs_m.recon_lbfgs) ;
[RMSEvn_4, RMSElbfgs] = RMSE(Nimgs, m_4, lbfgs_m.recon_lbfgs) ;
x = categorical({'Ray', 'RayIdeal', 'RayFull', 'RayIdealFull', 'LBFGS'}) ;
boxplot([RMSEvn_1(:), RMSEvn_2(:), RMSEvn_3(:), RMSEvn_4(:), RMSElbfgs(:)], x, 'Colors', 'k', 'Symbol', 'k+', 'DataLim', [0, 300], 'Jitter', 0.1)
ylabel('RMSE'); %title({'RMSE on FULLPIPELINE validation set'});
set(gca, 'FontName', 'Palatino', 'FontSize', 13);
saveas(gcf, fullfile(save_dir, 'exp8_validation_full.png'));
%%
% Plot fullpipeline test
filename = 'test-syn-0.0-0.0-test-fullpipeline_testset_6comb_32_imgs.mat' ;
lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
m_1 = load(fullfile(data_dir_1, filename));
m_2 = load(fullfile(data_dir_2, filename));
m_3 = load(fullfile(data_dir_3, filename));
m_4 = load(fullfile(data_dir_4, filename));
[Nimgs,nx, ny] = size(m_1.init_img) ;
[RMSEvn_1, RMSElbfgs] = RMSE(Nimgs, m_1, lbfgs_m.recon_lbfgs) ;
[RMSEvn_2, RMSElbfgs] = RMSE(Nimgs, m_2, lbfgs_m.recon_lbfgs) ;
[RMSEvn_3, RMSElbfgs] = RMSE(Nimgs, m_3, lbfgs_m.recon_lbfgs) ;
[RMSEvn_4, RMSElbfgs] = RMSE(Nimgs, m_4, lbfgs_m.recon_lbfgs) ;
x = categorical({'Ray', 'RayIdeal', 'RayFull', 'RayIdealFull', 'LBFGS'}) ;
boxplot([RMSEvn_1(:), RMSEvn_2(:), RMSEvn_3(:), RMSEvn_4(:), RMSElbfgs(:)], x, 'Colors', 'k', 'Symbol', 'k+', 'DataLim', [0, 300], 'Jitter', 0.1)
ylabel('RMSE'); %title({'RMSE on FULLPIPELINE validation set'});
set(gca, 'FontName', 'Palatino', 'FontSize', 13);
saveas(gcf, fullfile(save_dir, 'exp8_test_full.png'));
%%
% Plot validation set ray-based usm
x = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9] ;
for i = 1:numel(x)
    u = x(i);
    filename = sprintf('test-syn-0.0-%.1f-patchy-val_MS_6comb.mat', u);
    lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
    m_1 = load(fullfile(data_dir_1, filename));
    m_2 = load(fullfile(data_dir_2, filename));
	m_3 = load(fullfile(data_dir_3, filename));
    m_4 = load(fullfile(data_dir_4, filename));
    [Nimgs,nx, ny] = size(m_1.init_img) ;
    [RMSEvn_1, RMSElbfgs] = RMSE(Nimgs, m_1, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_2, RMSElbfgs] = RMSE(Nimgs, m_2, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_3, RMSElbfgs] = RMSE(Nimgs, m_3, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_4, RMSElbfgs] = RMSE(Nimgs, m_4, lbfgs_m.recon_lbfgs) ;
    rmse_vn_1(i) = median(RMSEvn_1(:));
    rmse_vn_2(i) = median(RMSEvn_2(:));
    rmse_vn_3(i) = median(RMSEvn_3(:));
    rmse_vn_4(i) = median(RMSEvn_4(:));
    rmse_lbfgs(i)= median(RMSElbfgs(:));
end
clf;
plot(x, rmse_vn_1, '-o', x, rmse_vn_2, '-o', x, rmse_vn_3, '-o', x, rmse_vn_4, '-o', x, rmse_lbfgs, 'k--o'); %title('RMSE on RAY-BASED validation set'); 
legend('Ray', 'RayIdeal', 'RayFull', 'RayIdealFull', 'LBFGS', 'Location', 'northwest'); xlabel('Undersampling rate'); ylabel('Median RMSE'); xticks(0:0.1:1); 
set(gca, 'FontName', 'Palatino', 'FontSize', 13);
saveas(gcf, fullfile(save_dir, 'exp8_validation_usr_ray.png'));

% Plot validation set ideal time usm
x = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9] ;
for i = 1:numel(x)
    u = x(i);
    filename = sprintf('test-ideal_time-0.0-%.1f-patchy-ideal_time_val.mat', u);
    lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
    m_1 = load(fullfile(data_dir_1, filename));
    m_2 = load(fullfile(data_dir_2, filename));
	m_3 = load(fullfile(data_dir_3, filename));
    m_4 = load(fullfile(data_dir_4, filename));
    [Nimgs,nx, ny] = size(m_1.init_img) ;
    [RMSEvn_1, RMSElbfgs] = RMSE(Nimgs, m_1, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_2, RMSElbfgs] = RMSE(Nimgs, m_2, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_3, RMSElbfgs] = RMSE(Nimgs, m_3, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_4, RMSElbfgs] = RMSE(Nimgs, m_4, lbfgs_m.recon_lbfgs) ;
    rmse_vn_1(i) = median(RMSEvn_1(:));
    rmse_vn_2(i) = median(RMSEvn_2(:));
    rmse_vn_3(i) = median(RMSEvn_3(:));
    rmse_vn_4(i) = median(RMSEvn_4(:));
    rmse_lbfgs(i)= median(RMSElbfgs(:));
end
clf;
plot(x, rmse_vn_1, '-o', x, rmse_vn_2, '-o', x, rmse_vn_3, '-o', x, rmse_vn_4, '-o', x, rmse_lbfgs, 'k--o'); %title('RMSE on IDEAL TIME DELAYS validation set'); 
legend('Ray', 'RayIdeal', 'RayFull', 'RayIdealFull', 'LBFGS', 'Location', 'northwest'); xlabel('Undersampling rate'); ylabel('Median RMSE'); xticks(0:0.1:1); 
set(gca, 'FontName', 'Palatino', 'FontSize', 13);
saveas(gcf, fullfile(save_dir, 'exp8_validation_usr_ideal.png'));


% Plot test set ray based usm
x = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9] ;
for i = 1:numel(x)
    u = x(i);
    filename = sprintf('test-syn-0.0-%.1f-patchy-testset_ideal_MS_32_imgs.mat', u);
    lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
    m_1 = load(fullfile(data_dir_1, filename));
    m_2 = load(fullfile(data_dir_2, filename));
	m_3 = load(fullfile(data_dir_3, filename));
    m_4 = load(fullfile(data_dir_4, filename));
    [Nimgs,nx, ny] = size(m_1.init_img) ;
    [RMSEvn_1, RMSElbfgs] = RMSE(Nimgs, m_1, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_2, RMSElbfgs] = RMSE(Nimgs, m_2, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_3, RMSElbfgs] = RMSE(Nimgs, m_3, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_4, RMSElbfgs] = RMSE(Nimgs, m_4, lbfgs_m.recon_lbfgs) ;
    rmse_vn_1(i) = median(RMSEvn_1(:));
    rmse_vn_2(i) = median(RMSEvn_2(:));
    rmse_vn_3(i) = median(RMSEvn_3(:));
    rmse_vn_4(i) = median(RMSEvn_4(:));
    rmse_lbfgs(i)= median(RMSElbfgs(:));
end
clf;
plot(x, rmse_vn_1, '-o', x, rmse_vn_2, '-o', x, rmse_vn_3, '-o', x, rmse_vn_4, '-o', x, rmse_lbfgs, 'k--o'); %title('RMSE on RAY-BASED testset set'); 
legend('Ray', 'RayIdeal', 'RayFull', 'RayIdealFull', 'LBFGS', 'Location', 'northwest'); xlabel('Undersampling rate'); ylabel('Median RMSE'); xticks(0:0.1:1); 
set(gca, 'FontName', 'Palatino', 'FontSize', 13);
saveas(gcf, fullfile(save_dir, 'exp8_test_usr_ray.png'));


% Plot testset ideal usr 
x = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9] ;
for i = 1:numel(x)
    u = x(i);
    filename = sprintf('test-ideal_time-0.0-%.1f-patchy-testset_ideal_MS_32_imgs.mat', u);
    lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
    m_1 = load(fullfile(data_dir_1, filename));
    m_2 = load(fullfile(data_dir_2, filename));
	m_3 = load(fullfile(data_dir_3, filename));
    m_4 = load(fullfile(data_dir_4, filename));
    [Nimgs,nx, ny] = size(m_1.init_img) ;
    [RMSEvn_1, RMSElbfgs] = RMSE(Nimgs, m_1, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_2, RMSElbfgs] = RMSE(Nimgs, m_2, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_3, RMSElbfgs] = RMSE(Nimgs, m_3, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_4, RMSElbfgs] = RMSE(Nimgs, m_4, lbfgs_m.recon_lbfgs) ;
    rmse_vn_1(i) = median(RMSEvn_1(:));
    rmse_vn_2(i) = median(RMSEvn_2(:));
    rmse_vn_3(i) = median(RMSEvn_3(:));
    rmse_vn_4(i) = median(RMSEvn_4(:));
    rmse_lbfgs(i)= median(RMSElbfgs(:));
end
clf;
plot(x, rmse_vn_1, '-o', x, rmse_vn_2, '-o', x, rmse_vn_3, '-o', x, rmse_vn_4, '-o', x, rmse_lbfgs, 'k--o'); %title('RMSE on IDEAL TIME DELAYS testset set'); 
legend('Ray', 'RayIdeal', 'RayFull', 'RayIdealFull', 'LBFGS', 'Location', 'northwest'); xlabel('Undersampling rate'); ylabel('Median RMSE'); xticks(0:0.1:1); 
set(gca, 'FontName', 'Palatino', 'FontSize', 13);
saveas(gcf, fullfile(save_dir, 'exp8_test_usr_ideal.png'));




%% Experiment 9 - base model against all the others
data_dir_1 = '/Volumes/MelSSD/runs/13_20l_reg_0_1e5/eval-vn-120000/' ;
data_dir_2 = '/Volumes/MelSSD/runs/fixedT/eval-vn-120000/' ;
data_dir_3  = '/Volumes/MelSSD/runs/nexp/eval-vn-120000/' ;
%data_dir_4 = '/Volumes/MelSSD/runs/25_20l_reg_0_1e5_T/eval-vn-120000/' ;
%%
% Plot fullpipeline validation
filename = 'test-syn-0.0-0.0-test-fullpipeline_val_64_6comb_64_imgs.mat' ;
lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
m_1 = load(fullfile(data_dir_1, filename));
m_2 = load(fullfile(data_dir_2, filename));
m_3 = load(fullfile(data_dir_3, filename));
[Nimgs,nx, ny] = size(m_1.init_img) ;
[RMSEvn_1, RMSElbfgs] = RMSE(Nimgs, m_1, lbfgs_m.recon_lbfgs) ;
[RMSEvn_2, RMSElbfgs] = RMSE(Nimgs, m_2, lbfgs_m.recon_lbfgs) ;
[RMSEvn_3, RMSElbfgs] = RMSE(Nimgs, m_3, lbfgs_m.recon_lbfgs) ;
x = categorical(1:4, 1:4, {'LBFGS', 'RegExp', 'NoRegExp', 'NoRegNoExp'}) ;
boxplot([RMSElbfgs(:), RMSEvn_1(:), RMSEvn_2(:), RMSEvn_3(:)], x, 'Colors', 'k', 'Symbol', 'k+', 'DataLim', [0, 300], 'Jitter', 0.1)
ylabel('RMSE'); %title({'RMSE on FULLPIPELINE validation set'});
bp = gca;
bp.XAxis.TickLabelInterpreter = 'latex';
set(gca, 'FontName', 'Palatino', 'FontSize', 13);
saveas(gcf, fullfile(save_dir, 'exp9_validation_full.png'));
%%
% Plot fullpipeline test
filename = 'test-syn-0.0-0.0-test-fullpipeline_testset_6comb_32_imgs.mat' ;
lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
m_1 = load(fullfile(data_dir_1, filename));
m_2 = load(fullfile(data_dir_2, filename));
m_3 = load(fullfile(data_dir_3, filename));
[Nimgs,nx, ny] = size(m_1.init_img) ;
[RMSEvn_1, RMSElbfgs] = RMSE(Nimgs, m_1, lbfgs_m.recon_lbfgs) ;
[RMSEvn_2, RMSElbfgs] = RMSE(Nimgs, m_2, lbfgs_m.recon_lbfgs) ;
[RMSEvn_3, RMSElbfgs] = RMSE(Nimgs, m_3, lbfgs_m.recon_lbfgs) ;
x = categorical(1:4, 1:4, {'LBFGS', 'RegExp', 'NoRegExp', 'NoRegNoExp'}) ;
boxplot([RMSElbfgs(:), RMSEvn_1(:), RMSEvn_2(:), RMSEvn_3(:)], x, 'Colors', 'k', 'Symbol', 'k+', 'DataLim', [0, 300], 'Jitter', 0.1)
ylabel('RMSE'); %title({'RMSE on FULLPIPELINE validation set'});
bp = gca;
bp.XAxis.TickLabelInterpreter = 'latex';
set(gca, 'FontName', 'Palatino', 'FontSize', 13);
saveas(gcf, fullfile(save_dir, 'exp9_test_full.png'));
%%

% Plot validation set ray-based usm
x = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9] ;
for i = 1:numel(x)
    u = x(i);
    filename = sprintf('test-syn-0.0-%.1f-patchy-val_MS_6comb.mat', u);
    lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
m_1 = load(fullfile(data_dir_1, filename));
m_2 = load(fullfile(data_dir_2, filename));
m_3 = load(fullfile(data_dir_3, filename));
[Nimgs,nx, ny] = size(m_1.init_img) ;
[RMSEvn_1, RMSElbfgs] = RMSE(Nimgs, m_1, lbfgs_m.recon_lbfgs) ;
[RMSEvn_2, RMSElbfgs] = RMSE(Nimgs, m_2, lbfgs_m.recon_lbfgs) ;
[RMSEvn_3, RMSElbfgs] = RMSE(Nimgs, m_3, lbfgs_m.recon_lbfgs) ;
    rmse_vn_1(i) = median(RMSEvn_1(:));
    rmse_vn_2(i) = median(RMSEvn_2(:));
    rmse_vn_3(i) = median(RMSEvn_3(:));
    rmse_lbfgs(i)= median(RMSElbfgs(:));
end
clf;
plot(x, rmse_vn_1, '-o', x, rmse_vn_2, '-o', x, rmse_vn_3, '-o', x, rmse_lbfgs, 'k--o'); %title('RMSE on RAY-BASED validation set'); 
legend('RegExp', 'NoRegExp', 'NoRegNoExp', 'LBFGS', 'Location', 'northwest'); xlabel('Undersampling rate'); ylabel('Median RMSE'); xticks(0:0.1:1); 
set(gca, 'FontName', 'Palatino', 'FontSize', 13);
saveas(gcf, fullfile(save_dir, 'exp9_validation_usr_ray.png'));

% Plot validation set ideal time usm
x = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9] ;
for i = 1:numel(x)
    u = x(i);
    filename = sprintf('test-ideal_time-0.0-%.1f-patchy-ideal_time_val.mat', u);
    lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
m_1 = load(fullfile(data_dir_1, filename));
m_2 = load(fullfile(data_dir_2, filename));
m_3 = load(fullfile(data_dir_3, filename));
[Nimgs,nx, ny] = size(m_1.init_img) ;
[RMSEvn_1, RMSElbfgs] = RMSE(Nimgs, m_1, lbfgs_m.recon_lbfgs) ;
[RMSEvn_2, RMSElbfgs] = RMSE(Nimgs, m_2, lbfgs_m.recon_lbfgs) ;
[RMSEvn_3, RMSElbfgs] = RMSE(Nimgs, m_3, lbfgs_m.recon_lbfgs) ;
    rmse_vn_1(i) = median(RMSEvn_1(:));
    rmse_vn_2(i) = median(RMSEvn_2(:));
    rmse_vn_3(i) = median(RMSEvn_3(:));
    rmse_lbfgs(i)= median(RMSElbfgs(:));
end
clf;
plot(x, rmse_vn_1, '-o', x, rmse_vn_2, '-o', x, rmse_vn_3, '-o', x, rmse_lbfgs, 'k--o'); %title('RMSE on RAY-BASED validation set'); 
legend('RegExp', 'NoRegExp', 'NoRegNoExp', 'LBFGS', 'Location', 'northwest'); xlabel('Undersampling rate'); ylabel('Median RMSE'); xticks(0:0.1:1); 
set(gca, 'FontName', 'Palatino', 'FontSize', 13);
saveas(gcf, fullfile(save_dir, 'exp9_validation_usr_ideal.png'));


% Plot test set ray based usm
x = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9] ;
for i = 1:numel(x)
    u = x(i);
    filename = sprintf('test-syn-0.0-%.1f-patchy-testset_ideal_MS_32_imgs.mat', u);
    lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
m_1 = load(fullfile(data_dir_1, filename));
m_2 = load(fullfile(data_dir_2, filename));
m_3 = load(fullfile(data_dir_3, filename));
[Nimgs,nx, ny] = size(m_1.init_img) ;
[RMSEvn_1, RMSElbfgs] = RMSE(Nimgs, m_1, lbfgs_m.recon_lbfgs) ;
[RMSEvn_2, RMSElbfgs] = RMSE(Nimgs, m_2, lbfgs_m.recon_lbfgs) ;
[RMSEvn_3, RMSElbfgs] = RMSE(Nimgs, m_3, lbfgs_m.recon_lbfgs) ;
    rmse_vn_1(i) = median(RMSEvn_1(:));
    rmse_vn_2(i) = median(RMSEvn_2(:));
    rmse_vn_3(i) = median(RMSEvn_3(:));
    rmse_lbfgs(i)= median(RMSElbfgs(:));
end
clf;
plot(x, rmse_vn_1, '-o', x, rmse_vn_2, '-o', x, rmse_vn_3, '-o', x, rmse_lbfgs, 'k--o'); %title('RMSE on RAY-BASED validation set'); 
legend('RegExp', 'NoRegExp', 'NoRegNoExp', 'LBFGS', 'Location', 'northwest'); xlabel('Undersampling rate'); ylabel('Median RMSE'); xticks(0:0.1:1); 
set(gca, 'FontName', 'Palatino', 'FontSize', 13);
saveas(gcf, fullfile(save_dir, 'exp9_test_usr_ray.png'));

% % Plot testset noise ray
% x = [0.0, 0.05, 0.10, 0.15, 0.2, 0.5] ;
% for i = 1:numel(x)
%     u = x(i);
%     filename = sprintf('test-syn-%.1f-0.7-patchy-testset_ideal_MS_32_imgs.mat', u);
%     lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
% m_1 = load(fullfile(data_dir_1, filename));
% m_2 = load(fullfile(data_dir_2, filename));
% m_3 = load(fullfile(data_dir_3, filename));
% [Nimgs,nx, ny] = size(m_1.init_img) ;
% [RMSEvn_1, RMSElbfgs] = RMSE(Nimgs, m_1, lbfgs_m.recon_lbfgs) ;
% [RMSEvn_2, RMSElbfgs] = RMSE(Nimgs, m_2, lbfgs_m.recon_lbfgs) ;
% [RMSEvn_3, RMSElbfgs] = RMSE(Nimgs, m_3, lbfgs_m.recon_lbfgs) ;
%     rmse_vn_1(i) = median(RMSEvn_1(:));
%     rmse_vn_2(i) = median(RMSEvn_2(:));
%     rmse_vn_3(i) = median(RMSEvn_3(:));
%     rmse_lbfgs(i)= median(RMSElbfgs(:));
% end
% clf;
% plot(x, rmse_vn_10, '-o', x, rmse_vn_20, '-o',x, rmse_lbfgs, 'k--o'); %title({'RMSE on RAY-BASED test set','undersampling rate 0.7'});
% legend('Model A', 'Model B', 'Model C', 'LBFGS', 'Location', 'northwest');xlabel('Noise rate'); ylabel('Median RMSE'); xticks(0:0.1:1); 
% set(gca, 'FontName', 'Palatino', 'FontSize', 13);
% saveas(gcf, fullfile(save_dir, 'exp9_test_noise.png'));

% Plot testset ideal usr 
x = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9] ;
for i = 1:numel(x)
    u = x(i);
    filename = sprintf('test-ideal_time-0.0-%.1f-patchy-testset_ideal_MS_32_imgs.mat', u);
    lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
m_1 = load(fullfile(data_dir_1, filename));
m_2 = load(fullfile(data_dir_2, filename));
m_3 = load(fullfile(data_dir_3, filename));
[Nimgs,nx, ny] = size(m_1.init_img) ;
[RMSEvn_1, RMSElbfgs] = RMSE(Nimgs, m_1, lbfgs_m.recon_lbfgs) ;
[RMSEvn_2, RMSElbfgs] = RMSE(Nimgs, m_2, lbfgs_m.recon_lbfgs) ;
[RMSEvn_3, RMSElbfgs] = RMSE(Nimgs, m_3, lbfgs_m.recon_lbfgs) ;
    rmse_vn_1(i) = median(RMSEvn_1(:));
    rmse_vn_2(i) = median(RMSEvn_2(:));
    rmse_vn_3(i) = median(RMSEvn_3(:));
    rmse_lbfgs(i)= median(RMSElbfgs(:));
end
clf;
plot(x, rmse_vn_1, '-o', x, rmse_vn_2, '-o', x, rmse_vn_3, '-o', x, rmse_lbfgs, 'k--o'); %title('RMSE on RAY-BASED validation set'); 
legend('RegExp', 'NoRegExp', 'NoRegNoExp', 'LBFGS', 'Location', 'northwest'); xlabel('Undersampling rate'); ylabel('Median RMSE'); xticks(0:0.1:1); 
set(gca, 'FontName', 'Palatino', 'FontSize', 13);
saveas(gcf, fullfile(save_dir, 'exp9_test_usr_ideal.png'));
%%
% opt plots
o1 = load(fullfile(data_dir_1, 'opt-eval/residuals.mat'));
o2 = load(fullfile(data_dir_2, 'opt-eval/residuals.mat'));
o3 = load(fullfile(data_dir_3, 'opt-eval/residuals.mat'));
%%
plot(0:20, mean(o1.data_res, 1), 0:20, mean(o2.data_res, 1), 0:20, median(o3.data_res, 1) ) ;
xlabel('Layer'); ylabel('Mean data residual'); set(gca, 'FontName', 'Palatino', 'FontSize', 13); 
legend('RegExp', 'NoRegExp', 'NoRegNoExp', 'Location', 'northwest');
saveas(gcf, fullfile(save_dir, 'exp9_data_res.png'));

plot(0:20, mean(o1.img_res, 1), 0:20, mean(o2.img_res, 1), 0:20, mean(o3.img_res, 1)) ;
xlabel('Layer'); 
ylabel('Mean image residual'); 
set(gca, 'FontName', 'Palatino', 'FontSize', 13); 
legend('RegExp', 'NoRegExp', 'NoRegNoExp', 'Location', 'northwest');
saveas(gcf, fullfile(save_dir, 'exp9_img_res.png')) ;

% %%
% plot(0:20, mean(sum(abs(o1.d_grad), [3,4]), 1), 0:20, mean(sum(abs(o2.d_grad), [3,4]), 1), 0:20, mean(sum(abs(o3.d_grad), [3,4]), 1), 0:20, mean(sum(abs(o4.d_grad), [3,4]), 1)) ; 
% xlabel('Layer') ; ylabel('||\nabla_d^{(layer)}||_1'); set(gca, 'FontName', 'Palatino', 'FontSize', 13);
% legend('NoRegFixed', 'RegFixed', 'NoRegDecreasing', 'RegDecreasing', 'Location', 'northwest');
% set(gca, 'FontName', 'Palatino', 'FontSize', 13); 
% saveas(gcf, fullfile(save_dir, 'exp9_d_grad.png'));
% 
% plot(0:20, mean(sum(abs(o1.r_grad), [3, 4]), 1), 0:20, mean(sum(abs(o2.r_grad), [3, 4]), 1), 0:20, mean(sum(abs(o3.r_grad), [3, 4]), 1), 0:20, mean(sum(abs(o4.r_grad), [3, 4]), 1));
% xlabel('Layer') ; ylabel('||\nabla_r^{(layer)}||_1'); set(gca, 'FontName', 'Palatino', 'FontSize', 13);
% legend('NoRegFixed', 'RegFixed', 'NoRegDecreasing', 'RegDecreasing', 'Location', 'northwest');
% set(gca, 'FontName', 'Palatino', 'FontSize', 13); 
% saveas(gcf, fullfile(save_dir, 'exp9_r_grad.png'));
% for i=1:64
%     i
% for l=1:21
%             c = corrcoef(o1.d_grad(i,l, :, :), o1.r_grad(i,l, :, :));
%             o1.corr_r_d(i,l) =  c(1,2);
%             c = corrcoef(o2.d_grad(i,l, :, :), o2.r_grad(i,l, :, :));
%             o2.corr_r_d(i,l) =  c(1,2);
%             c = corrcoef(o3.d_grad(i,l, :, :), o3.r_grad(i,l, :, :));
%             o3.corr_r_d(i,l) =  c(1,2);
%             c = corrcoef(o4.d_grad(i,l, :, :), o4.r_grad(i,l, :, :));
%             o4.corr_r_d(i,l) =  c(1,2);
% end
% end        
% plot(0:20, mean(o1.corr_r_d,1), 0:20, mean(o2.corr_r_d,1), 0:20, mean(o3.corr_r_d,1), 0:20, mean(o4.corr_r_d,1)); 
% xlabel('Layer') ; ylabel('Correlation coefficient'); 
% legend('NoRegFixed', 'RegFixed', 'NoRegDecreasing', 'RegDecreasing', 'Location', 'northwest');
% set(gca, 'FontName', 'Palatino', 'FontSize', 13);
% saveas(gcf, fullfile(save_dir, 'exp9_corr.png'));
%%
% opt plots full
o1 = load(fullfile(data_dir_1, 'opt-eval-full/residuals.mat'));
o2 = load(fullfile(data_dir_2, 'opt-eval-full/residuals.mat'));
o3 = load(fullfile(data_dir_3, 'opt-eval-full/residuals.mat'));
%%
plot(0:20, mean(o1.data_res, 1), 0:20, mean(o2.data_res, 1), 0:20, median(o3.data_res, 1) ) ;
xlabel('Layer'); ylabel('Mean data residual'); set(gca, 'FontName', 'Palatino', 'FontSize', 13); 
legend('RegExp', 'NoRegExp', 'NoRegNoExp', 'Location', 'northwest');
saveas(gcf, fullfile(save_dir, 'exp9_data_res_full.png'));

plot(0:20, mean(o1.img_res, 1), 0:20, mean(o2.img_res, 1), 0:20, mean(o3.img_res, 1)) ;
xlabel('Layer'); 
ylabel('Mean image residual'); 
set(gca, 'FontName', 'Palatino', 'FontSize', 13); 
legend('RegExp', 'NoRegExp', 'NoRegNoExp', 'Location', 'northwest');
saveas(gcf, fullfile(save_dir, 'exp9_img_res_full.png')) ;


%% Experiment 
data_dir_1 = '/Volumes/MelSSD/runs/33_mix_triple_20l_reg/eval-vn-120000/' ;
data_dir_2 = '/Volumes/MelSSD/runs/39_mix_triple_moremix/eval-vn-120000/' ;
data_dir_3 = '/Volumes/MelSSD/runs/40_mix_triple_moremixmore/eval-vn-120000/' ;
%%
% Plot fullpipeline validation
filename = 'test-syn-0.0-0.0-test-fullpipeline_val_64_6comb_64_imgs.mat' ;
lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
m_1 = load(fullfile(data_dir_1, filename));
m_2 = load(fullfile(data_dir_2, filename));
m_3 = load(fullfile(data_dir_3, filename));
[Nimgs,nx, ny] = size(m_1.init_img) ;
[RMSEvn_1, RMSElbfgs] = RMSE(Nimgs, m_1, lbfgs_m.recon_lbfgs) ;
[RMSEvn_2, RMSElbfgs] = RMSE(Nimgs, m_2, lbfgs_m.recon_lbfgs) ;
[RMSEvn_3, RMSElbfgs] = RMSE(Nimgs, m_3, lbfgs_m.recon_lbfgs) ;
x = categorical({'1-1-14', '4-1-14', '6-2-14', 'LBFGS'}) ;
boxplot([RMSEvn_1(:), RMSEvn_2(:), RMSEvn_3(:), RMSElbfgs(:)], x, 'Colors', 'k', 'Symbol', 'k+', 'DataLim', [0, 300], 'Jitter', 0.1)
ylabel('RMSE'); %title({'RMSE on FULLPIPELINE validation set'});
set(gca, 'FontName', 'Palatino', 'FontSize', 13);
saveas(gcf, fullfile(save_dir, 'exp10_validation_full.png'));

% Plot fullpipeline test
filename = 'test-syn-0.0-0.0-test-fullpipeline_testset_6comb_32_imgs.mat' ;
lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
m_1 = load(fullfile(data_dir_1, filename));
m_2 = load(fullfile(data_dir_2, filename));
m_3 = load(fullfile(data_dir_3, filename));
[Nimgs,nx, ny] = size(m_1.init_img) ;
[RMSEvn_1, RMSElbfgs] = RMSE(Nimgs, m_1, lbfgs_m.recon_lbfgs) ;
[RMSEvn_2, RMSElbfgs] = RMSE(Nimgs, m_2, lbfgs_m.recon_lbfgs) ;
[RMSEvn_3, RMSElbfgs] = RMSE(Nimgs, m_3, lbfgs_m.recon_lbfgs) ;
x = categorical({'1-1-14', '4-1-14', '6-2-14', 'LBFGS'}) ;
boxplot([RMSEvn_1(:), RMSEvn_2(:), RMSEvn_3(:), RMSElbfgs(:)], x, 'Colors', 'k', 'Symbol', 'k+', 'DataLim', [0, 300], 'Jitter', 0.1)
ylabel('RMSE'); %title({'RMSE on FULLPIPELINE validation set'});
set(gca, 'FontName', 'Palatino', 'FontSize', 13);
saveas(gcf, fullfile(save_dir, 'exp10_test_full.png'));

% Plot validation set ray-based usm
set(gcf, 'Position', [440, 286, 423, 512]);
x = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9] ;
for i = 1:numel(x)
    u = x(i);
    filename = sprintf('test-syn-0.0-%.1f-patchy-val_MS_6comb.mat', u);
    lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
    m_1 = load(fullfile(data_dir_1, filename));
    m_2 = load(fullfile(data_dir_2, filename));
    m_3 = load(fullfile(data_dir_3, filename));
    [Nimgs,nx, ny] = size(m_1.init_img) ;
    [RMSEvn_1, RMSElbfgs] = RMSE(Nimgs, m_1, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_2, RMSElbfgs] = RMSE(Nimgs, m_2, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_3, RMSElbfgs] = RMSE(Nimgs, m_3, lbfgs_m.recon_lbfgs) ;
    rmse_vn_1(i) = median(RMSEvn_1(:));
    rmse_vn_2(i) = median(RMSEvn_2(:));
    rmse_vn_3(i) = median(RMSEvn_3(:));
    rmse_lbfgs(i)= median(RMSElbfgs(:));
end
clf;
plot(x, rmse_vn_1, '-o', x, rmse_vn_2, '-o', x, rmse_vn_3, '-o', x, rmse_lbfgs, 'k--o'); %title('RMSE on RAY-BASED validation set'); 
legend('1-1-14', '4-1-14', '6-2-14', 'LBFGS', 'Location', 'northwest'); xlabel('Undersampling rate'); ylabel('Median RMSE'); xticks(0:0.1:1); 
set(gca, 'FontName', 'Palatino', 'FontSize', 13);
saveas(gcf, fullfile(save_dir, 'exp10_validation_usr_ray.png'));

% Plot validation set ideal time usm
x = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9] ;
for i = 1:numel(x)
    u = x(i);
    filename = sprintf('test-ideal_time-0.0-%.1f-patchy-ideal_time_val.mat', u);
    lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
    m_1 = load(fullfile(data_dir_1, filename));
    m_2 = load(fullfile(data_dir_2, filename));
    m_3 = load(fullfile(data_dir_3, filename));
    [Nimgs,nx, ny] = size(m_1.init_img) ;
    [RMSEvn_1, RMSElbfgs] = RMSE(Nimgs, m_1, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_2, RMSElbfgs] = RMSE(Nimgs, m_2, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_3, RMSElbfgs] = RMSE(Nimgs, m_3, lbfgs_m.recon_lbfgs) ;
    rmse_vn_1(i) = median(RMSEvn_1(:));
    rmse_vn_2(i) = median(RMSEvn_2(:));
    rmse_vn_3(i) = median(RMSEvn_3(:));
    rmse_lbfgs(i)= median(RMSElbfgs(:));
end
clf;
plot(x, rmse_vn_1, '-o', x, rmse_vn_2, '-o', x, rmse_vn_3, '-o', x, rmse_lbfgs, 'k--o'); %title('RMSE on RAY-BASED validation set'); 
legend('1-1-14', '4-1-14', '6-2-14', 'LBFGS', 'Location', 'northwest'); xlabel('Undersampling rate'); ylabel('Median RMSE'); xticks(0:0.1:1); 
set(gca, 'FontName', 'Palatino', 'FontSize', 13);
saveas(gcf, fullfile(save_dir, 'exp10_validation_usr_ideal.png'));

% Plot test set ray based usm
x = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9] ;
for i = 1:numel(x)
    u = x(i);
    filename = sprintf('test-syn-0.0-%.1f-patchy-testset_ideal_MS_32_imgs.mat', u);
        lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
    m_1 = load(fullfile(data_dir_1, filename));
    m_2 = load(fullfile(data_dir_2, filename));
    m_3 = load(fullfile(data_dir_3, filename));
    [Nimgs,nx, ny] = size(m_1.init_img) ;
    [RMSEvn_1, RMSElbfgs] = RMSE(Nimgs, m_1, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_2, RMSElbfgs] = RMSE(Nimgs, m_2, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_3, RMSElbfgs] = RMSE(Nimgs, m_3, lbfgs_m.recon_lbfgs) ;
    rmse_vn_1(i) = median(RMSEvn_1(:));
    rmse_vn_2(i) = median(RMSEvn_2(:));
    rmse_vn_3(i) = median(RMSEvn_3(:));
    rmse_lbfgs(i)= median(RMSElbfgs(:));
end
clf;
plot(x, rmse_vn_1, '-o', x, rmse_vn_2, '-o', x, rmse_vn_3, '-o', x, rmse_lbfgs, 'k--o'); %title('RMSE on RAY-BASED validation set'); 
legend('1-1-14', '4-1-14', '6-2-14', 'LBFGS', 'Location', 'northwest'); xlabel('Undersampling rate'); ylabel('Median RMSE'); xticks(0:0.1:1); 
set(gca, 'FontName', 'Palatino', 'FontSize', 13);
saveas(gcf, fullfile(save_dir, 'exp10_test_usr_ray.png'));


% Plot testset ideal usr 
x = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9] ;
for i = 1:numel(x)
    u = x(i);
    filename = sprintf('test-ideal_time-0.0-%.1f-patchy-testset_ideal_MS_32_imgs.mat', u);
    lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
    m_1 = load(fullfile(data_dir_1, filename));
    m_2 = load(fullfile(data_dir_2, filename));
    m_3 = load(fullfile(data_dir_3, filename));
    [Nimgs,nx, ny] = size(m_1.init_img) ;
    [RMSEvn_1, RMSElbfgs] = RMSE(Nimgs, m_1, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_2, RMSElbfgs] = RMSE(Nimgs, m_2, lbfgs_m.recon_lbfgs) ;
    [RMSEvn_3, RMSElbfgs] = RMSE(Nimgs, m_3, lbfgs_m.recon_lbfgs) ;
    rmse_vn_1(i) = median(RMSEvn_1(:));
    rmse_vn_2(i) = median(RMSEvn_2(:));
    rmse_vn_3(i) = median(RMSEvn_3(:));
    rmse_lbfgs(i)= median(RMSElbfgs(:));
end
clf;
plot(x, rmse_vn_1, '-o', x, rmse_vn_2, '-o', x, rmse_vn_3, '-o', x, rmse_lbfgs, 'k--o'); %title('RMSE on RAY-BASED validation set'); 
legend('1-1-14', '4-1-14', '6-2-14', 'LBFGS', 'Location', 'northwest'); xlabel('Undersampling rate'); ylabel('Median RMSE'); xticks(0:0.1:1); 
set(gca, 'FontName', 'Palatino', 'FontSize', 13);
saveas(gcf, fullfile(save_dir, 'exp10_test_usr_ideal.png'));




%% Non-reg versus reg network
data_1 = '/Volumes/MelSSD/runs/nexp/eval-vn-120000/';
data_2 =  '/Volumes/MelSSD/runs/13_20l_reg_0_1e5/eval-vn-120000/' ;
filename = 'test-syn-0.0-0.0-test-fullpipeline_val_64_6comb_64_imgs.mat' ; m_1 = load(fullfile(data_1, filename));
 m_2 = load(fullfile(data_2, filename));
lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
[RMSEvn_1, RMSElbfgs] = RMSE(64, m_1, lbfgs_m.recon_lbfgs) ;
[RMSEvn_2, RMSElbfgs] = RMSE(64, m_2, lbfgs_m.recon_lbfgs) ;
median(RMSEvn_1(:))
median(RMSEvn_2(:))
filename = 'test-syn-0.0-0.0-test-fullpipeline_testset_6comb_32_imgs.mat' ; m_1 = load(fullfile(data_1, filename));
 m_2 = load(fullfile(data_2, filename));
lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
[RMSEvn_1, RMSElbfgs] = RMSE(32, m_1, lbfgs_m.recon_lbfgs) ;
[RMSEvn_2, RMSElbfgs] = RMSE(32, m_2, lbfgs_m.recon_lbfgs) ;
median(RMSEvn_1(:))
median(RMSEvn_2(:))
%% Best against worse
%% Non-reg versus reg network
data_1 = '/Volumes/MelSSD/runs/nexp/eval-vn-120000/';
data_2 =  '/Volumes/MelSSD/runs/39_mix_triple_moremix/eval-vn-120000/' ;
filename = 'test-syn-0.0-0.0-test-fullpipeline_val_64_6comb_64_imgs.mat' ; m_1 = load(fullfile(data_1, filename));
 m_2 = load(fullfile(data_2, filename));
lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
[RMSEvn_1, RMSElbfgs] = RMSE(64, m_1, lbfgs_m.recon_lbfgs) ;
[RMSEvn_2, RMSElbfgs] = RMSE(64, m_2, lbfgs_m.recon_lbfgs) ;
median(RMSEvn_1(:))
median(RMSEvn_2(:))
median(RMSElbfgs(:))
'mean'
mean(RMSEvn_1(:))
mean(RMSEvn_2(:))
mean(RMSElbfgs(:))
filename = 'test-syn-0.0-0.0-test-fullpipeline_testset_6comb_32_imgs.mat' ; m_1 = load(fullfile(data_1, filename));
 m_2 = load(fullfile(data_2, filename));
lbfgs_m = load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', filename)));
[RMSEvn_1, RMSElbfgs] = RMSE(32, m_1, lbfgs_m.recon_lbfgs) ;
[RMSEvn_2, RMSElbfgs] = RMSE(32, m_2, lbfgs_m.recon_lbfgs) ;
median(RMSEvn_1(:))
median(RMSEvn_2(:))
median(RMSElbfgs(:))
'mean'
mean(RMSEvn_1(:))
mean(RMSEvn_2(:))
mean(RMSElbfgs(:))
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
