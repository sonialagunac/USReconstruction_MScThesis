%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Mélanie Bernhardt - M.Sc. Thesis - ETH Zurich
%% This file is used to plot and save the individual plots of 
%% of the test set reconstruction.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clear 
data_dir = '/Volumes/MelSSD/runs/39_mix_triple_moremix/eval-vn-120000/';
lbfgs_dir = '/Volumes/MelSSD/runs/lbfgs';
METHOD = 'MS';
cd '/Users/melaniebernhardt/Desktop/MScThesis/SimulationModels' ;
addpath('~/Desktop/MScThesis/SimulationModels');
addpath(genpath('/Users/melaniebernhardt/Desktop/MScThesis/Generic_Data_Processing_Structure-OGtests/'));
addpath(genpath('~/Desktop/MScThesis/USImageReconstruction/code/'));
save_dir = 'plots_testset';
filename = 'testset_ideal_MS_32_imgs.mat';
c = fullfile(data_dir, save_dir);
chan_comb = [15, 25 ; 
            35, 45 ; 
            55, 65;
            75, 85;
            95, 105;
            105, 115];
NA = 6;
mkdir(c)
Nimgs=32;
save_lbfgs = false;
data_type = 'syn';
data_name = 'RAY-BASED';
name = sprintf('test-syn-0.0-0.7-patchy-%s', filename);
set(gcf, 'Position', [440, 372, 363, 426]);
val = load(fullfile(data_dir, name));
[Nimgs, ~, ~] = size(val.gt_slowness);
if save_lbfgs == true
load(fullfile(lbfgs_dir, sprintf('lbfgs-%s', name)));
for i=1:Nimgs
    xg = reshape(1./val.gt_slowness(i,:,:), [64,84]);
    rec2 = reshape(recon_lbfgs(i,:,:), [84,64]);
    imagesc(rec2); cb=colorbar; title(cb, '[m/s]'); xticks([]); yticks([]); 
    if i == 10 || i==11 || i==12 || i==13 || i==14 || i==15
        caxis([1400, 1520]);
    else
        caxis([min(xg(:))-20, max(xg(:))+10]);
    end
    set(gca, 'FontName', 'Palatino', 'FontSize', 13); colormap(jet);
    saveas(gcf, fullfile(lbfgs_dir, sprintf('img_%d.png', i)));
end
end
    
 for i=1:Nimgs
    xg = reshape(1./val.gt_slowness(i,:,:), [64,84]);
    imagesc(xg'); cb=colorbar; title(cb, '[m/s]'); xticks([]); yticks([]); 
    set(gca, 'FontName', 'Palatino', 'FontSize', 13);
    if i == 10 || i==11 || i==12 || i==13 || i==14 || i==15
        caxis([1400, 1520]);
    else
        caxis([min(xg(:))-20, max(xg(:))+10]);
    end
    saveas(gcf, fullfile(lbfgs_dir, sprintf('gt_img_%d.png', i)));
    rec1 = reshape(val.recon(i,:,:), [64,84]);
    imagesc(rec1'); cb=colorbar; title(cb, '[m/s]'); xticks([]); yticks([]); 
    colormap(jet);
    if i == 10 || i==11 || i==12 || i==13 || i==14 || i==15
        caxis([1400, 1520]);
    else
        caxis([min(xg(:))-20, max(xg(:))+10]);
    end
    set(gca, 'FontName', 'Palatino', 'FontSize', 13);
    saveas(gcf, fullfile(data_dir, save_dir, sprintf('ray_img_%d.png', i)));
 end
 
full = load(fullfile(data_dir, 'test-syn-0.0-0.0-test-fullpipeline_testset_6comb_32_imgs.mat'));
 if save_lbfgs == true
load(fullfile(lbfgs_dir, 'lbfgs-test-syn-0.0-0.0-test-fullpipeline_testset_6comb_32_imgs.mat'));
for i=1:Nimgs
    xg = reshape(1./val.gt_slowness(i,:,:), [64,84]);
    rec2 = reshape(recon_lbfgs(i,:,:), [84,64]);
    imagesc(rec2); cb=colorbar; title(cb, '[m/s]'); xticks([]); yticks([]);
    if i == 10 || i==11 || i==12 || i==13 || i==14 || i==15
        caxis([1400, 1520]);
    else
        caxis([min(xg(:))-20, max(xg(:))+10]);
    end
    set(gca, 'FontName', 'Palatino', 'FontSize', 13);
    saveas(gcf, fullfile(lbfgs_dir, sprintf('full_img_%d.png', i)));
end

end
  for i=1:Nimgs
    xg = reshape(1./full.gt_slowness(i,:,:), [64,84]);
    rec1 = reshape(full.recon(i,:,:), [64,84]);
    imagesc(rec1'); cb=colorbar; title(cb, '[m/s]'); xticks([]); yticks([]); 
    if i == 10 || i==11 || i==12 || i==13 || i==14 || i==15
        caxis([1400, 1520]);
    else
        caxis([min(xg(:))-20, max(xg(:))+10]);
    end
    set(gca, 'FontName', 'Palatino', 'FontSize', 13);
    saveas(gcf, fullfile(data_dir, save_dir, sprintf('full_img_%d.png', i)));
  end
 
  
  
  
%%
set(gcf, 'Position', [440, 372, 363, 426]);
colormap(jet)
full = load(fullfile(data_dir, 'test-syn-0.0-0.0-test-fullpipeline_val_64_6comb_64_imgs.mat'));
save_dir = 'plots_valset';
c = fullfile(data_dir, save_dir);
mkdir(c);
save_lbfgs = false;
if save_lbfgs == true
    load(fullfile(lbfgs_dir, 'lbfgs-test-syn-0.0-0.0-test-fullpipeline_val_64_6comb_64_imgs.mat'));
for i=1:Nimgs
    xg = reshape(1./full.gt_slowness(i,:,:), [64,84]);
    rec2 = reshape(recon_lbfgs(i,:,:), [84,64]);
    imagesc(rec2); cb=colorbar; title(cb, '[m/s]'); xticks([]); yticks([]);
    caxis([min(xg(:))-10, max(xg(:))+10]);
    set(gca, 'FontName', 'Palatino', 'FontSize', 13);
    saveas(gcf, fullfile(lbfgs_dir, sprintf('val_full_img_%d.png', i)));
    imagesc(xg'); cb=colorbar; title(cb, '[m/s]'); xticks([]); yticks([]);
    caxis([min(xg(:))-10, max(xg(:))+10]);
    set(gca, 'FontName', 'Palatino', 'FontSize', 13);
    saveas(gcf, fullfile(lbfgs_dir, sprintf('val_gt_img_%d.png', i)));
end

end
  for i=1:Nimgs
    xg = reshape(1./full.gt_slowness(i,:,:), [64,84]);
    rec1 = reshape(full.recon(i,:,:), [64,84]);
    imagesc(rec1'); cb=colorbar; title(cb, '[m/s]'); xticks([]); yticks([]); 
    caxis([min(xg(:))-10, max(xg(:))+10]);
    set(gca, 'FontName', 'Palatino', 'FontSize', 13);
    saveas(gcf, fullfile(data_dir, save_dir, sprintf('full_img_%d.png', i)));
  end
 
         