%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Mélanie Bernhardt - M.Sc. Thesis - ETH Zurich
%% This file is used to investigate the behavior of the network in details
%% such as unrolled reconstruction plots, activations plots. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clear all
% addpath(genpath(['/Volumes/MelanieDisk/runs/3_F_noS_noU_noT_noLr/mat']));
%data_dir = '/Volumes/MelSSD/runs/13_20l_reg_0_1e5/eval-vn-120000';
data_dir = '/scratch_net/biwidl307/sonia/USImageReconstruction-Melanie/runs/32_mix_ideal_20l_reg/eval-vn-120000/';
save_dir = fullfile(data_dir, 'mat_viz/test');
m = load(fullfile(data_dir, 'test-syn-0.0-0.0-test-fullpipeline_testset_6comb_32_imgs.mat')) ;
%m = load(fullfile(data_dir, 'test-syn-0.0-test-phantom_trial1_14_imgs.mat'));
[nb,nx, ny] = size(m.init_img);
nlayer = 20;
%% Make dirs
mkdir(save_dir);

%% UNROLLING PLOTS
set(gcf, 'Position',  [1, 1, 1800, 500]); 
for image=1:5
    clf;
    set(gcf, 'Position',  [ -217, 1099, 1894, 786]); 
    layer_img = reshape(m.layer_sos(:,image,:,:), [nlayer+1, nx, ny]);
    subplot(3, 11, 1);  % layer 0
    imagesc(squeeze(layer_img(1, :,:))'); cb = colorbar('southoutside');  title('Initialization'); 
    set(cb, 'XTick', [1460, 1550]) ; xticks([]); yticks([]); axis equal tight;
    set(gca, 'FontName', 'Palatino', 'FontSize', 12) ; caxis([1460, 1550]);
    data_grad1 = reshape(m.data_grad(:,image,:,:), [nlayer, nx, ny]);
    reg_grad1 = reshape(m.reg_grad(:,image,:,:), [nlayer, nx, ny]);
    layer_img = reshape(m.layer_sos(:,image,:,:), [nlayer+1, nx, ny]);
    for layer=1:10
        subplot(3, 11, layer + 1) ;
        imagesc(squeeze(layer_img(layer+1, :,:))'); cb = colorbar('southoutside');
        set(cb, 'XTick', [1460, 1550]) ;
        set(gca, 'FontName', 'Palatino', 'FontSize', 13) ; caxis([1460, 1550]);
        title({'Reconstruction', sprintf('layer %d', layer)}); 
        xticks([]); yticks([]); axis equal tight;
        subplot(3, 11, 11 + layer + 1) ;
        imagesc(squeeze(data_grad1(layer, :,:))'); cb = colorbar('southoutside'); xticks([]); yticks([]); axis equal tight;
        set(gca, 'FontName', 'Palatino', 'FontSize', 13) ;
        title({'Data Gradient', sprintf('layer %d', layer)});        
        subplot(3, 11, 22 + layer + 1) ;
        imagesc(squeeze(reg_grad1(layer, :,:))'); cb = colorbar('southoutside'); xticks([]); yticks([]); axis equal tight;
        set(gca, 'FontName', 'Palatino', 'FontSize', 13) ;
        title({'Regularizer', 'gradient', sprintf('layer %d', layer)});
    end
    saveas(gcf, fullfile(save_dir, sprintf('image_%d_p1.png', image)));
    clf;
    set(gcf, 'Position',  [ -217, 1099, 1894, 786]);  
    for layer=1:10
        subplot(3, 11, layer) ;
        imagesc(squeeze(layer_img(layer+1+10, :,:))'); cb = colorbar('southoutside');
        set(cb, 'XTick', [1460, 1550]) ;
        set(gca, 'FontName', 'Palatino', 'FontSize', 13) ; caxis([1460, 1550]);
        title({'Reconstruction', sprintf('layer %d', layer+10)}); 
        xticks([]); yticks([]); axis equal tight;
        subplot(3, 11, 11 + layer) ;
        imagesc(squeeze(data_grad1(layer+10, :,:))'); cb = colorbar('southoutside'); xticks([]); yticks([]); axis equal tight;
        set(gca, 'FontName', 'Palatino', 'FontSize', 13) ;
        title({'Data Gradient', sprintf('layer %d', layer+10)});        
        subplot(3, 11, 22 + layer) ;
        imagesc(squeeze(reg_grad1(layer+10, :,:))'); cb = colorbar('southoutside'); xticks([]); yticks([]); axis equal tight;
        set(gca, 'FontName', 'Palatino', 'FontSize', 13) ;
        title({'Regularizer', 'gradient', sprintf('layer %d', layer+10)});
    end
    try
        try
            1./m.gt_slowness(image,:,:);
            subplot(3, 11, 11);  % Ground SoS
            imagesc(reshape(1./m.gt_slowness(image,:,:), [nx,ny])'); cb = colorbar('southoutside'); title('Ground truth SoS'); 
            set(cb, 'XTick', [1460, 1550]) ;
            set(gca, 'FontName', 'Palatino', 'FontSize', 13) ; caxis([1460, 1550]); title('Ground truth SoS'); set(gca, 'FontName', 'Palatino', 'FontSize', 13) ;
        xticks([]); yticks([]); axis equal tight;
        catch
            m.gt_sos(image,:,:);
            subplot(3, 11, 11);  % Ground SoS
            imagesc(reshape(m.gt_sos(image,:,:), [nx,ny])'); set(cb, 'XTick', [1460, 1550]) ;
            set(gca, 'FontName', 'Palatino', 'FontSize', 13) ; caxis([1460, 1550]); title('Ground truth SoS'); 
        xticks([]); yticks([]); axis equal tight;
        end     
    end
    saveas(gcf, fullfile(save_dir, sprintf('image_%d_p2.png', image)));
end
%% DATA TERM ACTIVATION
clf;
set(gcf, 'Position',  [100, 100, 1500, 1500]);
for p=1:20
    subplot(4,5,p);
    tmp = m.before_act_data(p,:,:);
    histogram(tmp(tmp~=0), 300, 'Normalization','probability') ;
    title({sprintf('Layer %d', p), ''}); set(gca, 'FontName', 'Palatino', 'FontSize', 13) ;
end
saveas(gcf, fullfile(save_dir, 'data_term.png'));

% %% REG TERM ACTIVATION 
% clf; %20    16    57    77    64
% set(gcf, 'Position',  [100, 100, 1500, 1500]);
% for layer=1:20
%     for p=1:32
%         subplot(6, 6, p);
%         tmp = m.before_act_reg(layer, :, :, :, p);
%         histogram(tmp(tmp~=0), 100);
%         title({sprintf('Filter %d', p), ''}); set(gca, 'FontName', 'Palatino', 'FontSize', 13) ;
%     end
%     saveas(gcf, fullfile(save_dir, 'activation_ranges', sprintf('reg_term_layer_%d.png', layer)));
% end