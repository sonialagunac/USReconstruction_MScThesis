%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Mélanie Bernhardt - M.Sc. Thesis - ETH Zurich
%% This file is used to investigate the parameters learned 
%% by the network in details: conv filters, spatial weights, 
%% activations function visualization.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Directories adapted to Sonia Laguna, MSc Thesis 2022
%% LOADING
clear all
data_dir = '/scratch_net/biwidl307/sonia/USImageReconstruction-Sonia/runs/32_mix_ideal_20l_reg_mix/'
save_dir = '/scratch_net/biwidl307/sonia/USImageReconstruction-Sonia/runs/32_mix_ideal_20l_reg_mix/eval-vn-120000/model_params'
m = load(fullfile(data_dir, '/mat/dm-train-120000.mat')) ;
nlayer = 20;
shared=true;
%% Create the dirs
mkdir(fullfile(save_dir, 'activations'));
mkdir(fullfile(save_dir, 'conv_filters'));
mkdir(fullfile(save_dir, 'spatial_weights'));
%% DATA TERM ACTIVATION
clf;
[z, k] = size(m.interp_knots) ;
if shared == true
    maxl = m.interp_max(1) ;
    plot(linspace(-maxl, maxl, k), m.interp_knots(1,:));
    set(gca, 'FontName', 'Palatino', 'FontSize', 12) ;
else
    for layer = 1:nlayer
        maxl = m.interp_max(layer) ;
        subplot(4,5,layer);
        plot(linspace(-maxl, maxl, k), m.interp_knots(layer,:));
        title(sprintf('Layer %d', layer));
        set(gca, 'FontName', 'Palatino', 'FontSize', 12) ;
    end
end
saveas(gcf, fullfile(save_dir, 'activations', 'data_term.png'));
%% ACTIVATION REGULARIZATION
clf;
[z, k, zz] = size(m.reg_act_params) ; 
if shared == true
    nlayer=1;
end
for layer = 1:nlayer
    set(gcf, 'Position',  [100, 100, 800, 800]);
    h = sgtitle(sprintf('Layer %d', layer));
    set(h, 'FontName', 'Palatino', 'FontSize', 15) ;
    maxl = m.interp_var_reg(layer) ;
    for filter = 1:32
        subplot(7,5,filter);
        plot(linspace(-maxl, maxl, k), m.reg_act_params(layer, :, filter));
        title(sprintf('Filter %d', filter)); set(gca, 'FontName', 'Palatino', 'FontSize', 12) ;
        ylim([-5,5]);
    end
    saveas(gcf, fullfile(save_dir, 'activations', sprintf('reg_term_layer_%d.png', layer)));
end
%% CONV FILTERS
clf;
 set(gcf, 'Position',  [214        1207         552         615]);
for layer = 1:nlayer
    h = sgtitle(sprintf('Filters of layer %d', layer));
    set(h, 'FontName', 'Palatino', 'FontSize', 14) ;
    for i = 1:32
        if i == 1
           subplot(7,5,i);
           imagesc(squeeze(m.conv_filt(layer,:,:,1,i))); set(gca,'XTick',[], 'YTick', []); axis equal tight ; caxis([-0.5, 0.5]);
        else
           subplot(7,5,i)
          imagesc(squeeze(m.conv_filt(layer,:,:,1,i))); set(gca,'XTick',[], 'YTick', []); axis equal tight ; caxis([-0.5, 0.5]);
        end
        set(gca, 'FontName', 'Palatino', 'FontSize', 12) ;
    end
    hp4 = get(subplot(7,5,1),'Position') ;
    colorbar('Position', [hp4(1)-hp4(3)/3 hp4(2) hp4(3)/5 hp4(4)]);
    caxis([-0.5, 0.5]); axis tight; set(gca, 'FontName', 'Palatino', 'FontSize', 12) ;
    saveas(gcf, fullfile(save_dir, 'conv_filters', sprintf('layer_%d.png', layer))); 
end
%% MODEL PARAMETERS PLOTTING
clf;
for layer = 1:nlayer
    set(gcf, 'Position',  [327        1080         598         598]);
    h = sgtitle(sprintf('Spatial weights of layer %d', layer));
    set(h, 'FontName', 'Palatino', 'FontSize', 14) ;
    for i = 1:32
          subplot(5,7,i)
          imagesc(squeeze(m.cWeight_us_sgm(layer,1, :,:,i))'); set(gca,'XTick',[], 'YTick', []); caxis([0,1]); axis equal tight
    end
hp4 = get(subplot(5,7,1),'Position') ;
colorbar('Position', [hp4(1)-hp4(3)/3 hp4(2) hp4(3)/5 hp4(4)]);
caxis([0,1]); axis tight; set(gca, 'FontName', 'Palatino', 'FontSize', 12) ;
saveas(gcf, fullfile(save_dir, 'spatial_weights', sprintf('layer_%d.png', layer)));
end
%% FUNCTION DEFINITIONS
function plot_summary(m, i,l,nx,ny)
clf;
for n = 0:l
    if n == 0
        figure();
        set(gcf, 'Position',  [100, 100, 500, 100]);
        layer_img = reshape(m.layer_sos(:,i,:,:), [21, nx, ny]);
        subplot(1, 3, 1);  % layer 0
        imagesc(squeeze(layer_img(1, :,:))'); colorbar; title('Init reconstruction');
    else
        figure();
        set(gcf, 'Position',  [100, 100, 500, 100]);
        plot_one_layer(i, n, m, nx, ny, l)
        if n == l
           figure();
           set(gcf, 'Position',  [100, 100, 500, 100]);
           subplot(1, 3, 1);  % Ground SoS
           try
                imagesc(reshape(1./m.xgt_slowness(i,:,:), [nx,ny])'); colorbar; title('Ground truth SoS');
           catch
                imagesc(reshape(m.gt_sos(i,:,:), [nx,ny])'); colorbar; title('Ground truth SoS');
          end
        end
    end
end
end


function plot_one_layer(i, n, m, nx, ny, l)
    data_grad1 = reshape(m.data_grad(:,i,:,:), [l, nx, ny]);
    reg_grad1 = reshape(m.reg_grad(:,i,:,:), [l, nx, ny]);
    layer_img = reshape(m.layer_sos(:,i,:,:), [l+1, nx, ny]);
    subplot(1, 3, 1) ;
    imagesc(squeeze(layer_img(n+1, :,:))'); colorbar;
    title(sprintf('Reconstruction layer %d', n));
    subplot(1, 3, 2) ;
    imagesc(squeeze(data_grad1(n, :,:))'); colorbar;
    title(sprintf('Data grad layer %d', n));        
    subplot(1, 3, 3) ;
    imagesc(squeeze(reg_grad1(n, :,:))'); colorbar;
    title(sprintf('Regularization grad layer %d', n));
end

function regCost = computeRegCost(i, m, D, lambda)
        for l=1:21
            xi = 1./squeeze(m.layer_sos(l,i,:,:)) ;
            xi = xi';	
            Dxi = D*xi(:);
            regCost(l) = lambda * mean(abs(Dxi(:)));
        end
end

function DataRMSE = computeDataCost(i, L, m)
    dtrue = m.din(:,i);  
    for l=1:21
    	xi = squeeze(1 ./ m.layer_sos(l, i, :, :)) ;
        xi = xi';
    	Lxhat = L*xi(:);
    	DataRMSE(l) = sqrt(mean(power(Lxhat(dtrue~=0) - dtrue(dtrue~=0), 2)));
    end
end


function ForwardDataRMSE = computeForwardDataCost(i, L, m)
    try
        xtrue = 1./m.gt_sos(i,:,:);
    catch 
        xtrue = m.xgt_slowness(i,:,:);
    end
    xtrue = squeeze(xtrue)';
    dtrue = L*xtrue(:);
    for l=1:21
    	xi = 1./squeeze(m.layer_sos(l,i,:,:)) ;
        xi = xi';
    	Lxhat = L*xi(:);
    	ForwardDataRMSE(l) = sqrt(mean((Lxhat(:) - dtrue(:)) .^ 2));
    end
end

function targetRMSE = computeTargetRMSE(i, m)
    try
        xtrue = 1./m.gt_sos(i,:,:);
    catch 
        xtrue = m.xgt_slowness(i,:,:);
    end
    for l=1:21
        xi = squeeze(1./m.layer_sos(l, i, :, :));	
        targetRMSE(l) = sqrt(mean(power(xi(:) - xtrue(:), 2)));
    end
end