%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Melanie Bernhardt - M.Sc. Thesis - ETH Zurich
%% This file is used to compute the reconstruction on phantom datasets
%% VN and LBFGS. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Directories adapted to Sonia Laguna, MSc Thesis 2022
% load files
clear
exp_dir = '/scratch_net/biwidl307/sonia/USImageReconstruction-Sonia/runs/4_ICFP_reg1e5_tau5/eval-vn-120000/'
Nimgs = 14;

load(fullfile(exp_dir, 'test-syn-0.0-test-phantom_trial1_14_imgs.mat'));
plot_dir = fullfile(exp_dir, 'phantom');
mkdir(plot_dir);
SAVE_BF = false ; 
if SAVE_BF == true
      load('/scratch_net/biwidl307/sonia/data_original/phantom/phantom_trial1_14_imgs.mat')
end
%plot

%%
load('/scratch_net/biwidl307/sonia/data_original/phantom/phantom_trial1_14_imgs.mat')

SAVE_BF = true;
Nimgs = 14; % 14

pitch = 3e-4;
pixelsize_bf =  [1/8, 1/2].*pitch; % ax/lat resolution for the bf/dispTrack
pixelsize_sos =  [2, 2].*pitch; % ax/lat resolution for the bf/dispTrack
Depth = 50e-3 ;%opts.acq.RX.depth; % imaging depth (For kwave)
Width = (128-1)/2*pitch; % imaging width
xax_sos = [-Width : pixelsize_sos(2) : Width];  
xax_sos = xax_sos-mean(xax_sos);
zax_sos = [0e-3:pixelsize_sos(1):Depth];
xax_bf = [xax_sos(1):pixelsize_bf(2):xax_sos(end)];  
xax_bf = xax_bf-mean(xax_bf);
zax_bf = [zax_sos(1):pixelsize_bf(1):zax_sos(end)];

%% Save BF image
tmp = '/scratch_net/biwidl307/sonia/USImageReconstruction-Sonia/figs_phantom/'
clf;
set(gcf, 'Position', [440, 372, 363*3, 426]);

%%
for i=1:Nimgs
    if SAVE_BF==true
        %i
        bf = bf_im(:,:,:, i);
        im = abs(hilbert(mean(bf,3)));
        %size(im)
        im = im./max(abs(im(:)));
        im = 20*log10(im);
        subplot(131)
        imagesc(xax_bf*1e3, zax_bf*1e3, im)
        xticks([]); yticks([]); 
        axis equal tight
        cb = colorbar;    
        title(cb, '[dB]'); colormap(gray(256));
        caxis([-60 0]);
        set(gca, 'FontName', 'Palatino', 'FontSize', 14) ;
        saveas(gcf, fullfile(tmp, sprintf('bmode_phantom_%d.png', i)));
        r = recon_lbfgs(:,:,i);
        subplot(132)
        imagesc(xax_sos*1e3, zax_sos*1e3, squeeze(r));
        axis equal tight; 
        cb = colorbar; title(cb, '[m/s]'); colormap(jet); caxis([1460, 1540]); xticks([]); yticks([]);
        set(gca, 'FontName', 'Palatino', 'FontSize', 14) ;
        %saveas(gcf, fullfile(tmp, sprintf('lbfgs_phantom_%d.png', i)));
    end
    subplot(133)
    r = recon(i,:,:)
    imagesc(xax_sos*1e3, zax_sos*1e3, squeeze(r)'); colorbar; caxis([1460, 1540]);
    axis equal tight; 
    cb = colorbar; 
    title(cb, '[m/s]'); colormap(jet); xticks([]); yticks([]);
    set(gca, 'FontName', 'Palatino', 'FontSize', 14);
end

 %% Good phantom
load(fullfile(exp_dir, 'test-syn-0.0-test-fukuda_1_imgs.mat'))
if SAVE_BF == true
    load('/scratch_net/biwidl307/sonia/data_original/phantom/fukuda_1_imgs.mat')
end

clf;
set(gcf, 'Position', [440, 372, 363, 426]);
i=1;
if SAVE_BF==true
    bf = bf_im(:,:,:, i);
    im = abs(hilbert(mean(bf,3)));
    size(im)
    im = im./max(abs(im(:)));
    im = 20*log10(im);
    imagesc(xax_bf*1e3, zax_bf*1e3, im)
    axis equal tight
    cb = colorbar;    
    title(cb,'[dB]'); colormap(gray(256)); xticks([]); yticks([]);
    caxis([-60 0]);
    set(gca, 'FontName', 'Palatino', 'FontSize', 14) ;
    saveas(gcf, fullfile(tmp, sprintf('bmode_good_phantom_%d.png', i)));
    r = recon_lbfgs(:,:,i);
    imagesc(xax_sos*1e3, zax_sos*1e3, squeeze(r)); xticks([]); yticks([]);
    axis equal tight; 
    cb = colorbar; title(cb, '[m/s]'); colormap(jet); caxis([1460, 1540]);
    set(gca, 'FontName', 'Palatino', 'FontSize', 14) ;
    saveas(gcf, fullfile(tmp, 'lbfgs_good_phantom.png'));
end
r = recon(i,:,:);
imagesc(xax_sos*1e3, zax_sos*1e3, squeeze(r)'); cb = colorbar; title(cb, '[m/s]'); caxis([1460, 1540]);
axis equal tight; colormap(jet); xticks([]); yticks([]);
set(gca, 'FontName', 'Palatino', 'FontSize', 14);
saveas(gcf, fullfile(plot_dir, 'good_phantom.png'));
