%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Melanie Bernhardt - M.Sc. Thesis - ETH Zurich
%% This file is used to compute the reconstruction on phantom datasets
%% VN and LBFGS. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% load files
clear
%exp_dir = '/Volumes/MelSSD/runs/39_mix_triple_moremix/eval-vn-120000/';
exp_dir = '/scratch_net/biwidl307/sonia/USImageReconstruction-Melanie/runs/32_mix_ideal_20l_reg/eval-vn-120000/'
Nimgs = 14;

load(fullfile(exp_dir, 'test-syn-0.0-test-phantom_trial1_14_imgs.mat'));
plot_dir = fullfile(exp_dir, 'phantom');
mkdir(plot_dir);
SAVE_BF = false ; 
if SAVE_BF == true
    %load('/Users/melaniebernhardt/Desktop/MScThesis/USImageReconstruction/data/phantom_trial1_14_imgs.mat');
    load('/scratch_net/biwidl307/sonia/data_original/phantom/phantom_trial1_14_imgs.mat')
end
%plot

%%
%load('/Users/lin/Movies/sos_recon/phantom_trial1_14_imgs.mat')
load('/scratch_net/biwidl307/sonia/data_original/phantom/phantom_trial1_14_imgs.mat')
%load('/Users/lin/Movies/sos_recon/new_breast_processed_12_imgs.mat')
%d_sos = load('/Users/lin/Movies/test_phantom_result.mat');
%d_slow = load('/Users/lin/Movies/test_phantom_result_slowness.mat');
%d_sos = load('/Users/lin/Movies/test_phantom_result.mat');
%d_slow = load('/Users/lin/Movies/test_phantom_result_slowness.mat');

%load('/Users/lin/Movies/sos_recon/fukuda_1_imgs.mat')
%d_sos = load('/Users/lin/Movies/test_best_phantom_result.mat');
%d_slow = load('/Users/lin/Movies/test_best_phantom_result_slowness.mat');

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
%tmp = '/Users/lin/Movies/figs_phantom/';
tmp = '/scratch_net/biwidl307/sonia/USImageReconstruction-Melanie/figs_phantom/'
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
        saveas(gcf, fullfile(tmp, sprintf('lbfgs_phantom_%d.png', i)));
    end
    subplot(133)
    r = recon(i,:,:)
    imagesc(xax_sos*1e3, zax_sos*1e3, squeeze(r)'); colorbar; caxis([1460, 1540]);
    %r = d_slow.rec(i,:,:);
    %imagesc(xax_sos*1e3, zax_sos*1e3, squeeze(1./r)'); colorbar; caxis([1460, 1540]);
    axis equal tight; 
    cb = colorbar; 
    title(cb, '[m/s]'); colormap(jet); xticks([]); yticks([]);
    set(gca, 'FontName', 'Palatino', 'FontSize', 14);
    %subplot(144)
    %r = d_sos.rec(i,:,:);
    %imagesc(xax_sos*1e3, zax_sos*1e3, squeeze(1./r)'); colorbar; caxis([1460, 1540]);
    %axis equal tight; 
    %cb = colorbar; 
    %title(cb, '[m/s]'); colormap(jet); xticks([]); yticks([]);
    %set(gca, 'FontName', 'Palatino', 'FontSize', 14);
    %pause(1)
    saveas(gcf, fullfile(tmp, sprintf('phantom_%d.png', i)));
    %saveas(gcf, fullfile(plot_dir, sprintf('phantom_%d.png', i)));
end


%%
% %%
% for i=1:Nimgs
%     clf;
%     din2 = reshape(din, [84, 64, 6, Nimgs]);
%     subplot(3,3,1);
%     imagesc(din2(:,:,1 ,i)); colorbar; axis equal tight; colormap(parula);
%     subplot(3,3,2);
%     imagesc(din2(:,:,2,i));colorbar; axis equal tight;
%     subplot(3,3,3);
%     imagesc(din2(:,:,3,i));colorbar; axis equal tight;
%     subplot(3,3,4);
%     imagesc(din2(:,:,4,i));colorbar; axis equal tight;
%     subplot(3,3,5);
%     imagesc(din2(:,:,5,i));colorbar; axis equal tight;
%     subplot(3,3,6);
%     imagesc(din2(:,:,6,i));colorbar; axis equal tight;
% 
%     % BF
%     ax7 = subplot(3,3,7);
%     bf = bf_im(:,:,:, i);
%     im = abs(hilbert(mean(bf,3)));
%     size(im)
%     im = im./max(abs(im(:)));
%     im = 20*log10(im);
%     imagesc(xax_bf*1e3, zax_bf*1e3, im)
%     axis equal tight
%     cb = colorbar;    
%     cb.Label.String = '[dB]'; colormap(ax7, gray(256));
%     caxis([-60 0]);
%     title('B-mode image') ;
%     % VN recon
%     ax8 = subplot(3,3,8);
%     r = recon(i,:,:);
%     imagesc(xax_sos*1e3, zax_sos*1e3, squeeze(r)'); colorbar; %caxis([1490, 1530]);
%     axis equal tight; colormap(ax8, hot);
%     title('VN reconstruction') ;
% 
%     % LBFGS recon
%     ax9 = subplot(3,3,9);
%     r = recon_lbfgs(:,:,i);
%     imagesc(xax_sos*1e3, zax_sos*1e3, squeeze(r)); colorbar; %caxis([1490, 1530]);
%     axis equal tight; colormap(ax9, hot);
%     title('LBFGS reconstruction') ;
% 
%     tmp = sprintf('phantom_vis_%d.png', i);
%     saveas(gcf, fullfile(exp_dir, tmp));
% end
% 
% 
% %% Page 1
% p=0
% set(gcf, 'Position',  [1, 1, 900, 1200]); 
% img_per_page = min(Nimgs, 5);
% page_nb = 1;
% %Nimgs=13;
% for i=1:Nimgs
%     ax7 = subplot(img_per_page, 3, p+1);
%     % BF
%     bf = bf_im(:,:,:, i);
%     im = abs(hilbert(mean(bf,3)));
%     size(im)
%     im = im./max(abs(im(:)));
%     im = 20*log10(im);
%     imagesc(xax_bf*1e3, zax_bf*1e3, im)
%     axis equal tight
%     cb = colorbar;    
%     cb.Label.String = '[dB]'; colormap(ax7, gray(256));
%     caxis([-60 0]);
%     title('B-mode image') ;
%     % VN recon
%     ax8 = subplot(img_per_page, 3, p+2);
%     r = recon(i,:,:);
%     imagesc(xax_sos*1e3, zax_sos*1e3, squeeze(r)'); colorbar; caxis([1460, 1540]);
%     axis equal tight; colormap(ax8, jet);
%     title('VN reconstruction') ;
% 
%     % LBFGS recon
%     ax9 = subplot(img_per_page,3,p+3);
%     r = recon_lbfgs(:,:,i);
%     imagesc(xax_sos*1e3, zax_sos*1e3, squeeze(r)); colorbar; caxis([1460, 1540]);
%     axis equal tight; colormap(ax9, jet);
%     title('LBFGS reconstruction') ;
%     p = p + 3;
%     if mod(i, img_per_page)== 0 || i == Nimgs
%         tmp = sprintf('all_phantom_vis_%d.png', page_nb);
%         saveas(gcf, fullfile(exp_dir, tmp));
%         clf;
%         p=0;
%         page_nb = page_nb + 1;
%     end
% end
%     drawnow 
%   
 %% Good phantom
load(fullfile(exp_dir, 'test-syn-0.0-test-fukuda_1_imgs.mat'))
if SAVE_BF == true
load('/Users/melaniebernhardt/Desktop/MScThesis/USImageReconstruction/data/fukuda_1_imgs.mat');
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

%     din2 = reshape(din, [84, 64, 6, 1]);
%     subplot(3,3,1);
%     imagesc(din2(:,:,1 ,i)); colorbar; axis equal tight; colormap(parula);
%     subplot(3,3,2);
%     imagesc(din2(:,:,2,i));colorbar; axis equal tight;
%     subplot(3,3,3);
%     imagesc(din2(:,:,3,i));colorbar; axis equal tight;
%     subplot(3,3,4);
%     imagesc(din2(:,:,4,i));colorbar; axis equal tight;
%     subplot(3,3,5);
%     imagesc(din2(:,:,5,i));colorbar; axis equal tight;
%     subplot(3,3,6);
%     imagesc(din2(:,:,6,i));colorbar; axis equal tight;
% 
%     % BF
%     ax7 = subplot(3,3,7);
%     bf = bf_im(:,:,:, i);
%     im = abs(hilbert(mean(bf,3)));
%     size(im)
%     im = im./max(abs(im(:)));
%     im = 20*log10(im);
%     imagesc(xax_bf*1e3, zax_bf*1e3, im)
%     axis equal tight
%     cb = colorbar;    
%     cb.Label.String = '[dB]'; colormap(ax7, gray(256));
%     caxis([-60 0]);
%     title('B-mode image') ;
%     % VN recon
%     ax8 = subplot(3,3,8);
%     r = recon(i,:,:);
%     imagesc(xax_sos*1e3, zax_sos*1e3, squeeze(r)'); colorbar; caxis([1460, 1540]);
%     axis equal tight; colormap(ax8, hot);
%     title('VN reconstruction') ;
% 
%     % LBFGS recon
%     ax9 = subplot(3,3,9);
%     r = recon_lbfgs(:,:,i);
%     imagesc(xax_sos*1e3, zax_sos*1e3, squeeze(r)); colorbar;caxis([1460, 1540]);
%     axis equal tight; colormap(ax9, hot);
%     title('LBFGS reconstruction') ;
% 
%     tmp = sprintf('good_phantom_vis_%d.png', i);
%     saveas(gcf, fullfile(exp_dir, tmp))
% 	clf;
% ax7 = subplot(1,3,1);
%     bf = bf_im(:,:,:, i);
%     im = abs(hilbert(mean(bf,3)));
%     size(im)
%     im = im./max(abs(im(:)));
%     im = 20*log10(im);
%     imagesc(xax_bf*1e3, zax_bf*1e3, im)
%     axis equal tight;
%     cb = colorbar;    
%     cb.Label.String = '[dB]'; colormap(ax7, gray(256));
%     caxis([-60 0]);
%     title('B-mode image') ;
%     % VN recon
%     ax8 = subplot(1,3,2);
%     r = recon(i,:,:);
%     imagesc(xax_sos*1e3, zax_sos*1e3, squeeze(r)'); colorbar; caxis([1460, 1540]);
%     axis equal tight; colormap(ax8, jet);
%     title('VN reconstruction') ;
% 
%     % LBFGS recon
%     ax9 = subplot(1,3,3);
%     r = recon_lbfgs(:,:,i);
%     imagesc(xax_sos*1e3, zax_sos*1e3, squeeze(r)); colorbar;caxis([1460, 1540]);
%     axis equal tight; colormap(ax9, jet);
%     title('LBFGS reconstruction') ;
% tmp = sprintf('one_good_phantom_vis_%d.png', i);
%     saveas(gcf, fullfile(exp_dir, tmp))
