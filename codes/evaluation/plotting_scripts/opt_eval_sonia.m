%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Melanie Bernhardt - M.Sc. Thesis - ETH Zurich
%% This file is used to compute the image and data residuals
%% for evaluation of the optimization behavior learned by the network.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Directories adapted to Sonia Laguna, MSc Thesis 2022
clear
data_dir = '/scratch_net/biwidl307/sonia/USImageReconstruction-Melanie/runs/4_ICFP_reg1e5_tau5/eval-vn-120000'
save_dir = fullfile(data_dir, 'opt-eval');
m_val = load(fullfile(data_dir, 'test-syn-0.0-0.1-patchy-testset_ideal_MS_32_imgs.mat'));
data = load('/scratch_net/biwidl307/sonia/data_original/test/fullpipeline_testset_6comb_32_imgs.mat');
addpath(genpath('/scratch_net/biwidl307/sonia/USImageReconstruction-Melanie/codes/evaluation/Generic_Data_Processing_Structure-master'));
mkdir(save_dir);

%% Retrieve L, compute D
L = data.L * data.L_fact;
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
ElementPositions = [0:128-1].*pitch;
ElementPositions = ElementPositions-mean(ElementPositions);
opts.postprocess.sos_minus.RegularizationDensityWeight = 2;
chan_comb = [15, 25 ; 
            35, 45 ; 
            55, 65;
            75, 85;
            95, 105;
            105, 115];
Mask = gather(reshape(data.maskFixed, [84, 64, 6]));
D = SobelReg_MS(Mask,xax_sos,zax_sos,ElementPositions,chan_comb,opts.postprocess.sos_minus.RegularizationDensityWeight);
D = full(D);
%%
[data_res, img_res, d_grad, r_grad, c, corr_r_d] = compute_all_residuals(L, D, m_val, permute(m_val.gt_slowness, [1,3,2]));

%%
figure();
plot(0:20, mean(data_res, 1)) ; xlabel('Layer'); ylabel('Mean data residual'); set(gca, 'FontName', 'Palatino', 'FontSize', 13);
saveas(gcf, fullfile(save_dir, 'data_res.png'));
figure();
plot(0:20, mean(img_res, 1)); xlabel('Layer'); ylabel('Mean image residual'); set(gca, 'FontName', 'Palatino', 'FontSize', 13);
saveas(gcf, fullfile(save_dir, 'img_res.png'));
%%
plot(0:20, mean(sum(abs(d_grad), [3,4]), 1)) ; xlabel('Layer') ; ylabel('||\nabla_d^{(layer)}||_1'); set(gca, 'FontName', 'Palatino', 'FontSize', 13);
saveas(gcf, fullfile(save_dir, 'd_grad.png'));
%%
plot(0:20, mean(sum(abs(r_grad), [3, 4]), 1)); xlabel('Layer') ; ylabel('||\nabla_r^{(layer)}||_1'); set(gca, 'FontName', 'Palatino', 'FontSize', 13);
saveas(gcf, fullfile(save_dir, 'r_grad.png'));
%%
plot(0:20, mean(corr_r_d,1)); xlabel('Layer') ; ylabel('Correlation coefficient'); set(gca, 'FontName', 'Palatino', 'FontSize', 13);
saveas(gcf, fullfile(save_dir, 'corr.png'));
%%
save(fullfile(save_dir, 'residuals.mat'), 'data_res', 'img_res', 'd_grad', 'r_grad'); 
%%
function [data_res, img_res, d_grad, r_grad, c, corr_r_d] = compute_all_residuals(L, D, m_val_ray, xstar)
    xs_hat = permute(m_val_ray.layer_sos, [1, 4, 3, 2]) ; % [layer, 84, 64, Nimgs]
    [Nlayers, ~, ~, Nimgs] = size(xs_hat) ;
    for i=1:Nimgs
        i
        dhat = m_val_ray.din(:,i);
        xstari = xstar(i,:,:);
        for l=1:Nlayers
            xi = 1./xs_hat(l, :,:,i);
            sigmai = xs_hat(l, :,:,i);
            data_res(i,l) = sum(abs(L*xi(:) - dhat));
            img_res(i,l) = sum(abs(sigmai(:) - 1./xstari(:)));
            d_grad(i,l, :, :) = L' * sign(L*xi(:) - dhat) ;
            r_grad(i,l, :, :) = D' * sign(D*xi(:));
            c = corrcoef(L' * sign(L*xi(:) - dhat), D' * sign(D*xi(:)));
            corr_r_d(i,l) =  c(1,2);
        end
    end        
end
