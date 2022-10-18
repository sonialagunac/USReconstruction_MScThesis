% Sonia Laguna ETH Zurich, September 2022
% Codes used to create the Matlab figures in the MSc thesis

%% Explanation of SoS recons pipeline, bmode figures and measurements
%% Using phantom for MS
load('/scratch_net/biwidl307/sonia/USImageReconstruction-Melanie/runs/4_ICFP_reg1e5_tau5/eval-vn-120000/test-syn-0.0-test-newMSphantom_03.mat', 'recon');
load('/scratch_net/biwidl307/sonia/data_original/phantom_mert/newMSphantom_03.mat')
chan_comb = [15, 25 ; 35, 45 ; 55, 65; 75, 85; 95, 105;105, 115]; %channels to be used in DT
%%
BF = bf_im(:,:,55);
BF = abs(hilbert((BF))); BF = BF./max(abs(BF(:)));
BF = 20 * log10(BF);
%f1 = figure;
xax_sos= [-0.0189, 0.0189];
zax_sos = [0,0.04];
pixelsize_bf = [1/8, 1/2]*0.0003
xax_bf = xax_sos(1):pixelsize_bf(2):xax_sos(end);
zax_bf = [zax_sos(1):pixelsize_bf(1):zax_sos(end)];
figure;
imagesc(xax_bf*1000, zax_bf*1000, BF(1:1067,:))
%imagesc(BF(1:1067,:))
axis equal tight
ax = gca;
ax.XTick = [];
ax.YTick = [];
cb = colorbar;
cb.Label.String = '[dB]';
sp2 = gca;
colormap(sp2,gray(256))
caxis([-50 0])

BF = RF;
BF = abs(hilbert(mean(BF,3))); BF = BF./max(abs(BF(:)));
BF = 20 * log10(BF);
%f1 = figure;
xax_sos= [-0.0189, 0.0189];
zax_sos = [0,0.04];
pixelsize_bf = [1/8, 1/2]*0.0003
xax_bf = xax_sos(1):pixelsize_bf(2):xax_sos(end);
zax_bf = [zax_sos(1):pixelsize_bf(1):zax_sos(end)];
figure;
imagesc(xax_bf*1000, zax_bf*1000, BF(1:1067,:))
%imagesc(BF(1:1067,:))
axis equal tight
ax = gca;
ax.XTick = [];
ax.YTick = [];
cb = colorbar;
cb.Label.String = '[dB]';
sp2 = gca;
colormap(sp2,gray(256))
caxis([-50 0])
msm = reshape(measmnts, [84,64,6,1]);
figure, imagesc(msm(1:67,:,3)), colorbar
axis equal tight
figure, imagesc(squeeze(recon(1,:,1:67))'), colorbar, axis equal tight

%% Virtual Source data example, clinical
load('/scratch_net/biwidl307/sonia/data_original/VS/subjects/mat_sos/mpBUS017_L1_large.mat')
figure, imagesc(msm(1:67,:,4,2)), colorbar, axis equal tight
chan_comb = [1,20; 20,9; 9,16; 16,4; 4,23; 23,12; 12,19; 19,7; 32,27; 27,45; 45,40; 40,35; 35,30; 30,48; 48,33]; % Fast Sequence, Delta = 12, Ncomb = 15
BF = bf_im(:,:,7,2);
BF = abs(hilbert((BF))); BF = BF./max(abs(BF(:)));
BF = 20 * log10(BF);
%f1 = figure;
xax_sos= [-0.0189, 0.0189];
zax_sos = [0,0.04];
pixelsize_bf = [1/8, 1/2]*0.0003
xax_bf = xax_sos(1):pixelsize_bf(2):xax_sos(end);
zax_bf = [zax_sos(1):pixelsize_bf(1):zax_sos(end)];
figure;
imagesc(xax_bf*1000, zax_bf*1000, BF(1:1067,:))
%imagesc(BF(1:1067,:))
axis equal tight
ax = gca;
ax.XTick = [];
ax.YTick = [];
cb = colorbar;
cb.Label.String = '[dB]';
sp2 = gca;
colormap(sp2,gray(256))
caxis([-50 0])
%% 
load('/scratch_net/biwidl307/sonia/USImageReconstruction-Melanie/runs/4_ICFP_reg1e5_tau5_L2_VS/sos_eval-vn-120000/test-syn-0.0-test-mpBUS017_L1.mat', 'recon')
figure, imagesc(squeeze(recon(2,:,1:67))'), colorbar

%% Correlation maps, phantom data
load('/scratch_net/biwidl307/sonia/data_original/phantom_mert/newMSphantom_04.mat', 'CorrCoeff', 'measmnts')
msm = reshape(measmnts, [84,64,6,1]);
figure, imagesc(CorrCoeff(1:67,:,4)), colorbar, colormap('hot'),axis equal tight
figure, imagesc(msm(1:67,:,4)*1e6), colorbar,axis equal tight
load('/scratch_net/biwidl307/sonia/data_original/phantom_mert/newMSphantom_02.mat', 'CorrCoeff', 'measmnts')
msm = reshape(measmnts, [84,64,6,1]);
figure, imagesc(CorrCoeff(1:67,:,4)), colorbar,colormap('hot'),axis equal tight
figure, imagesc(msm(1:67,:,4)*1e6), colorbar,axis equal tight

%% L meas, confidence map of phantom data
load('/scratch_net/biwidl307/sonia/data_original/phantom_mert/newMSphantom_04.mat', 'measmnts','L')
msm = reshape(measmnts, [84,64,6,1]);
for i = 1:6
figure, imagesc(msm(1:67,:,i)*1e6),colorbar,axis equal tight, caxis([-1 1])
end
cop = ones(size(measmnts));
cop(isnan(measmnts)) = 0 ;
mult_a = abs(L')*cop;
mult = reshape(mult_a,[84,64]);
mult = mult(7:67,:);
mult = (mult - min(min(mult(7:end,:))))/(max(max(mult(7:end)))-max(min(mult(7:end,:))));
figure, imagesc(mult),colorbar, axis equal tight, colormap('hot')

%% Residual of IC data
load('/scratch_net/biwidl307/sonia/data_original/test/testset_ideal_MS_32_imgs.mat')
%load('/scratch_net/biwidl307/sonia/USImageReconstruction-Melanie/runs/4_ICFP_reg1e5_tau5_L2/eval-vn-120000/test-syn-0.0-0.1-patchy-testset_ideal_MS_32_imgs.mat')
load('/scratch_net/biwidl307/sonia/USImageReconstruction-Melanie/runs/4_ICFP_reg1e5_tau5_nomask/eval-vn-120000/test-syn-0.0-0.0-patchy-testset_ideal_MS_32_imgs.mat')
num = 1;
%recon_1 = ((L_fact./squeeze(recon(num,:,:))-k(num))/std(num))';
%recon_1 =(squeeze(xs_last(num,:,:))');
recon_1 =(squeeze(xs(end,num,:,:))');
orig = L*(recon_1(:));
orig(din_whiten(:,num) ==0) = nan;
din_whiten(din_whiten == 0) = nan;
res = orig - din_whiten(:,num);
mes = reshape(din_whiten(:,num),[84,64,6]);
res = reshape(res,[84,64,6]);
p = reshape(orig, [84,64,6]);  p(p==0) = nan;

figure, imagesc((res(:,:,3))), colorbar, title('res 1'),axis equal tight
figure, imagesc((p(:,:,3))), colorbar, title('orig 1'),axis equal tight
figure, imagesc((mes(:,:,3))), colorbar, title('meas 1'),axis equal tight

mes(isnan(mes)) = 0 ;
res(isnan(res)) = 0;
p(isnan(p))=0;

mes_m = mean(mes,3);mes_m(mes_m==0) = nan;
res_m = mean(res,3);res_m(res_m==0) = nan;
p_m = mean(p,3); p_m(p_m==0) = nan;

figure, imagesc((res_m(:,:))), colorbar, title('res 1'),axis equal tight
figure, imagesc((p_m(:,:))), colorbar, title('orig 1'),axis equal tight
figure, imagesc((mes_m(:,:))), colorbar, title('meas 1'),axis equal tight