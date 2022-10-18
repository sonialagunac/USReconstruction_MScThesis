%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Mélanie Bernhardt - M.Sc. Thesis - ETH Zurich
%% This file is used to investigate the difference in measurements and layer 
%% values in presence of the domain shift.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%data_dir = '/Volumes/MelSSD/runs/15_prog_usm_30l_noise/eval-vn-150000/';
data_dir = '/scratch_net/biwidl307/sonia/USImageReconstruction-Melanie/runs/32_mix_ideal_20l_reg/eval-vn-120000/';
m_test_syn = load(fullfile(data_dir, 'test-syn-0.0-0.6-patchy-testset_ideal_MS_32_imgs.mat'));
m_test_full = load(fullfile(data_dir, 'test-syn-0.0-0.6-test-fullpipeline_testset_6comb_32_imgs.mat'));
m_14_phanthom = load(fullfile(data_dir, 'test-syn-0.0-test-phantom_trial1_14_imgs.mat'));
m_val = load(fullfile(data_dir, 'test-syn-0.0-0.6-patchy-val_MS_6comb.mat'));
m_good_ph = load(fullfile(data_dir, 'test-syn-0.0-test-fukuda_1_imgs.mat'));
%% Comparative statistics inmeasurement distibutions (useless)
mask_test_full = logical(m_test_full.din~=0) ;
mask_phantom = logical(m_14_phanthom.din~=0) ;
mask_test_syn = logical(m_test_syn.din~=0) ;
m1 = max(m_test_syn.din(:));
mask1 = logical(mask_test_full .* mask_test_syn) ;
clf;
subplot(1,2,1) ;
histogram((m_test_full.din(mask1) - m_test_syn.din(mask1))./m1); title({'Distribution of difference', 'in din for test set', 'ray-based vs fullpipeline', 'normalized by max of ray-based'}) ;
subplot(1,2,2) ;
histogram((m_test_full.din_whiten(mask1) - m_test_syn.din_whiten(mask1))./max(m_test_syn.din_whiten(:))); title({'Distribution of difference', 'in din whiten for test set', 'ray-based vs fullpipeline', 'normalized by max of ray-based'}) ;
%% Histogram of meas values (overall not per image)
subplot(1,3,1) ; 
histogram(m_test_full.din(m_test_full.din~=0)); title('Distribution of din for test set ray-based') ; 
subplot(1,3,2);
histogram(m_test_syn.din(m_test_syn.din~=0)); title('Distribution of din for test set full-pipeline') ; 
subplot(1,3,3);
histogram(m_14_phanthom.din(m_14_phanthom.din~=0)); title('Distribution of din for 14 phantom dataset') ; 
%% Histogram of whiten meas values (overall not per image)
subplot(1,4,1);
histogram(m_train.din_whiten(m_train.din_whiten~=0), 100, 'BinLimits', [-5, 5],'Normalization', 'probability'); title('Distribution of din whiten for one training batch') ; 
subplot(1,4,2);
histogram(m_test_syn.din_whiten(m_test_syn.din_whiten~=0), 100, 'BinLimits', [-5, 5],'Normalization', 'probability'); title('Distribution of din whiten for test set ray-based') ; 
subplot(1,4,3) ; 
histogram(m_test_full.din_whiten(m_test_full.din_whiten~=0), 100, 'BinLimits', [-5, 5],'Normalization', 'probability'); title('Distribution of din whiten for test set full-pipeline') ; 
subplot(1,4,4);
histogram(m_14_phanthom.din_whiten(m_14_phanthom.din_whiten~=0), 100, 'BinLimits', [-5, 5],'Normalization', 'probability'); title('Distribution of din whiten for 14 phantom dataset') ; 

%% Test set comparison between syn and full
figure()
for i = 1:5
t2 = m_test_full.din_whiten(:,i+10);
t1 = m_test_syn.din_whiten(:,i+10);
meas1 = t1;
meas1(meas1==0) = nan ; 
meas1 = reshape(meas1, [84, 64*6]);
meas2 = t2;
meas2(meas2==0) = nan ; 
meas2 = reshape(meas2, [84, 64*6]);
std(t1(t1~=0))
std(t2(t2~=0))
r1 = squeeze(m_test_syn.recon(i+10, :,:));
r2 = squeeze(m_test_full.recon(i+10, :,:));
init1 = squeeze(m_test_syn.init_img(i+10, :,:));
init2 = squeeze(m_test_full.init_img(i+10, :,:));
subplot(5,4,(i-1)*4+1);
imagesc(meas1) ; set(gca,'XTick',[], 'YTick', []); axis equal tight; title(sprintf('Whiten syn meas %d', i)); colorbar;
subplot(5,12,(i-1)*12+4);
imagesc(init1') ; set(gca,'XTick',[], 'YTick', []); axis equal tight; title(sprintf('X init syn %d', i)); colorbar;
subplot(5,12,(i-1)*12+5);
imagesc(r1'); set(gca,'XTick',[], 'YTick', []); axis equal tight; title(sprintf('Recon syn %d', i)); colorbar;
subplot(5,12,(i-1)*12+6);
histogram(t1(t1~=0), 200, 'BinLimits', [-5, 5], 'Normalization', 'probability') ; title(sprintf('Syn test image %d', i)); ylim([0, 0.25]);
subplot(5,12,(i-1)*12+7);
histogram(t2(t2~=0), 200, 'BinLimits', [-5, 5], 'Normalization', 'probability') ; title(sprintf('Full test image %d', i)); ylim([0, 0.25]);
subplot(5,12,(i-1)*12+8);
imagesc(r2'); set(gca,'XTick',[], 'YTick', []); axis equal tight; title(sprintf('Recon full %d', i)); colorbar;
subplot(5,12,(i-1)*12+9);
imagesc(init2') ; set(gca,'XTick',[], 'YTick', []); axis equal tight; title(sprintf('X init full %d', i)); colorbar;
subplot(5,4,(i-1)*4+4);
imagesc(meas2) ; set(gca,'XTick',[], 'YTick', []); axis equal tight; title(sprintf('Whiten syn meas %d', i)); colorbar;
end

%% Same for phantom 14
figure()
for i = 1:5
t1 = m_14_phanthom.din_whiten(:,i);
meas1 = t1;
meas1(meas1==0) = nan ; 
meas1 = reshape(meas1, [84, 64*6]);
std(t1(t1~=0))
r1 = squeeze(m_14_phanthom.recon(i, :,:));
init1 = squeeze(m_14_phanthom.init_img(i, :,:));
subplot(5,2,(i-1)*2+1);
imagesc(meas1) ; set(gca,'XTick',[], 'YTick', []); axis equal tight; title(sprintf('Whiten meas %d', i)); colorbar;
subplot(5,6,(i-1)*6+4);
imagesc(init1') ; set(gca,'XTick',[], 'YTick', []); axis equal tight; title(sprintf('X init %d', i)); colorbar;
subplot(5,6,(i-1)*6+5);
imagesc(r1'); set(gca,'XTick',[], 'YTick', []); axis equal tight; title(sprintf('Recon %d', i)); colorbar;
subplot(5,6,(i-1)*6+6);
histogram(t1(t1~=0), 200, 'BinLimits', [-5, 5], 'Normalization', 'probability') ; title(sprintf('Phantom image %d', i)); ylim([0, 0.25]);
end

%% Phantom 1
clf;
t1 = m_good_ph.din_whiten(:,1);
meas1 = t1;
meas1(meas1==0) = nan ; 
meas1 = reshape(meas1, [84, 64*6]);
std(t1(t1~=0))
r1 = squeeze(m_good_ph.recon(1,:,:));
init1 = squeeze(m_good_ph.init_img(1,:,:));
subplot(1,2,1);
imagesc(meas1) ; set(gca,'XTick',[], 'YTick', []); axis equal tight; title(sprintf('Whiten meas %d', i)); colorbar;
subplot(1,6,4);
imagesc(init1') ; set(gca,'XTick',[], 'YTick', []); axis equal tight; title(sprintf('X init %d', i)); colorbar;
subplot(1,6,5);
imagesc(r1'); set(gca,'XTick',[], 'YTick', []); axis equal tight; title(sprintf('Recon %d', i)); colorbar;
subplot(1,6,6);
histogram(t1(t1~=0), 200, 'BinLimits', [-5, 5], 'Normalization', 'probability') ; title(sprintf('Phantom image %d', i));



%% Comparative statistics in pre-activations distibutions - data term
clf;
for p=1:10
    ymax = 0.5;
    xlim = 1 ;
    maxl = m_test_syn.interp_max(p) ;
    tmp_syn = m_test_syn.before_act_data(p,:,:);
    tmp_full = m_test_full.before_act_data(p,:,:);
    tmp_phantom =  m_14_phanthom.before_act_data(p,:,:);
    tmp_train =  m_train.before_act_data(p,:,:);
    subplot(10,4,4*(p-1)+2);
    histogram(tmp_syn(tmp_syn~=0), 200, 'BinLimits', [-xlim,xlim], 'Normalization', 'probability') ; pbaspect([1 1 1]) ; ylim([0, ymax]); %title({sprintf('Layer %d ray test', p)});
    subplot(10,4,4*(p-1)+3);
    histogram(tmp_full(tmp_full~=0), 200, 'BinLimits', [-xlim,xlim], 'Normalization', 'probability'); pbaspect([1 1 1]) ;  ylim([0, ymax]); %title({sprintf('Layer %d full test', p)});
    subplot(10,4,4*(p-1)+4);
    histogram(tmp_phantom(tmp_phantom~=0), 200, 'BinLimits', [-xlim,xlim], 'Normalization', 'probability') ; pbaspect([1 1 1]) ; ylim([0, ymax]); %title({sprintf('Layer %d phantom', p)});
    subplot(10,4,4*(p-1)+1);
    histogram(tmp_train(tmp_train~=0), 200, 'BinLimits', [-xlim,xlim], 'Normalization', 'probability') ;  pbaspect([1 1 1]) ;ylim([0, ymax]);
end
%%
clf;
for p=1:10
    ymax = 0.4;
    xlim = 1 ;
    maxl = m_test_syn.interp_max(p) ;
    tmp_syn = m_test_syn.before_act_data(p+10,:,:);
    tmp_full = m_test_full.before_act_data(p+10,:,:);
    tmp_phantom =  m_14_phanthom.before_act_data(p+10,:,:);
    tmp_train =  m_train.before_act_data(p,:,:);
    subplot(10,4,4*(p-1)+2);
    histogram(tmp_syn(tmp_syn~=0), 200, 'BinLimits', [-xlim,xlim], 'Normalization', 'probability') ; pbaspect([1 1 1]) ; ylim([0, ymax]); %title({sprintf('Layer %d ray test', p)});
    subplot(10,4,4*(p-1)+3);
    histogram(tmp_full(tmp_full~=0), 200, 'BinLimits', [-xlim,xlim], 'Normalization', 'probability'); pbaspect([1 1 1]) ;  ylim([0, ymax]); %title({sprintf('Layer %d full test', p)});
    subplot(10,4,4*(p-1)+4);
    histogram(tmp_phantom(tmp_phantom~=0), 200, 'BinLimits', [-xlim,xlim], 'Normalization', 'probability') ; pbaspect([1 1 1]) ; ylim([0, ymax]); %title({sprintf('Layer %d phantom', p)});
    subplot(10,4,4*(p-1)+1);
    histogram(tmp_train(tmp_train~=0), 200, 'BinLimits', [-xlim,xlim], 'Normalization', 'probability') ;  pbaspect([1 1 1]) ;ylim([0, ymax]);
end
%%
clf;
for p=1:10
    ymax = 0.4;
    xlim = 1 ;
    maxl = m_test_syn.interp_max(p) ;
    tmp_syn = m_test_syn.before_act_data(p+20,:,:);
    tmp_full = m_test_full.before_act_data(p+20,:,:);
    tmp_phantom =  m_14_phanthom.before_act_data(p+20,:,:);
    tmp_train =  m_train.before_act_data(p,:,:);
    subplot(10,4,4*(p-1)+2);
    histogram(tmp_syn(tmp_syn~=0), 200, 'BinLimits', [-xlim,xlim], 'Normalization', 'probability') ; pbaspect([1 1 1]) ; ylim([0, ymax]); %title({sprintf('Layer %d ray test', p)});
    subplot(10,4,4*(p-1)+3);
    histogram(tmp_full(tmp_full~=0), 200, 'BinLimits', [-xlim,xlim], 'Normalization', 'probability'); pbaspect([1 1 1]) ;  ylim([0, ymax]); %title({sprintf('Layer %d full test', p)});
    subplot(10,4,4*(p-1)+4);
    histogram(tmp_phantom(tmp_phantom~=0), 200, 'BinLimits', [-xlim,xlim], 'Normalization', 'probability') ; pbaspect([1 1 1]) ; ylim([0, ymax]); %title({sprintf('Layer %d phantom', p)});
    subplot(10,4,4*(p-1)+1);
    histogram(tmp_train(tmp_train~=0), 200, 'BinLimits', [-xlim,xlim], 'Normalization', 'probability') ;  pbaspect([1 1 1]) ;ylim([0, ymax]);
end

%% Compute undersampling rate on phantom
usr = mean(mean(m_14_phanthom.din==0, 1)) % mean usm rate is 0.8
usr_max = max(mean(m_14_phanthom.din==0, 1)) % max usm rate is 0.94
usr_min = min(mean(m_14_phanthom.din==0, 1)) % min usm rate is 0.73
usr_good = mean(m_good_ph.din==0) % usm rate 'good' phantom is 0.78

%% Good phnatom against bad.

i = 13; % is bad
usr_bad = mean(m_14_phanthom.din(:, i)==0) % 0.78
usr_good = mean(m_good_ph.din==0) % 0.78
bad = m_14_phanthom.din_whiten(:, i);
good = m_good_ph.din_whiten(:,1);
good(good==0) = nan
goodtest = m_test_syn.din_whiten(:, 3);
goodtest(goodtest==0) = nan;
bad(bad==0) = nan;
figure()
clf;
subplot(3,3,7) ;
histogram(bad(bad~=0), 300, 'BinLimits', [-2, 2], 'Normalization', 'probability'); title({'Distribution of', 'din whiten on bad phantom example'}) ; ylim([0 0.15]);
subplot(3,3,8);
histogram(m_good_ph.din_whiten(m_good_ph.din_whiten~=0), 300, 'BinLimits', [-2, 2], 'Normalization', 'probability'); title({'Distribution of', 'din whiten on good phantom example'}) ; ylim([0 0.15]);
subplot(3,3,9);
histogram(goodtest(goodtest~=0), 300, 'BinLimits', [-2, 2], 'Normalization', 'probability'); title({'Distribution of', 'din whiten for one good test set ray-based example'}) ; ylim([0 0.15]);
subplot(3,1,1);
imagesc(reshape(bad, [84, 64*6])); title('Meas bad'); colorbar;
subplot(3,1,2);
imagesc(reshape(good, [84, 64*6])); title('Meas good'); colorbar;
%%
clf;
image=1;
m = m_good_ph ;
nlayer = 30;
nx = 64; ny = 84
    set(gcf, 'Position',  [1, 1, 600, 2000]); 
    layer_img = reshape(m.layer_sos(:,image,:,:), [nlayer+1, nx, ny]);
    subplot(8, 3, 1);  % layer 0
    imagesc(squeeze(layer_img(1, :,:))'); colorbar; title('Init reconstruction');
    data_grad1 = reshape(m.data_grad(:,image,:,:), [nlayer, nx, ny]);
    reg_grad1 = reshape(m.reg_grad(:,image,:,:), [nlayer, nx, ny]);
    layer_img = reshape(m.layer_sos(:,image,:,:), [nlayer+1, nx, ny]);
    for layer=1:7
        subplot(8, 3, layer*3 + 1) ;
        imagesc(squeeze(layer_img(layer+1, :,:))'); colorbar;
        title(sprintf('Reconstruction layer %d', layer));
        subplot(8, 3, layer*3 + 2) ;
        imagesc(squeeze(data_grad1(layer, :,:))'); colorbar;
        title(sprintf('Data grad layer %d', layer));        
        subplot(8, 3, layer*3 + 3) ;
        imagesc(squeeze(reg_grad1(layer, :,:))'); colorbar;
        title(sprintf('Regularization grad layer %d', layer));
    end
    %%
    clf;
    for layer=0:6
        subplot(7, 3, layer*3 + 1) ;
        imagesc(squeeze(layer_img(8+layer+1, :,:))'); colorbar;
        title(sprintf('Reconstruction layer %d', 8+layer));
        subplot(7, 3, layer*3 + 2) ;
        imagesc(squeeze(data_grad1(8+layer, :,:))'); colorbar;
        title(sprintf('Data grad layer %d', 8+layer));        
        subplot(7, 3, layer*3 + 3) ;
        imagesc(squeeze(reg_grad1(8+layer, :,:))'); colorbar;
        title(sprintf('Regularization grad layer %d', 8+layer));
    end
    %%
    
    clf;
    for layer=0:6
        subplot(7, 3, layer*3 + 1) ;
        imagesc(squeeze(layer_img(15+layer+1, :,:))'); colorbar;
        title(sprintf('Reconstruction layer %d', layer+15));
        subplot(7, 3, layer*3 + 2) ;
        imagesc(squeeze(data_grad1(15+layer, :,:))'); colorbar;
        title(sprintf('Data grad layer %d', layer+15));        
        subplot(7, 3, layer*3 + 3) ;
        imagesc(squeeze(reg_grad1(15+layer, :,:))'); colorbar;
        title(sprintf('Regularization grad layer %d', layer+15));
    end
    %%
        clf;
    for layer=0:8
        subplot(9, 3, layer*3 + 1) ;
        imagesc(squeeze(layer_img(22+layer+1, :,:))'); colorbar;
        title(sprintf('Reconstruction layer %d', layer+22));
        subplot(9, 3, layer*3 + 2) ;
        imagesc(squeeze(data_grad1(22+layer, :,:))'); colorbar;
        title(sprintf('Data grad layer %d', layer+22));        
        subplot(9, 3, layer*3 + 3) ;
        imagesc(squeeze(reg_grad1(22+layer, :,:))'); colorbar;
        title(sprintf('Regularization grad layer %d', layer+22));
    end
    
    %%
    m=m_14_phanthom;
    clf;
set(gcf, 'Position',  [100, 100, 1500, 1500]);
for p=1:30
    subplot(5,6,p);
    tmp = m.before_act_data(p,:,13);
    histogram(tmp(tmp~=0), 300) ;
    title({sprintf('Layer %d', p),sprintf('min %0.2f max %0.2f', min(min(m.before_act_data(p,:,:))), max(max(m.before_act_data(p,:,:)))),''});
end

%% Sanity check
std(goodtest(goodtest~=0))
std(m_good_ph.din_whiten(m_good_ph.din_whiten~=0))
%% Compute sensitivity of reconstruction quality with respect to undersampling rate of ray-based test set.  
m_2 = load(fullfile(data_dir, 'test-syn-0.0-0.2-patchy-testset_ideal_MS_32_imgs.mat'));
m_4 = load(fullfile(data_dir, 'test-syn-0.0-0.4-patchy-testset_ideal_MS_32_imgs.mat'));
m_6 = load(fullfile(data_dir, 'test-syn-0.0-0.6-patchy-testset_ideal_MS_32_imgs.mat'));
m_7 = load(fullfile(data_dir, 'test-syn-0.0-0.7-patchy-testset_ideal_MS_32_imgs.mat'));
m_8 = load(fullfile(data_dir, 'test-syn-0.0-0.8-patchy-testset_ideal_MS_32_imgs.mat'));
m_9 = load(fullfile(data_dir, 'test-syn-0.0-0.9-patchy-testset_ideal_MS_32_imgs.mat'));

RMSEvn_2 = RMSE(32, m_2);
RMSEvn_4 = RMSE(32, m_4);
RMSEvn_6 = RMSE(32, m_6);
RMSEvn_7 = RMSE(32, m_7);
RMSEvn_8 = RMSE(32, m_8);
RMSEvn_9 = RMSE(32, m_9);

plot(1:6, [mean(RMSEvn_2(:)), mean(RMSEvn_4(:)), mean(RMSEvn_6(:)), mean(RMSEvn_7(:)), mean(RMSEvn_8(:)), mean(RMSEvn_9(:))]);
%% Functions
function RMSEvn = RMSE(Nimgs, m)
    for p=1:Nimgs
        try
            xtrue = m.gt_sos(p,:,:);
        catch 
            xtrue = 1./m.xgt_slowness(p,:,:);
        end
        recon = squeeze(m.recon(p,:,:));	
        RMSEvn(p) = sqrt(mean(power(recon(:) - xtrue(:), 2)));
    end
end