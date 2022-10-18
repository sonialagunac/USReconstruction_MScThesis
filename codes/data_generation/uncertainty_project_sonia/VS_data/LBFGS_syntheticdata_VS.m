% LBFGS computation of simulated VS data
%Sonia Laguna, ETH Zurich MSc Thesis, August 2022
%%
filenames = [
    "test-syn-0.0-0.0-test-train_VS_15comb_fullpipeline_30.mat",
    "test-syn-0.0-0.1-patchy-train_VS_15comb_IC_30.mat",
    "test-syn-0.0-0.5-patchy-train_VS_15comb_IC_30.mat"
   ];
data_dir = '/scratch_net/biwidl307/sonia/USImageReconstruction-Melanie/runs/4_ICFP_reg1e5_tau5_VS/eval-vn-120000/';
lbfgs_dir = '/scratch_net/biwidl307/sonia/USImageReconstruction-Melanie/runs/lbfgs';
NA = 15;
load('/scratch_net/biwidl307/sonia/data_original/VS/subjects/opts_2.mat', 'opts')
for idx =1:numel(filenames)
    filename = filenames(idx)
    m = load(fullfile(data_dir, sprintf('%s', filename)));
    [Nimgs,nx, ny] = size(m.init_img) ;
    for p =1:Nimgs
        disp(p);
        out = m.din(:,p);
        out(out==0) = nan ;
        out = reshape(out, [84, 64, NA, 1]);
        out_d = cat(4,out,out);
        opts.postprocess.pipeline = {'sos_minus'}; % For the sos reconstruction iterative pipeline
        [sos_recon,opts] = postprocess(opts,out_d);
        recon_lbfgs(p,:,:) = sos_recon;
    end
    save(fullfile(lbfgs_dir, sprintf('lbfgs-2%s', filename)), 'recon_lbfgs');
end