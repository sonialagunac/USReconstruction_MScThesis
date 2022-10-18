% Sonia Laguna -ETH Zurich -MSc Thesis
% Stacking frames of same lesions in clinical data

clear all
close all
%Creating the plots all togethe
load('/scratch_net/biwidl307/sonia/data_original/VS/subjects/mat/mpBUS045_L1_large.mat','L','L_fact','Linv','maskFixed','S','U','V');

num = '02';
a_2 = load(['/scratch_net/biwidl307/sonia/data_original/VS/FA_subjects/mpBUS0',num,'/0',num,'_002/output_sos.mat']);
a_3 = load(['/scratch_net/biwidl307/sonia/data_original/VS/FA_subjects/mpBUS0',num,'/0',num,'_003/output_sos.mat']);
a_4 = load(['/scratch_net/biwidl307/sonia/data_original/VS/FA_subjects/mpBUS0',num,'/0',num,'_004/output_sos.mat']);
a_5 = load(['/scratch_net/biwidl307/sonia/data_original/VS/FA_subjects/mpBUS0',num,'/0',num,'_005/output_sos.mat']);

bf_im = cat(4,a_2.BF,a_3.BF,a_4.BF,a_5.BF);
try 
    RF = cat(4,a_2.RF,a_3.RF,a_4.RF,a_5.RF);
catch
    RF =a_2.RF;
end
recon_lbfgs = cat(3,a_2.recon_lbfgs, a_3.recon_lbfgs, a_4.recon_lbfgs, a_5.recon_lbfgs);
measmnts = cat(2,a_2.measmnts, a_3.measmnts, a_4.measmnts, a_5.measmnts);
CorrCoeff = cat(2,a_2.CorrCoeff, a_3.CorrCoeff, a_4.CorrCoeff, a_5.CorrCoeff);
opts = a_2.opts;

save(['/scratch_net/biwidl307/sonia/data_original/VS/subjects/mat_sos/mpBUS0',num,'_L1.mat'], 'L','L_fact','maskFixed','measmnts')
save(['/scratch_net/biwidl307/sonia/data_original/VS/subjects/mat_sos/mpBUS0',num,'_L1_large.mat'],'L','L_fact','maskFixed','measmnts','CorrCoeff','RF','bf_im','opts','recon_lbfgs')