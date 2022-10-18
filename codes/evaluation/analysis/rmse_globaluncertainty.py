"""
Sonia Laguna - ETH Zurich, MSc Thesis
Combining results of all uncertainty estimation methods
Computing RMSE reconstruction metrics and differential RMSE results
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import hdf5storage
import statistics
import scipy.stats

def residual(L,meas, recon):
    """
    Computes the residual (greatness of fit) of the input data
    """
    RESrecon = []
    meas[np.isnan(meas)] = 0
    for p in range(meas.shape[1]):
        try:
            rec = (orig_data['L_fact'] / recon[p,...] - mat['k'][p,...]) / mat['std'][p,...]
            recon_s = (np.transpose(rec).T).ravel()
        except:
            rec = recon[p,...]
            recon_s = (1/np.transpose(rec).T).ravel()
        RESrecon.append(np.mean(np.abs(np.matmul(L,(recon_s)) - meas[:,p])))
    return RESrecon


def RMSE_duo(img, recon, lbfgs):
    """
    Computes the RMSE of two input frames
    Args:
        img: Ground truth img
        recon: Reconstructed VN
        lbfgs: Reconstructed LBFGS

    Returns: RMSE
    """
    RMSEvn = []
    RMSElbfgs = []
    for p in range(img.shape[0]):
        xtrue = 1 / img[p, ...]
        recon_vn = recon[p, ...]
        recon_l = lbfgs[p, ...]
        RMSEvn.append(np.sqrt(np.mean(np.power(recon_vn - xtrue, 2))))
        RMSElbfgs.append(np.sqrt(np.mean(np.power(np.transpose(recon_l) - xtrue, 2))))
    return RMSEvn, RMSElbfgs


def RMSE_single(img, recon):
    """
    Computes the RMSE of one input frame
    Args:
        img: Ground truth img
        recon: Reconstructed
    Returns: RMSE
    """
    RMSErecon = []
    for p in range(img.shape[0]):
        xtrue = 1 / img[p, ...]
        recon_s = recon[p, ...]
        RMSErecon.append(np.sqrt(np.mean(np.power(recon_s - xtrue, 2))))
    return RMSErecon


if __name__ == "__main__":
    # Dataset we are analyzing
    filename = ['test-syn-0.0-0.0-test-train_VS_15comb_fullpipeline_30.mat']
    # filename = ['test-syn-0.0-0.0-patchy-train_VS_15comb_IC_30.mat']
    # filename = ['test-syn-0.0-0.0-test-fullpipeline_testset_6comb_32_imgs.mat']
    # filename = ['test-syn-0.0-0.1-patchy-testset_ideal_MS_32_imgs.mat']
    # filename = ['test-syn-0.0-0.5-patchy-testset_ideal_MS_32_imgs.mat']
    # filename = ['test-syn-0.0-0.9-patchy-testset_ideal_MS_32_imgs.mat']

    # Original datafile from set under study
    # orig_data = hdf5storage.loadmat('/scratch_net/biwidl307/sonia/data_original/test/fullpipeline_testset_6comb_32_imgs.mat')
    orig_data = hdf5storage.loadmat('/scratch_net/biwidl307/sonia/data_original/VS/train_VS_15comb_fullpipeline_30.mat')

    experiment = [' Train ICFP - Test FP']
    gt = '4_ICFP_reg1e5_tau5_L2_VS'  # Model of the plain VN

    # Uncertainty: Ensembles, load desired combination of models
    titles_drop = ['ensembles']
    docs = ['eval-vn-120000']
    exps = [
            '4_ICFP_reg1e5_tau5_5filt_20lay_VS',
            '4_ICFP_reg1e5_tau5_8filt_25lay_VS',
            '4_ICFP_reg1e5_tau5_8filt_15lay_VS',
            '4_ICFP_reg1e5_tau5_8filt_20lay_VS',
            '4_ICFP_reg1e5_tau5_16filt_25lay_VS',
            '4_ICFP_reg1e5_tau5_16filt_15lay_VS',
            '4_ICFP_reg1e5_tau5_16filt_20lay_VS',
            '4_ICFP_reg1e5_tau5_32filt_25lay_VS',
            '4_ICFP_reg1e5_tau5_32filt_15lay_VS',
            '4_ICFP_reg1e6_tau5_VS',
            '4_ICFP_reg1e5_tau.25_VS',
            '4_ICFP_reg1e5_tau5_VS',
            '4_ICFP_reg1e5_tau5_L2_VS',
            '4_ICFP_reg1e5_tau2.5_L2_VS',
            ]

    for type in range(len(docs)):
        titles = []
        RMSEdrop = []
        uncert = {'mean': [], 'stdev': [], 'samples': []}
        # Loading all ensemble combinations
        for exp in range(len(exps)):
            dir = os.path.join('/scratch_net/biwidl307/sonia/USImageReconstruction-Sonia/runs/', exps[exp], docs[type])
            mat = hdf5storage.loadmat(os.path.join(dir, filename[0]))
            vn_path = os.path.join('/scratch_net/biwidl307/sonia/USImageReconstruction-Sonia/runs/',gt,'eval-vn-120000',
                filename[0])
            vn = hdf5storage.loadmat(vn_path)
            lbfgs_path = '/scratch_net/biwidl307/sonia/USImageReconstruction-Sonia/runs/lbfgs/lbfgs-' + filename[0]
            lbfgs_recon = hdf5storage.loadmat(lbfgs_path)
            if np.size(uncert['samples']) == 0:
                uncert['samples'] = mat['recon'][None,...]
            else:
                uncert['samples'] = np.concatenate((uncert['samples'], mat['recon'][None,...]))
        # Adding LBFGS result to the ensembles if desired
        # uncert['samples'] = np.concatenate((uncert['samples'],np.reshape(lbfgs_recon['recon_lbfgs'], [lbfgs_recon['recon_lbfgs'].shape[0],64,84])[None,...]))

        # Creating the estimated SoS and uncertainty
        uncert['stdev'] = np.std(uncert['samples'], axis=0)
        uncert['mean'] = np.mean(uncert['samples'], axis=0) #Using the mean as estimate
        # mode= scipy.stats.mode(uncert['samples'], axis=0)
        # uncert['mean'] = mode.mode[0] #Using the mode as estimate
        # uncert['mean'] = np.median(uncert['samples'], axis=0) # Using the median as estimate

        # Logging the RMSE
        RMSEdrop.append(RMSE_single(mat['gt_slowness'][...], uncert['mean']))
        # RMSEdrop.append(residual(orig_data['L'], mat['din_whiten'], uncert['mean'])) #If interested on residuals
        titles.append(titles_drop[0])

    # Uncertainty: MC Dropout
    exps = ['4_ICFP_reg1e5_tau5_dropK0.5_L2_VS'] # Can add as many models as desired
    titles_drop = ['K0.5']
    docs = ['eval-vn-120000'] # Can add as many folders as desired
    for type in range(len(docs)):
        for exp in range(len(exps)):
            dir = os.path.join('/scratch_net/biwidl307/sonia/USImageReconstruction-Sonia/runs/', exps[exp], docs[type])
            uncert = {'mean': [], 'stdev': []}
            mat = hdf5storage.loadmat(os.path.join(dir, filename[0]))

            # Creating the estimated SoS and uncertainty
            uncert['stdev'] = np.std(mat['recon'], axis=0)
            uncert['mean'] = np.mean(mat['recon'], axis=0) # Using the mean as estimate
            # mode = scipy.stats.mode(mat['recon'], axis=0)
            # uncert['mean'] = mode.mode[0] #Using the mode as estimate
            # uncert['mean'] = np.median(mat['recon'], axis=0) # Using the median as estimate

            # Logging the RMSE
            RMSEdrop.append(RMSE_single(mat['gt_slowness'][1,...], uncert['mean']))
            # RMSEdrop.append(residual(orig_data['L'], orig_data['measmnts'], uncert['mean'])) #If interested on residuals
            titles.append(titles_drop[exp])


    # Uncertainty: Bayesian Variational Inference
    exps = ['4_ICFP_reg1e5_tau5_KLa-1b-1_L2_VS'] # Can add as many models as desired
    titles_drop = ['a-1b-1']
    docs = ['eval-vn-120000'] # Can add as many folders as desired

    for type in range(len(docs)):
        for exp in range(len(exps)):
            dir = os.path.join('/scratch_net/biwidl307/sonia/USImageReconstruction-Sonia/runs/', exps[exp], docs[type])
            uncert = {'mean': [], 'stdev': []}
            mat = hdf5storage.loadmat(os.path.join(dir, filename[0]))

            # Creating the estimated SoS and uncertainty
            uncert['stdev'] = np.std(mat['recon'], axis=0)
            uncert['mean'] = np.mean(mat['recon'], axis=0)  # Using the mean as estimate
            # mode = scipy.stats.mode(mat['recon'], axis=0)
            # uncert['mean'] = mode.mode[0] #Using the mode as estimate
            # uncert['mean'] = np.median(mat['recon'], axis=0) # Using the median as estimate

            # Logging the RMSE
            RMSEdrop.append(RMSE_single(mat['gt_slowness'][1, ...], uncert['mean']))
            # RMSEdrop.append(residual(orig_data['L'], orig_data['measmnts'], uncert['mean'])) #If interested on residuals
            titles.append(titles_drop[exp])

    # Plotting the final RMSEs
    RMSEvn, RMSElbfgs = RMSE_duo(mat['gt_slowness'][1,...], vn['recon'], lbfgs_recon['recon_lbfgs'])
    # RMSEvn = residual(orig_data['L'], orig_data['measmnts'], vn['recon']) # If interested on residual
    # RMSElbfgs = residual(orig_data['L'], orig_data['measmnts'], lbfgs_recon['recon_lbfgs']) # If interested on residual
    # Paired data
    rest_VN = []
    rest_LBFGS = []
    titles_vn = titles.copy()
    titles_lbfgs = titles.copy()
    for k in range(len(RMSEdrop)):
        rest_VN.append(np.asarray(RMSEdrop[k]) - np.asarray(RMSEvn))
        rest_LBFGS.append(np.asarray(RMSEdrop[k]) - np.asarray(RMSElbfgs))
    rest_VN.append(np.asarray(RMSElbfgs) - np.asarray(RMSEvn))
    rest_LBFGS.append(np.asarray(RMSEvn) - np.asarray(RMSElbfgs))
    titles_vn.append('LBFGS')
    titles_lbfgs.append('VN')

    RMSEdrop.append(RMSEvn)
    RMSEdrop.append(RMSElbfgs)
    titles.append('VN')
    titles.append('LBFGS')

    fig1, ax1 = plt.subplots()
    ax1.set_title('RMSE [m/s]' + experiment[0])
    ax1.boxplot(RMSEdrop)
    ax1.set_xticklabels(titles)
    for i in range(len(RMSEdrop)):
        ax1.text(i+1, 10, np.around(statistics.median(RMSEdrop[i]), decimals =2), rotation = 45,  ha='left', size = 'medium' )
    ax1.tick_params()
    plt.show()

    fig1, ax1 = plt.subplots()
    ax1.set_title('Differential RMSEs VN [m/s]' + experiment[0])
    ax1.boxplot(rest_VN)
    ax1.set_xticklabels(titles_vn)
    for i in range(len(rest_VN)):
        ax1.text(i+1, 1, np.around(statistics.median(rest_VN[i]), decimals =2), rotation = 45,  ha='left', size = 'medium' )
    ax1.tick_params()
    plt.show()

    fig1, ax1 = plt.subplots()
    ax1.set_title('Differential RMSEs LBFGS [m/s]' + experiment[0])
    ax1.boxplot(rest_LBFGS)
    ax1.set_xticklabels(titles_lbfgs)
    for i in range(len(rest_LBFGS)):
        ax1.text(i + 1, 1, np.around(statistics.median(rest_LBFGS[i]), decimals=2), rotation=45, ha='left', size='medium')
    ax1.tick_params()
    plt.show()

pass
