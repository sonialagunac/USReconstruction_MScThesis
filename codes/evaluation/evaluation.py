"""
Sonia Laguna - M.Sc. Thesis - ETH Zurich
Code adapted from Mélanie Bernhardt
Main file to run evaluation of the network.
Reads csv file stated in the config file
Saves to {checkpoint dir}/eval-{name of model}/{filename}:
    - reconstruction (incl. interdemediate layers)
    - ground truth
    - measurements
    - parameters of the network
Used to construct the matrices needed by mat files 
for further processing and visualization.
See Readme.md for more instructions.
"""
from codes.VNSolver.vn_net_sos import VNSolver
from codes.data_utils.data_loader import DataGenerator
import tensorflow as tf
import os
from codes.utils.config import getConfig
import numpy as np
import scipy.io as sio
import pandas as pd
import time
import hdf5storage

t = time.time() # Computing the time cost of each inference pass

# Data loading parameters
data_dir = os.getenv('DATA_PATH')
exp_dir = os.getenv("EXP_PATH")
US_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                       '..', '..')
config_dir = os.path.join(US_path, 'configs')
tf.flags.DEFINE_string("config_file", 'config.yaml',
                       "Name of the config file to use for the current exp")
tf.flags.DEFINE_string("cpkt", None,
                       "Number of the checkpoint to restore")
FLAGS = tf.flags.FLAGS

config = getConfig(os.path.join(config_dir, FLAGS.config_file))
config.cpkt_to_restore = FLAGS.cpkt

msz = [config.msz_x, config.msz_y]

checkpoint_dir = os.path.join(exp_dir, config.exp_name)
if config.init_type == 'constant':
    eval_dir = os.path.join(
        checkpoint_dir, 'eval-{}-{}'.format(FLAGS.cpkt, config.c_init))
else:
    #Modify 'eval-' if different saving name is desired
    eval_dir = os.path.join(checkpoint_dir, 'eval-{}'.format(FLAGS.cpkt))
if not os.path.exists(eval_dir):
    os.makedirs(eval_dir)
#  ============= DATA LOADING and VAR DEFINITION ==========
try:
    if config.restore:
        restore_checkpoint = True
        cpkt_to_restore = 'vn-{}'.format(config.cpkt)
        cpkt_to_restore = os.path.join(checkpoint_dir, cpkt_to_restore)
    else:
        restore_checkpoint = False
except AttributeError:
    restore_checkpoint = False

print(config.filename)

df = pd.read_csv(os.path.join(os.path.dirname(
    os.path.realpath(__file__)), 'csv', config.csv))
files = df['filename'].values
n_test = df['n_test'].values
usr_rate = df['usr_rate'].values
noise_rate = df['noise_rate'].values
type_mask = df['type_mask'].values
data_type = df['data_type'].values
eval_type = df['eval_type'].values

t_load = time.time()
#Loading the test data is fastest to get main config
DataLoader = DataGenerator(files[0],
                           n_test[0],
                           msz, config.NA,
                           noise_rate[0],
                           usr_rate[0],
                           type_mask[0],
                           config.n_batch,
                           config.n_iter,
                           data_type[0],
                           config.use_med_filter,
                           config.use_mean_filter,
                           type=eval_type[0],
                           imz=[64, 84],
                           fixed_rate=True)
print('DataLoader time: ', time.time()-t_load)
Lnrm = DataLoader.Lnrm
aa = DataLoader.aa
mF = DataLoader.fixed_mask
s1 = DataLoader.s1
n_reads = DataLoader.n_reads
img_sz = DataLoader.img_sz
print(img_sz)
del DataLoader

print(tf.test.is_gpu_available(), 'gpu availability')
## =========== BUILD THE MODEL ============ #
tf.reset_default_graph()

model = VNSolver(Lmat=Lnrm,
                 aa=aa,
                 mF=mF,
                 s1=s1,
                 n_layers=config.n_layers,
                 n_filters_vn=config.n_filters_vn,
                 filter_sz_vn=config.filter_sz_vn,
                 n_batch=config.n_batch,
                 n_reads=n_reads,
                 img_sz=img_sz,
                 measurement_sz=msz,
                 n_angle_pairs=config.NA,
                 cost_type=config.cost_type,
                 data_term_type=config.data_term_type,
                 n_interp_knots_data=config.n_interp_knots_data,
                 n_interp_knots_reg=config.n_interp_knots_reg,
                 minx_data=config.minx_data,
                 maxx_data=config.maxx_data,
                 minx_reg=config.minx_reg,
                 maxx_reg=config.maxx_reg,
                 use_preconditioner=config.use_preconditioner,
                 use_spatial_filter_weighting=config.use_spatial_filter_weighting,
                 momentum_term=config.momentum,
                 init_type=config.init_type,
                 c_init=config.c_init,
                 weight_us=config.weight_us,
                 D_standardize=config.D_standardize,
                 adaptive_interpolator=config.readjust_rng,
                 use_temperature=config.use_temperature,
                 reg_reg_act_func=config.use_reg_activation_reg,
                 reg_data_act_func=config.use_reg_activation_data,
                 share_weights=config.share_weights,
                 KL = config.KL,
                 alpha_KL=config.alpha_KL,
                 beta_KL=config.beta_KL,
                 aleat = config.aleat,
                 drop = config.drop,
                 rate_K = config.rate)
params = model.get_all_model_params()
batch_size = config.n_batch
# ============= GET THE NODES NEEEDED IN THE GRAPH =========== #
trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
interpolation_vars_1 = tf.contrib.framework.get_variables_by_suffix(
    'flt_resp_max')
interpolation_vars_2 = tf.contrib.framework.get_variables_by_suffix(
    'flt_resp_var')
interpolation_vars = np.append(
    interpolation_vars_1, interpolation_vars_2).tolist()
var_list = np.append(trainable_vars, interpolation_vars).tolist()

# ====== RESTORE THE GRAPH =========== #
saver = tf.train.Saver(var_list=var_list)
config_sess = tf.ConfigProto(log_device_placement=False)
config_sess.gpu_options.allow_growth = True
ckpt_to_restore = os.path.join(checkpoint_dir, config.cpkt_to_restore)

n_obs = len(df)
print('N obs: ',n_obs)
sess = tf.Session(config=config_sess)
sess.run(tf.global_variables_initializer())
saver.restore(sess, ckpt_to_restore)

for i in range(n_obs):
    print('Evaluating file ' + str(i))
    print(files[i])
    print(n_test[i])
    print(noise_rate[i])
    print(data_type[i])
    print(eval_type[i])
    if os.path.exists(os.path.join(eval_dir, 'test-{}-{}-{}-{}-{}'
                                   .format(data_type[i],
                                           noise_rate[i],
                                           usr_rate[i],
                                           type_mask[i],
                                           os.path.basename(files[i])))):
        print('Already exists: ', os.path.join(eval_dir, 'test-{}-{}-{}-{}-{}'
                                   .format(data_type[i],
                                           noise_rate[i],
                                           usr_rate[i],
                                           type_mask[i],
                                           os.path.basename(files[i]))))
    else:
        if ((batch_size > n_test[i]) or (n_test[i] % batch_size > 0 and n_test[i] < 65)
                or batch_size < 16):
            print('have to reload model with a smaller batch size in')
            t_model = time.time()
            sess.close()
            tf.reset_default_graph()
            batch_size = n_test[i]
            print('new batch size {}'.format(batch_size))
            model = VNSolver(Lmat=Lnrm,
                             aa=aa,
                             mF=mF,
                             s1=s1,
                             n_layers=config.n_layers,
                             n_filters_vn=config.n_filters_vn,
                             filter_sz_vn=config.filter_sz_vn,
                             n_batch=batch_size,
                             n_reads=n_reads,
                             img_sz=img_sz,
                             measurement_sz=msz,
                             n_angle_pairs=config.NA,
                             cost_type=config.cost_type,
                             data_term_type=config.data_term_type,
                             n_interp_knots_data=config.n_interp_knots_data,
                             n_interp_knots_reg=config.n_interp_knots_reg,
                             minx_data=config.minx_data,
                             maxx_data=config.maxx_data,
                             minx_reg=config.minx_reg,
                             maxx_reg=config.maxx_reg,
                             use_preconditioner=config.use_preconditioner,
                             use_spatial_filter_weighting=config.use_spatial_filter_weighting,
                             momentum_term=config.momentum,
                             init_type=config.init_type,
                             c_init=config.c_init,
                             weight_us=config.weight_us,
                             D_standardize=config.D_standardize,
                             adaptive_interpolator=config.readjust_rng,
                             use_temperature=config.use_temperature,
                             reg_reg_act_func=config.use_reg_activation_reg,
                             reg_data_act_func=config.use_reg_activation_data,
                             share_weights=config.share_weights,
                             KL=config.KL,
                             alpha_KL=config.alpha_KL,
                             beta_KL=config.beta_KL,
                             aleat=config.aleat,
                             drop = config.drop,
                             rate_K = config.rate)
            params = model.get_all_model_params()
            # ============= GET THE NODES NEEEDED IN THE GRAPH =========== #
            trainable_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES)
            interpolation_vars_1 = tf.contrib.framework.get_variables_by_suffix(
                'flt_resp_max')
            interpolation_vars_2 = tf.contrib.framework.get_variables_by_suffix(
                'flt_resp_var')
            interpolation_vars = np.append(
                interpolation_vars_1, interpolation_vars_2).tolist()
            var_list = np.append(trainable_vars, interpolation_vars).tolist()
            # ====== RESTORE THE GRAPH =========== #
            saver = tf.train.Saver(var_list=var_list)
            sess = tf.Session(config=config_sess)
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, ckpt_to_restore)
            print('Time it takes to reload model: ', time.time() - t_model)
        t_dload = time.time()
        DataLoader = DataGenerator(files[i],
                                   n_test[i],
                                   msz, config.NA,
                                   noise_rate[i],
                                   usr_rate[i],
                                   type_mask[i],
                                   batch_size,
                                   config.n_iter,
                                   data_type[i],
                                   config.use_med_filter,
                                   config.use_mean_filter,
                                   type=eval_type[i],
                                   imz=[64, 84],
                                   fixed_rate=True)
        print('data loader time real: ', time.time() - t_dload)
        print('dataLoader batch size {}'.format(DataLoader.n_batch))
        for number in range(config.n_samp):
            t_tot = time.time()
            done = False
            first = True
            # Evaluate
            print('starting evaluation')
            if eval_type[i] == 'val':
                validation_iterator = DataLoader.getBatchIterator(val=True)
                print(DataLoader.n_batch)
                print(type(validation_iterator))
                RMSE = []
                print('batch size {}'.format(batch_size))
                count = 0
                while not done:
                    try:
                        din, dinpaint, dmask, imgs = next(validation_iterator)
                        #Uncomment section below if specific usm is desired
                        # if float(usr_rate[i]) > 0:
                        #     if first:
                        #         #Directory of the desired mask
                        #         dir_mask ='/scratch_net/biwidl307/sonia/USImageReconstruction-Melanie/runs/4_ICFP_reg1e5_tau5_32filt_15lay_VS/eval-vn-120000/'
                        #         print('Loading previous undersampling mask from: ' + dir_mask)
                        #         if float(usr_rate[i]) == 0.1:
                        #             file_mask = 'test-syn-0.0-0.1-patchy-train_VS_15comb_IC_30.mat'
                        #         elif float(usr_rate[i]) == 0.5:
                        #             file_mask = 'test-syn-0.0-0.5-patchy-train_VS_15comb_IC_30.mat'
                        #          elif float(usr_rate[i]) == 0.9:
                        #             file_mask = 'test-syn-0.0-0.9-patchy-testset_ideal_MS_32_imgs.mat'
                        #         dmask_file = hdf5storage.loadmat(os.path.join(dir_mask, file_mask))
                        #     # dmask = dmask_file['usm'][:, count * 15:count * 15 + 15]
                        #     dmask = dmask_file['usm']
                        feed_dict = {params['placeholders']['d_in']: din,
                                     params['placeholders']['usm']: dmask,
                                     params['placeholders']['gt_image']: imgs,
                                     params['placeholders']['d_inpaint']: dinpaint}
                        if first:
                            print('first')
                            t1 = time.time()
                            val = model.eval_batch_n_export(sess, feed_dict)
                            t2 = time.time()
                            print('Elapsed time first batch {}'.format(t2 - t1))
                            first = False
                            recon = val['recon']
                            current_recon = recon
                            x_init = val['init_img']
                            xs = val['xs']
                            layer_sos = val['layer_sos']
                            xgt_nrm = val['xgt_nrm']
                            k = val['k']
                            std = val['std']
                            din_whiten = val['din_whiten']
                            usm = val['usm']
                            gt_slowness = imgs
                            d_in = din
                            if config.aleat:
                                ua = val['ua']
                                ua_sos = val['ua_sos']
                            print('the shape of xs {}'.format(len(xs)))
                        else:
                            print('else')
                            t1 = time.time()
                            if config.aleat:
                                current_xs, current_layer_sos, current_din_whiten, current_k, current_std, current_xgt_nrm, ua_current, ua_sos_current = sess.run(
                                    [params['xs'], params['layer_sos'], params['din_whiten'], params['k'], params['std'], params['xgt_nrm'], params['ua'], params['ua_sos']], feed_dict=feed_dict)
                            else:
                                current_xs, current_layer_sos, current_din_whiten, current_k, current_std, current_xgt_nrm = sess.run(
                                    [params['xs'], params['layer_sos'], params['din_whiten'], params['k'], params['std'], params['xgt_nrm']], feed_dict=feed_dict)

                            t2 = time.time()
                            print('Elapsed time reconstruction {}'.format(t2-t1))
                            d_in = np.concatenate((d_in, din), axis=1)
                            print('shape of din {}'.format(d_in.shape))
                            current_recon = current_layer_sos[-1]
                            print(np.asarray(current_xs).shape)
                            print(np.asarray(xs).shape)
                            xs = np.concatenate((xs, current_xs), axis=1)
                            print('shape of xs {}'.format(xs.shape))
                            recon = np.concatenate((recon, current_recon))
                            print('shape of recon {}'.format(recon.shape))
                            x_init = np.concatenate((x_init, current_xs[0]))
                            print('shape of xinit {}'.format(x_init.shape))
                            layer_sos = np.concatenate(
                                (layer_sos, current_layer_sos), axis=1)
                            gt_slowness = np.concatenate((gt_slowness, imgs))
                            print('shape of layer sos {}'.format(layer_sos.shape))
                            din_whiten = np.concatenate(
                                (din_whiten, current_din_whiten), axis=1)
                            print('shape of d whiten sos {}'.format(
                                din_whiten.shape))
                            k = np.concatenate((k, current_k))
                            print('shape of k {}'.format(k.shape))
                            std = np.concatenate((std, current_std))
                            print('shape of std {}'.format(std.shape))
                            usm = np.concatenate((usm, dmask), axis=1)
                            print('shape of usm {}'.format(usm.shape))
                            xgt_nrm = np.concatenate(
                                (xgt_nrm, current_xgt_nrm))
                            if config.aleat:
                                ua = np.concatenate((ua, ua_current))
                                ua_sos = np.concatenate((ua_sos, ua_sos_current))
                            print('shape of xgt_nrm {}'.format(xgt_nrm.shape))
                            print('shape of gt_slowness {}'.format(gt_slowness.shape))
                        sos = 1/imgs
                        print(np.max(recon))
                        tmpRMSE = np.sqrt(
                            np.mean(np.square(current_recon-sos), axis=(1, 2)))
                        RMSE = np.append(RMSE, tmpRMSE)
                        count += 1
                    except StopIteration:
                        print('RMSE', np.mean(RMSE))
                        done = True
                        val['recon'] = recon
                        val['k'] = k
                        val['std'] = std
                        val['usm'] = usm
                        val['xgt_nrm'] = xgt_nrm
                        val['layer_sos'] = layer_sos
                        val['din_whiten'] = din_whiten
                        val['xs'] = xs
                        val['init_img'] = x_init
                        val['gt_slowness'] = gt_slowness
                        val['din'] = d_in
                        if config.aleat:
                            val['ua'] = ua
                            val['ua_sos'] = ua_sos
                        print('shape of din {}'.format(d_in.shape))
                        if not config.drop and not config.KL:
                            #Saving the whole set of info
                            print('Doing regular VN')
                            sio.savemat(os.path.join(eval_dir, 'test-{}-{}-{}-{}-{}'
                                                     .format(data_type[i],
                                                             noise_rate[i],
                                                             usr_rate[i],
                                                             type_mask[i],
                                                             os.path.basename(files[i]))), val)
                        else:
                            print('Doing dropout or bayesian inference')
                            #Saving only reconstruction and ground truth
                            if config.aleat:
                                val_short = {'gt_slowness' : gt_slowness, 'recon': recon, 'ua' : ua, 'ua_sos': ua_sos}
                            else:
                                val_short = {'gt_slowness': gt_slowness, 'recon': recon, 'din_whiten': din_whiten, 'k': k, 'std': std}

                            if number != 0:
                                print(number, 'Sampling number')
                                recon_short_stack = np.concatenate((recon_short_stack, recon[None, ...]))
                                gt_short_stack = np.concatenate((gt_short_stack, gt_slowness[None, ...]))
                            else:
                                recon_short_stack = recon[None, ...]
                                gt_short_stack = gt_slowness[None, ...]
            elif eval_type[i] == 'test':
                validation_iterator = DataLoader.getBatchIterator()
                while not done:
                    try:
                        din, dinpaint, dmask = next(validation_iterator)
                        feed_dict = {params['placeholders']['d_in']: din,
                                     params['placeholders']['usm']: dmask,
                                     params['placeholders']['d_inpaint']: dinpaint}
                        if first:
                            print('First')
                            t1 = time.time()
                            val = model.eval_batch_n_export(
                                sess, feed_dict, test=True)
                            t2 = time.time()
                            print('Elapsed time first batch {}'.format(t2-t1))
                            first = False
                            recon = val['recon']
                            x_init = val['init_img']
                            xs = val['xs']
                            layer_sos = val['layer_sos']
                            k = val['k']
                            std = val['std']
                            din_whiten = val['din_whiten']
                            usm = val['usm']
                            d_in = din
                            if config.aleat:
                                ua = val['ua']
                                ua_sos = val['ua_sos']
                        else:
                            t1 = time.time()
                            if config.aleat:
                                current_xs, current_layer_sos, current_din_whiten, current_k, current_std , ua_current, ua_sos_current= sess.run(
                                    [params['xs'], params['layer_sos'], params['din_whiten'], params['k'], params['std'], params['ua'], params['ua_sos']], feed_dict=feed_dict)
                            else:
                                current_xs, current_layer_sos, current_din_whiten, current_k, current_std = sess.run(
                                    [params['xs'], params['layer_sos'], params['din_whiten'], params['k'], params['std'],
                                     ], feed_dict=feed_dict)
                            t2 = time.time()
                            print('Elapsed time reconstruction {}'.format(t2-t1))
                            d_in = np.concatenate((d_in, din), axis=1)
                            print('shape of din {}'.format(d_in.shape))
                            current_recon = current_layer_sos[-1]
                            print(np.asarray(current_xs).shape)
                            print(np.asarray(xs).shape)
                            xs = np.concatenate((xs, current_xs), axis=1)
                            print('shape of xs {}'.format(xs.shape))
                            recon = np.concatenate((recon, current_recon))
                            print('shape of recon {}'.format(recon.shape))
                            x_init = np.concatenate((x_init, current_xs[0]))
                            print('shape of xinit {}'.format(x_init.shape))
                            layer_sos = np.concatenate(
                                (layer_sos, current_layer_sos), axis=1)
                            print('shape of layer sos {}'.format(layer_sos.shape))
                            din_whiten = np.concatenate(
                                (din_whiten, current_din_whiten), axis=1)
                            print('shape of d whiten sos {}'.format(
                                din_whiten.shape))
                            k = np.concatenate((k, current_k))
                            print('shape of k {}'.format(k.shape))
                            std = np.concatenate((std, current_std))
                            print('shape of std {}'.format(std.shape))
                            usm = np.concatenate((usm, dmask), axis=1)
                            print('shape of usm {}'.format(usm.shape))
                            if config.aleat:
                                ua = np.concatenate((ua, ua_current))
                                ua_sos = np.concatenate((ua_sos, ua_sos_current))
                    except StopIteration:
                        done = True
                        val['recon'] = recon
                        val['k'] = k
                        val['std'] = std
                        val['usm'] = usm
                        val['layer_sos'] = layer_sos
                        val['din_whiten'] = din_whiten
                        val['xs'] = xs
                        val['init_img'] = x_init
                        val['din'] = d_in
                        if config.aleat:
                            val['ua'] = ua
                            val['ua_sos'] = ua_sos
                        if not config.drop and not config.KL:
                            print('Doing regular VN')
                            # Saving all the details
                            sio.savemat(os.path.join(eval_dir, 'test-{}-{}-{}-{}'
                                                     .format(data_type[i],
                                                             noise_rate[i],
                                                             type_mask[i],
                                                             os.path.basename(files[i]))), val)
                        else:
                            print('Doing dropout or bayesian samples')
                            #Saving only reconstruction and standarization parameters
                            if config.aleat:
                                val_short = {'recon': recon, 'ua': ua, 'ua': ua_sos}
                            else:
                                val_short = {'recon': recon, 'din_whiten': din_whiten, 'k': k,'std': std}
                            if number != 0:
                                recon_short_stack = np.concatenate((recon_short_stack, recon[None, ...]))
                                print('Sampling Number: ', number)
                            else:
                                recon_short_stack =  recon[None, ...]
            print('One iteration takes: ', time.time()-t_tot)
        if config.drop or config.KL:
            if eval_type[i] == 'test':
                if config.aleat:
                    val_short = {'recon': recon_short_stack, 'ua': ua, 'ua_sos':ua_sos}
                else:
                    val_short = {'recon': recon_short_stack,  'din_whiten': din_whiten, 'k': k, 'std': std,'din':d_in}
                sio.savemat(os.path.join(eval_dir, 'test-{}-{}-{}-{}'
                                                             .format(data_type[i],
                                                                     noise_rate[i],
                                                                     type_mask[i],
                                                                     os.path.basename(files[i]))), val_short)

            elif eval_type[i] == 'val':
                if config.aleat:
                    val_short = {'gt_slowness' : gt_short_stack, 'recon': recon_short_stack, 'ua': ua, 'ua_sos':ua_sos}
                else:
                    val_short = {'gt_slowness': gt_short_stack, 'recon': recon_short_stack, 'din_whiten': din_whiten, 'k': k, 'std': std, 'din':d_in}
                sio.savemat(os.path.join(eval_dir, 'test-{}-{}-{}-{}-{}'
                                                             .format(data_type[i],
                                                                     noise_rate[i],
                                                                     usr_rate[i],
                                                                     type_mask[i],
                                                                     os.path.basename(files[i]))), val_short)

elapsed = time.time() - t
print('Elapsed time overall: ', elapsed)