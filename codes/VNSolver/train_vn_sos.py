"""
Main file for network training. 
Code adapted from Mélanie Bernhardt
Sonia Laguna - M.Sc. Thesis - ETH Zurich

Takes a config file at console argument.
See readme for usage explanation.
"""

from codes.VNSolver.train_utils import saveBatchMatrices
from codes.VNSolver.vn_net_sos import VNSolver
from codes.data_utils.data_loader import DataGenerator
import tensorflow as tf
import os
import time
import numpy as np
import logging
from codes.utils.config import getConfig


#  ============= PARAMETERS SET UP ========== #
SEED = 706
np.random.seed(SEED)

# Data loading parameters
data_dir = os.getenv('DATA_PATH')
exp_dir = os.getenv("EXP_PATH")
US_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..')
config_dir = os.path.join(US_path, 'configs')
tf.flags.DEFINE_string("config_file", 'config.yaml', "Name of the config file to use for the current exp")
FLAGS = tf.flags.FLAGS

config = getConfig(os.path.join(config_dir, FLAGS.config_file))

msz = [config.msz_x, config.msz_y]

# ============ SAVING FOLDERS SET UP ======== #
checkpoint_dir = os.path.join(exp_dir, config.exp_name)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
mat_dir = os.path.join(checkpoint_dir, 'mat')
if not os.path.exists(mat_dir):
    os.makedirs(mat_dir)

# ============= LOGGER SETUP ============== #
# create logger
global logger
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')
logger = logging.getLogger('my_log')
# create console handler and set level to debug
ch = logging.StreamHandler()
# create formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
# add ch to logger
logger.addHandler(ch)
log_filename = checkpoint_dir + '/logfile' + '.log'
file_handler = logging.FileHandler(log_filename)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)


#  ============= DATA LOADING and VAR DEFINITION ========== #
try:
    if config.restore:
        restore_checkpoint = True
        cpkt_to_restore = 'vn-{}'.format(config.cpkt)
        cpkt_to_restore = os.path.join(checkpoint_dir, cpkt_to_restore)
    else:
        restore_checkpoint = False
except AttributeError:
    restore_checkpoint = False

DataLoader = DataGenerator(config.filename,
                           config.n_val,
                           msz, config.NA,
                           config.noise_rate,
                           config.usm_rate,
                           config.random_mask_type,
                           config.n_batch,
                           config.n_iter,
                           config.data_type,
                           config.use_med_filter,
                           config.use_mean_filter,
                           mix=config.mix,
                           filename_mix=config.filename_mix,
                           mix_type=config.mix_type,
                           mix_triple=config.mix_triple,
                           filename_mix_triple=config.filename_mix_triple,
                           mix_type_triple=config.mix_type_triple,
                           p_mix=config.p_mix,
                           p_triple=config.p_triple)

train_iterator = DataLoader.getBatchIterator()
N_val = DataLoader.img_val.shape[0]
DataLoaderFull = DataGenerator(config.filename_val_fullpipeline,
                               config.n_val,
                               msz, config.NA,
                               config.noise_rate,
                               config.usm_rate,
                               'test',
                               config.n_batch,
                               config.n_iter,
                               config.data_type,
                               config.use_med_filter,
                               config.use_mean_filter)
N_val_full = DataLoaderFull.img_val.shape[0]
del DataLoaderFull.Lnrm
# ==============  START THE EXPERIMENT =========== #
logger.info(tf.test.is_gpu_available())
print(tf.test.is_gpu_available(), 'gpu availability')
logger.info('Running %s experiment ...' % config.cost_type)
logger.info('\n Settings for this expriment are: \n')
for key in config.keys():
    logger.info('  {}: {}'.format(key.upper(), config[key]))
logger.info('Saving checkpoint to {}'.format(checkpoint_dir))

## =========== BUILD THE MODEL ============ #
tf.reset_default_graph()
model = VNSolver(Lmat=DataLoader.Lnrm,
                 aa=DataLoader.aa,
                 mF=DataLoader.fixed_mask,
                 s1=DataLoader.s1,
                 n_layers=config.n_layers,
                 n_filters_vn=config.n_filters_vn,
                 filter_sz_vn=config.filter_sz_vn,
                 n_batch=config.n_batch,
                 n_reads=DataLoader.n_reads,
                 img_sz=DataLoader.img_sz,
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
                 D_standardize=config.D_standardize,
                 use_preconditioner=config.use_preconditioner,
                 use_spatial_filter_weighting=config.use_spatial_filter_weighting,
                 momentum_term=config.momentum,
                 init_type=config.init_type,
                 c_init=config.c_init,
                 weight_us=config.weight_us,
                 adaptive_interpolator=config.readjust_rng,
                 use_temperature=config.use_temperature,
                 reg_reg_act_func=config.use_reg_activation_reg,
                 reg_data_act_func=config.use_reg_activation_data,
                 share_weights=config.share_weights,
                 KL = config.KL,
                 alpha_KL = config.alpha_KL,
                 beta_KL = config.beta_KL,
                 aleat = config.aleat,
                 drop = config.drop,
                 rate_K = config.rate)

# ============= GET THE NODES NEEEDED IN THE GRAPH =========== #
params = model.get_all_model_params()
# placeholders
din = params['placeholders']['d_in']
dinpaint = params['placeholders']['d_inpaint']
usm = params['placeholders']['usm']
gt_image = params['placeholders']['gt_image']
weighting_temperature = params['placeholders']['w_temp']
# nodes references
xgt_nrm = params['xgt_nrm']
recon_sos = params['recon_sos']
threshold_ops = params['threshold_ops']
netcost = params['net_cost']
KL_cost = params['KL_cost']
pre_netcost = params['pre_net_cost']
learning_rate = tf.placeholder(tf.float32)
SADtrain = params['SAD_train']
raymat = params['L']

logger.info('Start unrolling training ...')
unrolling_vars = tf.get_collection(
    tf.GraphKeys.TRAINABLE_VARIABLES, 'unrolling')
unroll_var_list = [var for var in unrolling_vars]
train_step = tf.train.AdamOptimizer(learning_rate).minimize(
    netcost, var_list=unroll_var_list)
print(unroll_var_list)

# ====== CONFIGURE THE GRAPH AND THE TENSORBOARD SUMMARIES =========== #
trainable_vars = [var for var in tf.get_collection(
    tf.GraphKeys.TRAINABLE_VARIABLES)]
# print(trainable_vars)

if config.freeze:
    last_var_1 = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES,
        'unrolling/layer{}'.format(config.n_layers-1))
    last_var = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES,
        'unrolling/layer{}'.format(config.n_layers))
    unfreeze = np.append(last_var, last_var_1).tolist()
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(
        netcost, var_list=unfreeze)
    print(unfreeze)

interpolation_vars_1 = tf.contrib.framework.get_variables_by_suffix(
    'flt_resp_max')
interpolation_vars_2 = tf.contrib.framework.get_variables_by_suffix(
    'flt_resp_var')
interpolation_vars = np.append(
    interpolation_vars_1, interpolation_vars_2).tolist()
var_list = np.append(trainable_vars, interpolation_vars).tolist()
print(var_list)
# print(var_list)
saver = tf.train.Saver(max_to_keep=10, var_list=var_list)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
config_sess = tf.ConfigProto(
    log_device_placement=False, gpu_options=gpu_options)
config_sess.gpu_options.allow_growth = True
val_loss = tf.placeholder(dtype=tf.float32, shape=())
val_sos_loss = tf.placeholder(dtype=tf.float32, shape=())
val_loss_full = tf.placeholder(dtype=tf.float32, shape=())
val_sos_loss_full = tf.placeholder(dtype=tf.float32, shape=())
loss_summary = tf.summary.scalar("{}_loss".format(config.cost_type), netcost)
sad_summary_train = tf.summary.scalar("MAE_loss_train", SADtrain)
# Train summaries for tensorboard
if config.cost_type == 'recon_iter':
    train_summary_op = tf.summary.merge([loss_summary, sad_summary_train])
else:
    train_summary_op = tf.summary.merge([loss_summary])
train_summary_dir = os.path.join(checkpoint_dir, "summaries", "train")

# Validation summaries for tensorboard
val_summary_dir = os.path.join(checkpoint_dir, "summaries", "val")
val_recon_loss_summary = tf.summary.scalar(
    'validation_reconstruction_loss', val_loss)
val_sos_loss_summary = tf.summary.scalar('validation_sos_loss', val_sos_loss)
val_recon_loss_summary_full = tf.summary.scalar(
    'validation_reconstruction_loss_full', val_loss_full)
val_sos_loss_summary_full = tf.summary.scalar(
    'validation_sos_loss_full', val_sos_loss_full)
val_summary_writer = tf.summary.FileWriter(val_summary_dir)
val_summary_op = tf.summary.merge(
    [val_recon_loss_summary,
     val_sos_loss_summary,
     val_recon_loss_summary_full,
     val_sos_loss_summary_full])

# Prepare variables for error saving
train_err_recon_list = []
val_err_recon_list = []
val_err_recon_sos_list = []

cur_w = 0.0
lr = float(config.lr_init)
assert type(lr) == float
minmax_ops = model.get_interp_minmax_ops()
del DataLoader.Lnrm

with tf.Session(config=config_sess) as sess:
    train_summary_writer = tf.summary.FileWriter(train_summary_dir)
    sess.run(tf.global_variables_initializer())
    if restore_checkpoint:
        saver.restore(sess, cpkt_to_restore)
        logger.info('Model restored from {}'.format(config.cpkt))
        start_iter = config.cpkt + 1
        for i in range(1, start_iter+1):
            cur_w += 1e-3
            if config.decrease_lr:
                if (i + 1) % 80000 == 0:
                    lr *= 0.1
                elif (i + 1) % 40000 == 0:
                    lr *= 0.1
    else:
        start_iter = 1
    logger.info(config.cost_type)
    start_time = time.time()
    first_val = True
    for i_iter in range(start_iter, config.n_iter+1):
        cur_w += 1e-3
        # One training step
        batch_d, batch_dinpaint, batch_mask, batch_img = next(train_iterator)
        feed_train = {din: batch_d, usm: batch_mask,
                      gt_image: batch_img, dinpaint: batch_dinpaint,
                      learning_rate: lr,
                      weighting_temperature: cur_w}  # raymat: rayValue
        _, loss, summaries, _, KL_loss, preloss = sess.run(
            [train_step, netcost, train_summary_op, minmax_ops, KL_cost, pre_netcost],
            feed_dict=feed_train)
        train_summary_writer.add_summary(summaries, i_iter)
        # Decrease the learning rate over training
        if config.decrease_lr:
            if (i_iter + 1) % 80000 == 0:
                lr *= 0.1
            elif (i_iter + 1) % 40000 == 0:
                lr *= 0.1
        train_err_recon_list.append(loss)
        if i_iter == start_iter or i_iter % config.print_interv == 0:
            logger.info("Step %d, unrolling training loss: %.5f " %
                        (i_iter, loss))
        if i_iter > 50 and config.readjust_rng:
            if (i_iter % 20 == 0 and i_iter < 1500) or ((i_iter-1) % 1100 == 0 and i_iter != 0):
                model.readjust_interp_resp(sess)
                logger.info("{} readjust_response_range".format(i_iter))

        # Threshold operation
        sess.run(threshold_ops)

        # Save batch information for debugging
        if config.save_matrices and i_iter % config.save_matrices_interv == 0:
            print(feed_train.keys())
            saveBatchMatrices(model, feed_train, batch_d, batch_dinpaint, batch_img,
                              sess, i_iter, mat_dir, True)
        # Save model
        if i_iter % config.save_model_interv == 0:
            ckpt_name = 'vn-' + str(i_iter)
            saver.save(sess, os.path.join(checkpoint_dir,
                                          ckpt_name), write_meta_graph=False)

        # Validation step
        if i_iter % config.val_interv == 0:
            val_iterator = DataLoader.getBatchIterator(val=True)
            if first_val:
                val_err_recon, val_err_recon_sos = model.validate(
                    sess=sess,
                    validation_iterator=val_iterator,
                    N_val=N_val,
                    mat_dir=mat_dir,
                    i_iter=i_iter,
                    use_med=config.use_med_filter,
                    use_mean=config.use_mean_filter,
                    save=config.save_matrices_val,
                    logger=logger)
            else:
                val_err_recon, val_err_recon_sos = model.validate(
                    sess=sess,
                    validation_iterator=val_iterator,
                    N_val=N_val,
                    mat_dir=mat_dir,
                    i_iter=i_iter,
                    use_med=config.use_med_filter,
                    use_mean=config.use_mean_filter,
                    save=config.save_matrices_val,
                    logger=None)
            val_err_recon_list.append(val_err_recon)
            val_err_recon_sos_list.append(val_err_recon_sos)
            val_iterator_full = DataLoaderFull.getBatchIterator(val=True)
            if first_val:
                val_err_recon_full, val_err_recon_sos_full = model.validate(
                    sess=sess,
                    validation_iterator=val_iterator_full,
                    N_val=N_val_full,
                    mat_dir=mat_dir,
                    i_iter=i_iter,
                    use_med=config.use_med_filter,
                    use_mean=config.use_mean_filter,
                    save=config.save_matrices_val,
                    logger=logger,
                    name='full')
                first_val = False
            else:
                val_err_recon_full, val_err_recon_sos_full = model.validate(
                    sess=sess,
                    validation_iterator=val_iterator_full,
                    N_val=N_val_full,
                    mat_dir=mat_dir,
                    i_iter=i_iter,
                    use_med=config.use_med_filter,
                    use_mean=config.use_mean_filter,
                    save=config.save_matrices_val,
                    logger=None,
                    name='full')
            summary = sess.run(
                val_summary_op,
                {val_loss_full: val_err_recon_full,
                 val_sos_loss_full: val_err_recon_sos_full,
                 val_loss: val_err_recon,
                 val_sos_loss: val_err_recon_sos})
            val_summary_writer.add_summary(summary, i_iter)
            logger.info("Step %d, validation reconstruction error RAY: %.5f " %
                        (i_iter, val_err_recon))
            logger.info("Step %d, validation sos reconstruction error RAY: %.5f " %
                        (i_iter, val_err_recon_sos))
            logger.info("Step %d, validation reconstruction error FULL: %.5f " %
                        (i_iter, val_err_recon_full))
            logger.info("Step %d, validation sos reconstruction error FULL: %.5f " %
                        (i_iter, val_err_recon_sos_full))
