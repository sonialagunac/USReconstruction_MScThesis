"""
Sonia Laguna - ETH Zurich M.Sc. Thesis
Code adapted from Mélanie Bernhardt
Based on code from Valeriy Vishnevksiy

Core file to defining the Variational Network architecture.
Initialization parameters are explained in the readme (see config file)
"""

import numpy as np
import tensorflow as tf
import os
import scipy.io as sio
import sys
from codes.VNSolver.interpolator import FixedInterpolator, AdaptiveInterpolator
import random
# SEED =random.randint(200,700)
SEED = 706
tf.set_random_seed(SEED)


class VNSolver():
    def __init__(self, Lmat, mF, aa,
                 s1,
                 n_layers, n_filters_vn, filter_sz_vn,
                 n_batch, n_reads, img_sz,
                 measurement_sz,
                 n_angle_pairs,
                 data_term_type='L1',
                 cost_type='recon',
                 n_interp_knots_data=35,
                 n_interp_knots_reg=35,
                 minx_data=-1., maxx_data=1.,
                 minx_reg=-1., maxx_reg=1.,
                 use_preconditioner=False,
                 use_spatial_filter_weighting=False,
                 D_standardize=True,
                 momentum_term=0.1,
                 inpaint=True,
                 init_type='Linv',
                 c_init=None,
                 weight_us=True,
                 adaptive_interpolator=False,
                 use_temperature=True,
                 reg_reg_act_func=False,
                 reg_data_act_func=False,
                 lambda_reg=1e5,
                 lambda_reg_data=1e6,
                 share_weights=False,
                 KL=False,
                 alpha_KL=10,
                 beta_KL=1e3,
                 aleat=False,
                 drop=False,
                 rate_K=0.5):
        self.share_weights = share_weights
        self.__D2_potf_reg_list = []
        self.__D2_potf_data_list = []
        self.__KL = KL #Activates Bayesian Variational Inference
        self.__beta_KL = float(beta_KL)
        self.__alpha_KL = float(alpha_KL)
        self.__aleatoric = aleat #Activates aleatoric uncertainty computation
        self.__drop = drop #Activates MC Dropout
        self.__rate = rate_K #Dropout rate if MC Dropout
        self.__regularize_act_func = reg_reg_act_func
        self.__regularize_act_func_data = reg_data_act_func
        self.__lambda = lambda_reg
        self.__lambda_data = lambda_reg_data
        self.__use_temperature = use_temperature
        self.aa = aa
        self.__ray_mat = tf.Variable(Lmat, dtype=tf.float32, trainable=False)
        self.__ada_interp_objects = []
        self.__mF = tf.Variable(mF, dtype=tf.float32, trainable=False)
        self.__ada_interp_minmax_ops = []
        self.ada_interpolator = adaptive_interpolator
        self.__use_tunable_us_rate = weight_us
        self.__n_layers = n_layers
        self.__n_filters_vn = n_filters_vn
        self.__filter_sz_vn = filter_sz_vn
        self.__n_batch = n_batch
        self.__n_reads = n_reads
        self.__img_sz = img_sz
        self.__data_term_type = data_term_type
        self.__n_interp_knots_data = n_interp_knots_data
        self.__n_interp_knots_reg = n_interp_knots_reg
        self.minx_data = tf.Variable(minx_data, dtype=tf.float32, trainable=False)
        self.maxx_data = tf.Variable(maxx_data, dtype=tf.float32, trainable=False)
        self.minx_reg = tf.Variable(minx_reg, dtype=tf.float32, trainable=False)
        self.maxx_reg = tf.Variable(maxx_reg, dtype=tf.float32, trainable=False)
        self.__use_preconditioner = use_preconditioner
        self.__use_spatial_filter_weighting = use_spatial_filter_weighting
        self.__D_standardize = D_standardize
        self.__momentum_term = momentum_term
        self.__msz = measurement_sz
        self.__na = n_angle_pairs
        self.__s1 = s1
        self.__inpaint = inpaint
        self.init_type = init_type
        if c_init is not None:
            self.c_init = c_init
        self.__layers = []  # store layer information
        self.__learnable_vars = []  # store trainable variables of each layer
        self.__xs = []  # bookkeeping reconstructed image at each iteration
        self.__netcost = []  # final network loss
        self.__threshold_ops = []
        self.__eps_cf = 3e-8  # small constant to add to denominator when normalizing conv. filters
        # small constant to add to denominator when computing data term gradient
        self.__eps_dg = 1e-5

        # Input placeholders
        self.__din = tf.placeholder(
            tf.float32,
            shape=[measurement_sz[0]*measurement_sz[1]*n_angle_pairs, n_batch],
            name='undersampled_measurement')
        self.__dinpaint_ori = tf.placeholder(
            tf.float32,
            shape=[measurement_sz[0]*measurement_sz[1]*n_angle_pairs, n_batch],
            name='d_inpainted')
        self.__usm = tf.placeholder(
            tf.float32,
            shape=[measurement_sz[0]*measurement_sz[1]*n_angle_pairs, n_batch],
            name='undersampling_mask')
        self.__img_gt = tf.placeholder(
            tf.float32,
            shape=[n_batch, img_sz[0], img_sz[1]],
            name='gt_image')
        self.weighting_temperature = tf.placeholder(
            tf.float32, shape=[], name='temperature')

        # Normalize undersampled input measurement and gt image
        self.__din_whiten, self.__k, self.__std, self.__k2, self.__k1, self.__k3 = self.__whiten_undersampled_d(
            self.__din, self.__usm)
        print('normalize done')
        self.__xgt_nrm = self.__normalize_gt_image(
            self.__img_gt, self.__s1, self.__k3, self.__std)
        if self.share_weights:
            with tf.variable_scope('unrolling', reuse=tf.AUTO_REUSE):
                self.__unroll_gdm_shared()
        else:
            with tf.variable_scope('unrolling'):
                self.__unroll_gdm()
        self.__construct_threshold_ops()
        # compute final reconstruction in SoS values
        self.__recon_sos = 1 / \
            ((self.__xs[-1] * self.__std + self.__k3) / self.__s1)

        # compute intermediate reconstructions in SoS values
        # (for loss computation and layer visualization)
        self.__layer_sos = []
        for l in range(len(self.__xs)):
            self.__layer_sos.append(
                1 / ((self.__xs[l] * self.__std + self.__k3) / self.__s1))
        self.cost_type = cost_type
        # compute final recon. cost
        self.__ada_interp_minmax_ops = tf.group(self.__ada_interp_minmax_ops)
        if self.__D2_potf_reg_list:
            self.D2_potf_reg = tf.add_n(self.__D2_potf_reg_list)
        else:
            self.D2_potf_reg = tf.constant(0.)
        if self.__D2_potf_data_list:
            self.D2_potf_data = tf.add_n(self.__D2_potf_data_list)
        else:
            self.D2_potf_data = tf.constant(0.)
        self.__set_cost(cost_type)     
        if self.__aleatoric:
            #Converting the aleatoric uncertainty from x to SoS domain
            self.ua_sos =  self.ua * self.__std / \
            ((self.__xs[-1] * self.__std + self.__k3)**2 / self.__s1)

    def __compute_init_x(self, d, usm):
        """
        Note: pass in normalized inpainted d as input
        """
        [_, Nimgs] = d.get_shape().as_list()
        if self.init_type == 'Lt':
            x_init = self.__LTop(d, usm)
        elif self.init_type == 'constant':
            # normlized x_init equivalent to init to self.c_init ms.
            normalize_slowness = (
                self.__s1/self.c_init - self.__k3) / self.__std
            x_init = normalize_slowness * \
                tf.ones([Nimgs, self.__img_sz[0], self.__img_sz[1]],
                        dtype=tf.float32)
        elif self.init_type == 'zero_nrm':
            x_init = tf.zeros(
                [Nimgs, self.__img_sz[0], self.__img_sz[1]], dtype=tf.float32)
        else:
            print('WRONG INIT ARGUMENT')
            sys.exit(1)
        print('x init computed')
        return x_init

    def __whiten_undersampled_d(self, d, mask):
        print('whiten under begin')
        [_, Nimgs] = d.get_shape().as_list()
        aa = tf.tile(self.aa, multiples=[1, Nimgs])  # [Nreads, Nimgs]
        # compute unmasked mean of aa and d
        mean_aa = tf.reduce_sum(
            aa*mask, axis=0, keepdims=True) / tf.reduce_sum(mask, axis=0, keepdims=True)
        mean_a2 = tf.reduce_sum(
            (aa*mask)**2, axis=0, keepdims=True) / tf.reduce_sum(mask, axis=0, keepdims=True)
        mean_d = tf.reduce_sum(d*mask, axis=0, keepdims=True) / \
            tf.reduce_sum(mask, axis=0, keepdims=True)
        k_orig = mean_d / mean_aa  # [1, Nimgs]
        k = k_orig
        k2 = mean_d / mean_a2
        k3 = tf.reduce_sum((d*aa*mask), keepdims=True, axis=0) / \
            tf.reduce_sum((aa*mask)**2, keepdims=True, axis=0)
        print('k shape {}'.format(k_orig.shape))
        print('k2 shape {}'.format(k2.shape))
        print('mean a2 shape {}'.format(mean_a2.shape))
        print('mean_aa shape {}'.format(mean_aa.shape))
        dnrm = d - k3 * aa
        dnrm = dnrm * mask
        # stm = tf.reduce_max(tf.abs(dnrm), axis=0, keepdims=True) * \
        #    self.__whiten_factor  # [1, Nimgs]
        var = tf.reduce_sum(dnrm**2, keep_dims=True, axis=0) / tf.reduce_sum(mask, axis=0, keepdims=True)
        std = tf.sqrt(var)
        dnrm = dnrm / std
        k = tf.reshape(k, [Nimgs, 1, 1])
        k3 = tf.reshape(k3, [Nimgs, 1, 1])
        # stm = tf.reshape(stm, [Nimgs, 1, 1])
        std = tf.reshape(std, [Nimgs, 1, 1])
        print('whiten done')
        return dnrm, k, std, k2, k_orig, k3

    def __normalize_gt_image(self, imgs, s1, k, std):
        Nimgs = imgs.get_shape().as_list()[0]
        imgs_nrm = imgs * s1
        k = tf.reshape(k, [Nimgs, 1, 1])
        imgs_nrm = imgs_nrm - k
        std = tf.reshape(std, [Nimgs, 1, 1])
        imgs_nrm = imgs_nrm / std
        return imgs_nrm

    def __whiten_inpaint_d(self, d_inpaint):
        [_, Nimgs] = d_inpaint.get_shape().as_list()
        print('whiten begining')
        aa = tf.tile(self.aa, multiples=[1, Nimgs])  # [Nreads, Nimgs]
        # compute unmasked mean of aa and d with fixed mask
        # mean_aa = tf.reduce_sum(aa, axis=0, keepdims=True) / \
        #    tf.reduce_sum(self.__mF, axis=0, keepdims=True)
        # mean_d = tf.reduce_sum(d_inpaint*self.__mF, axis=0, keepdims=True) / \
        #    tf.reduce_sum(self.__mF, axis=0, keepdims=True)
        # k = mean_d / mean_aa  # [1, Nimgs]
        k3 = tf.reduce_sum((d_inpaint*aa*self.__mF), keepdims=True, axis=0) / \
            tf.reduce_sum(aa**2, keepdims=True, axis=0)
        d_inpaint_nrm = d_inpaint - k3 * aa
        d_inpaint_nrm = d_inpaint_nrm * self.__mF
        # stm = tf.reduce_max(tf.abs(d_inpaint_nrm), axis=0,
        #                    keepdims=True) * self.__whiten_factor
        var = tf.reduce_sum(d_inpaint_nrm**2, keepdims=True, axis=0) / tf.reduce_sum(self.__mF, axis=0, keepdims=True)
        std = tf.sqrt(var)
        d_inpaint_nrm = d_inpaint_nrm / std
        print('whiten done')
        return d_inpaint_nrm

    def __unroll_gdm(self):
        """
        Unroll GD with momentum

        NOTE: ONLY use inpainted measurement during initialization,
        use undersampled measurement during unrolling
        """
        # alpha0 = tf.get_variable('alpha0', initializer=1., dtype=tf.float32, tf.)
        # compute x0
        # dinpaint_whiten = self.__whiten_inpaint_d(self.__dinpaint_ori)
        x0 = self.__compute_init_x(self.__din_whiten, self.__usm) # use non inpainted meas
        self.__xs.append(x0)
        self.__layers.append({})
        self.__learnable_vars.append({})
        if self.__aleatoric:
            #Computing aleatoric uncertainty, map with 3 layer CNN:
            Kua_1 = tf.get_variable('ua1_filter',initializer=tf.truncated_normal([self.__filter_sz_vn, self.__filter_sz_vn, 1, self.__n_filters_vn], mean = 0.5),dtype=tf.float32)
            Kua_2 = tf.get_variable('ua2_filter',
                                    initializer=tf.truncated_normal([self.__filter_sz_vn, self.__filter_sz_vn, self.__n_filters_vn, self.__n_filters_vn],
                                                                    mean = 0.5), dtype=tf.float32)
            Kua_3 = tf.get_variable('ua3_filter',
                                    initializer=tf.truncated_normal([self.__filter_sz_vn, self.__filter_sz_vn, self.__n_filters_vn, 1],
                                                                    mean = 0.5), dtype=tf.float32)
            x0_res = tf.reshape(x0, [-1, self.__img_sz[0], self.__img_sz[1], 1])
            x0_1 = tf.nn.conv2d(x0_res, Kua_1, strides=[1, 1, 1, 1], padding='SAME', name='ua_1')
            x0_1 = tf.nn.relu(x0_1)
            x0_2 = tf.nn.conv2d(x0_1, Kua_2, strides=[1, 1, 1, 1], padding='SAME', name='ua_2')
            x0_2 = tf.nn.relu(x0_2)
            ua_pre = tf.nn.conv2d(x0_2, Kua_3, strides=[1, 1, 1, 1], padding='SAME', name='ua_3')
            self.ua = tf.nn.relu(ua_pre)
            self.ua = tf.reshape(self.ua, [-1, self.__img_sz[0], self.__img_sz[1]])
            self.ua = self.ua + 3e-8 #Adding small epsilon for stability
            self.__learnable_vars[-1]['ua1_filter'] = Kua_1
            self.__learnable_vars[-1]['ua2_filter'] = Kua_2
            self.__learnable_vars[-1]['ua3_filter'] = Kua_3
        else:
            self.ua = 1
            self.ua_sos = 1
        for i in range(self.__n_layers):
            with tf.variable_scope('layer' + str(i+1)): # variable_scope will rename the variables underneath with thus inside as beggining
                dx = self.__compute_total_grad(
                    self.__ray_mat, self.__xs[i],
                    self.__din_whiten, self.__usm,
                    self.__n_filters_vn, self.__filter_sz_vn)
                if i == 0:
                    s = dx
                    self.__learnable_vars[-1]['momentV'] = tf.get_variable('moment',
                        initializer=0., dtype=tf.float32, trainable=False)
                else:
                    momentV = tf.get_variable('moment',
                        initializer=self.__momentum_term, dtype=tf.float32)
                    s = tf.abs(momentV) * s + dx
                    self.__learnable_vars[-1]['momentV'] = momentV
            self.__xs.append(self.__xs[i] - s)
            self.__layers[-1]['xs'] = self.__xs[-1]
            #self.__learnable_vars[-1]['alpha0'] = alpha0

    def __unroll_gdm_shared(self):
        """
        Unroll GD with momentum shared everything
        """
        # bookkeeping each layer,
        self.__layers.append({})
        self.__learnable_vars.append({})
        # compute x0
        x0 = self.__compute_init_x(self.__din_whiten, self.__usm) # use non inpainted meas
        self.__xs.append(x0)
        if self.__use_tunable_us_rate:
            n_batch = self.__n_batch
            Nus = 35
            yK_usv = tf.ones([Nus, 2])
            interp_us = FixedInterpolator(
                n_batch, 2, 0, 1, Nus, 0, init_yk=yK_usv, scope='interpolation_4')
            us_val = tf.reduce_mean(self.__usm, axis=[0], keepdims=True)
            tmp_for_x = tf.concat(
                    [tf.reshape(us_val, [n_batch, 1]), tf.reshape(us_val, [n_batch, 1])], 1)
            us_weights = interp_us.apply_linear(tmp_for_x)
            us_w1 = tf.reshape(us_weights[:, 0], [n_batch, 1])
            us_w2 = tf.reshape(us_weights[:, 1], [n_batch, 1])
            us_w1 = tf.tile(us_w1, multiples=[
                            1, self.__img_sz[0]*self.__img_sz[1]])
            us_w2 = tf.tile(us_w2, multiples=[
                            1, self.__img_sz[0]*self.__img_sz[1]])
            us_w1 = tf.reshape(
                    us_w1, [n_batch, self.__img_sz[0], self.__img_sz[1]])
            us_w2 = tf.reshape(
                    us_w2, [n_batch, self.__img_sz[0], self.__img_sz[1]])
        for i in range(self.__n_layers):
            with tf.variable_scope('layer' + str(i+1)):
                step_i = tf.get_variable('step', dtype=tf.float32,  initializer=1.)
                momentV = tf.get_variable('moment', dtype=tf.float32, initializer=self.__momentum_term)
             # for data term
            if self.__data_term_type == 'adaptive':
                grad_data = self.__compute_data_grad(self.__ray_mat, self.__xs[i], self.__din_whiten, self.__usm)

            elif self.__data_term_type == 'adaptive_atan':
                grad_data = self.__compute_data_grad(self.__ray_mat, self.__xs[i], self.__din_whiten, self.__usm)
            else:
                alpha_data = tf.Variable(
                    1., dtype=tf.float32, name='data_term_step_size')
                self.__learnable_vars[-1]['alpha_data'] = alpha_data
                grad_data = self.__compute_data_grad(
                    self.__ray_mat, self.__xs[i], self.__din_whiten, self.__usm) * tf.abs(alpha_data)
            grad_reg = self.__compute_reg_grad_full(self.__xs[i], self.__n_filters_vn, self.__filter_sz_vn)
            if self.__use_tunable_us_rate:
                total_grad = us_w1 * grad_data + us_w2 * grad_reg
            else:
                total_grad = grad_data + grad_reg
            self.__layers[-1]['data_grad'] = grad_data
            self.__layers[-1]['reg_grad'] = grad_reg
            self.__layers[-1]['total_grad'] = total_grad
            dx = total_grad
            if i == 0:
                s = dx
                self.__learnable_vars[-1]['momentV'] = tf.get_variable('moment',
                    initializer=0., dtype=tf.float32, trainable=False)
            else:
                s = tf.abs(momentV) * s + step_i * dx
                self.__learnable_vars[-1]['momentV'] = momentV
                self.__learnable_vars[-1]['step'] = step_i
            self.__xs.append(self.__xs[i] - s)
            self.__layers[-1]['xs'] = self.__xs[-1]

    def __compute_total_grad(self, L, x, b, usm, n_filters_vn, filter_sz_vn):
        """
        compute total gradient of objective function, i.e. sum of data term and regularizer gradient
        INPUT: ray matrix L of shape [n_reads, n_pixel]
               image x of shape [n_batch, img_sz[0], img_sz[1]];
               measurement b of shape [n_reads, n_batch];
               n_filters_vn: number of filters; filter_sz_vn: filter size;
        OUTPUT: total gradient of shape [n_batch, img_sz[0], img_sz[1]]
        """
        # bookkeeping each layer
        #Doing it somewhere else now
        # self.__layers.append({})
        # self.__learnable_vars.append({})

        # for data term
        if self.__data_term_type == 'adaptive':
            grad_data = self.__compute_data_grad(L, x, b, usm)

        elif self.__data_term_type == 'adaptive_atan':
            grad_data = self.__compute_data_grad(L, x, b, usm)

        else:
            alpha_data = tf.Variable(
                1., dtype=tf.float32, name='data_term_step_size')
            self.__learnable_vars[-1]['alpha_data'] = alpha_data
            grad_data = self.__compute_data_grad(
                L, x, b, usm) * tf.abs(alpha_data)

        # for regularization term
        grad_reg = self.__compute_reg_grad_full(
            x, n_filters_vn, filter_sz_vn)
        if self.__use_tunable_us_rate:
            n_batch = self.__n_batch
            Nus = 35
            yK_usv = tf.ones([Nus, 2])
            us_val = tf.reduce_mean(usm, axis=[0], keepdims=True)
            tmp_for_x = tf.concat(
                [tf.reshape(us_val, [n_batch, 1]), tf.reshape(us_val, [n_batch, 1])], 1)
            interp_us = FixedInterpolator(
                n_batch, 2, 0, 1, Nus, 0, init_yk=yK_usv, scope='interpolation_4')
            us_weights = interp_us.apply_linear(tmp_for_x)
            us_w1 = tf.reshape(us_weights[:, 0], [n_batch, 1])
            us_w2 = tf.reshape(us_weights[:, 1], [n_batch, 1])
            self.__learnable_vars[-1]['yk_us'] = interp_us.get_knots_variable()
            us_w1 = tf.tile(us_w1, multiples=[
                        1, self.__img_sz[0]*self.__img_sz[1]])
            us_w2 = tf.tile(us_w2, multiples=[
                        1, self.__img_sz[0]*self.__img_sz[1]])
            us_w1 = tf.reshape(
                us_w1, [n_batch, self.__img_sz[0], self.__img_sz[1]])
            us_w2 = tf.reshape(
                us_w2, [n_batch, self.__img_sz[0], self.__img_sz[1]])
            total_grad = us_w1 * grad_data + us_w2 * grad_reg
        else:
            total_grad = grad_data + grad_reg

        self.__layers[-1]['data_grad'] = grad_data
        self.__layers[-1]['reg_grad'] = grad_reg
        self.__layers[-1]['total_grad'] = total_grad
        return total_grad

    def __compute_data_grad(self, L, x, b, usm):
        """
        compute data term gradient
        INPUT: image x of shape [n_batch, img_sz[0], img_sz[1]];
               ray matrix L of shape [n_reads, n_pixel]
               measurement b of shape [n_reads, n_batch]
        OUTPUT: data term gradient of shape [n_batch, img_sz[0], img_sz[1]]
        """
        n_batch = self.__n_batch
        if self.__use_preconditioner:
            read_weights = tf.get_variable('preconditioner',
                                                   initializer=tf.ones([self.__n_reads, 1]),
                                                   dtype=tf.float32)

            self.__learnable_vars[-1]['preconditioner'] = read_weights
            Lxb = read_weights * (self.__Lop(x, usm) - b)
            self.__layers[-1]['Lx'] = self.__Lop(x, usm)
            x2 = tf.reshape(x, [-1, self.__img_sz[0]*self.__img_sz[1]])
            x2 = tf.transpose(x2)  # [n_pixel, n_batch]
            self.__layers[-1]['x_flat'] = x2
            self.__layers[-1]['b'] = b
            self.__layers[-1]['x'] = x
            self.__layers[-1]['before_act_data'] = Lxb
            psiLxb = self.__apply_activation_data(
                tf.reshape(Lxb, [-1, 1]))  # [n_reads, n_batch]
            psiLxb = tf.reshape(psiLxb, [self.__n_reads, n_batch])
            # [n_batch, img_sz[0], img_sz[1]]
            grad_data = self.__LTop(psiLxb*read_weights, usm)
        else:
            Lxb = self.__Lop(x, usm) - b  # [n_reads, n_batch]
            psiLxb = self.__apply_activation_data(Lxb)  # [n_reads, n_batch]
            # [n_batch, img_sz[0], img_sz[1]]
            grad_data = self.__LTop(psiLxb, usm)
        return grad_data

    def __apply_activation_data(self, Lxb):
        """
        Apply activation function in derivative of data term
        """
        if self.__data_term_type == 'L1':
            return tf.sign(Lxb)

        elif self.__data_term_type == 'L1_smooth':
            return Lxb / tf.sqrt(tf.square(Lxb) + self.__eps_dg)

        elif self.__data_term_type == 'L2':
            return Lxb  # L2 norm squared, constant 2 seems not to matter much

        elif self.__data_term_type == 'adaptive':
            n_pix = self.__n_reads * self.__n_batch
            shape_in = tf.shape(Lxb)
            Lxb_reshaped = tf.reshape(Lxb, [-1, 1])
            if self.ada_interpolator:
                interpolator_data = AdaptiveInterpolator(
                    n_pix, 1, self.__n_interp_knots_data, 0.01)
                self.__ada_interp_minmax_ops.append(
                    interpolator_data.register_input(Lxb_reshaped))
                self.__ada_interp_objects.append(interpolator_data)
                self.__learnable_vars[-1]['interp_knots'] = interpolator_data.get_knots_variable()
                self.__learnable_vars[-1]['interp_var'], self.__learnable_vars[-1]['interp_max'] = interpolator_data.get_responce_vars()
                self.__D2_potf_data_list.append(interpolator_data.D2_L1_regularization())
            else:
                interpolator_data = FixedInterpolator(
                    n_pix, 1, self.minx_data, self.maxx_data,
                    self.__n_interp_knots_data, 0.01)
                self.__learnable_vars[-1]['data_act_params'] = \
                    interpolator_data.get_knots_variable()
            data_activation = interpolator_data.apply_linear(Lxb_reshaped)
            data_activation = tf.reshape(data_activation, shape_in)
            return data_activation
        else:
            print('Unknown data term type')
            sys.exit(1)

    def __compute_reg_grad_full(self, x, n_filters_vn, filter_sz_vn):
        if self.__drop:
            rate = self.__rate #Dropout rate for K and K^T
            print('dropout K rate: ', rate) #Sanity check
        x = tf.reshape(x, [-1, self.__img_sz[0], self.__img_sz[1], 1])
        if not self.__KL: #Regular filter initialization
            Ds = tf.get_variable('filters',
               initializer= tf.truncated_normal(
                    [filter_sz_vn, filter_sz_vn, 1, n_filters_vn], stddev=1e-2),
                dtype=tf.float32)

        if self.__KL: #Bayesian Variational Inference
            #Initialization of Gausians mean
            Ds_pre = tf.get_variable('filters', initializer=tf.truncated_normal([filter_sz_vn, filter_sz_vn, 1, n_filters_vn], stddev=1e-2),dtype=tf.float32)
            Ds_pre = tf.reshape(Ds_pre, shape = [ filter_sz_vn*filter_sz_vn*n_filters_vn, 1])
            # Initialization of Gausians covariance
            st_dev_mat_pre = tf.get_variable('filters_stdev',initializer=tf.random_uniform([1, n_filters_vn, tf.cast(filter_sz_vn * filter_sz_vn * (filter_sz_vn * filter_sz_vn + 1) / 2, dtype=tf.int32)], 0.9, 1.1),dtype=tf.float32)
            st_dev_mat_block = tf.contrib.distributions.fill_triangular(st_dev_mat_pre)
            d_op = []
            #Stacking stdevs blockwise, creating block diagonal matrix
            for i in range(st_dev_mat_block.shape[1]):
                d_op.append(tf.linalg.LinearOperatorFullMatrix(st_dev_mat_block[0,i,...]))
            st_dev_mat = tf.linalg.LinearOperatorBlockDiag(d_op).to_dense()
            self.__st_dev_mat = st_dev_mat
            #Sampling from Gausian with unit variance
            epsilon = tf.random_normal(shape=[filter_sz_vn*filter_sz_vn*n_filters_vn, 1])
            mult = tf.matmul(st_dev_mat, epsilon)
            Ds = Ds_pre + mult
            Ds = tf.reshape(Ds, shape = [filter_sz_vn, filter_sz_vn, 1, n_filters_vn])
            self.__learnable_vars[-1]['conv_filt_stdev'] = st_dev_mat
            self.__learnable_vars[-1]['conv_filt_mean'] = tf.reshape(Ds_pre, shape = [filter_sz_vn, filter_sz_vn, 1, n_filters_vn])

        if self.__D_standardize:
            # re-parameterize by standardization
            Ds = Ds - tf.reduce_mean(Ds, axis=[0, 1], keepdims=True)
            Ds = Ds / tf.sqrt(tf.reduce_sum(tf.square(Ds),
                                            axis=[0, 1], keepdims=True) + self.__eps_cf)
        else:
            # Learn the mean (new version from Valery)
            Dnrm = Ds - tf.reduce_mean(Ds, axis=[0, 1], keepdims=True)
            Dnrm = Dnrm / tf.sqrt(tf.reduce_sum(tf.square(Dnrm),
                                                axis=[0, 1], keepdims=True) + self.__eps_cf)
            Dnrm = Dnrm + \
                tf.nn.softsign(tf.get_variable('meanfilters',
                    initializer=tf.zeros([1, 1, 1, self.__n_filters_vn]), dtype=tf.float32))
            Ds = Dnrm
        # bookkeeping conv. filters
        self.__learnable_vars[-1]['conv_filt'] = Ds
        # gradient computation [n_batch, img_sz[0], img_sz[1], n_filters_vn]
        if self.__drop:
            # Implementation with MC dropout
            Dx_preDrop = tf.nn.conv2d(
                x, Ds, strides=[1, 1, 1, 1], padding='VALID', name='before_act')
            Dx = tf.nn.dropout(Dx_preDrop, rate)
        else:
            # Original implementation
            Dx = tf.nn.conv2d(x, Ds, strides=[1, 1, 1, 1], padding='VALID', name='before_act')
        self.__layers[-1]['before_act_reg'] = Dx
        # image size after VALID conv.
        imsz_v = [self.__img_sz[i] - filter_sz_vn + 1 for i in [0, 1]]
        # spatial filter weighting
        if self.__use_spatial_filter_weighting:
            cw_sz = 8  # was 16, now 20
            cWeight = tf.get_variable('cWeight', initializer=tf.truncated_normal(
                [1, cw_sz, cw_sz, n_filters_vn], stddev=5e-2),
                dtype=tf.float32)
            cWeight = tf.sigmoid(cWeight*2.5, name='cWeight_sigmoid')
            self.__learnable_vars[-1]['cWeight_ds_sgm'] = cWeight
            cWeight_us = tf.image.resize_bilinear(
                cWeight, [imsz_v[0], imsz_v[1]], align_corners=True)
            self.__learnable_vars[-1]['cWeight_us_sgm'] = cWeight_us

        n_pix = imsz_v[0] * imsz_v[1] * self.__n_batch
        if self.ada_interpolator:
            interpolator_reg = AdaptiveInterpolator(
                n_pix, n_filters_vn, self.__n_interp_knots_reg, 0.01, scope='interpolation_2')
            self.__ada_interp_minmax_ops.append(
                interpolator_reg.register_input(Dx))
            self.__D2_potf_reg_list.append(interpolator_reg.D2_L1_regularization())
            self.__ada_interp_objects.append(interpolator_reg)
            self.__learnable_vars[-1]['interp_var_reg'], self.__learnable_vars[-1]['interp_max'] = interpolator_reg.get_responce_vars()

        else:
            interpolator_reg = FixedInterpolator(
                n_pix, n_filters_vn, self.minx_reg, self.maxx_reg, self.__n_interp_knots_reg, 0.01, scope='interpolation_2')
        self.__learnable_vars[-1]['reg_act_params'] = interpolator_reg.get_knots_variable()
        psiDx = interpolator_reg.apply_linear(
            tf.reshape(Dx, [n_pix, n_filters_vn]))
        psiDx = tf.reshape(psiDx, [self.__n_batch, imsz_v[0],
                                   imsz_v[1], n_filters_vn], name='after_interp')

        if self.__use_spatial_filter_weighting:
            psiDx = cWeight_us * psiDx

        if self.__drop:
            # Dropout implementation
            DTDx_preDrop = tf.nn.conv2d_transpose(
                psiDx, Ds,
                output_shape=[self.__n_batch,
                              self.__img_sz[0], self.__img_sz[1], 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name='reg_grad')  # [n_batch, img_sz[0], img_sz[1]]
            DTDx = tf.nn.dropout(DTDx_preDrop, rate)
        else:
            DTDx = tf.nn.conv2d_transpose(
                psiDx, Ds,
                output_shape=[self.__n_batch,
                              self.__img_sz[0], self.__img_sz[1], 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name='reg_grad')  # [n_batch, img_sz[0], img_sz[1]]

        DTDx = tf.reshape(DTDx, [-1, self.__img_sz[0], self.__img_sz[1]])
        return DTDx

    def __Lop(self, x, usm):
        """
        compute matrix multiplication Lx
        INPUT: ray matrix L of shape [n_reads, n_pixel]; image x of shape [n_batch, img_sz[0], img_sz[1]]
        OUTPUT: Lx of shape [n_reads, n_batch]
        """
        x = tf.reshape(x, [-1, self.__img_sz[0]*self.__img_sz[1]])
        x = tf.transpose(x)  # [n_pixel, n_batch]
        Lx = tf.matmul(self.__ray_mat, x)  # [n_reads, n_batch]
        Lx = Lx * usm
        return Lx

    def __LTop(self, b, usm):
        """
        compute matrix multiplication transpose(L)b
        INPUT: ray matrix L of shape [n_reads, n_pixel]; measurement b of shape [n_reads, n_batch]
        OUTPUT: LTb of shape [n_batch, img_sz[0], img_sz[1]]
        """
        if usm is not None:
            b = usm * b
        # [n_pixel, n_batch]
        LTb = tf.matmul(self.__ray_mat, b, transpose_a=True)
        LTb = tf.transpose(LTb)  # [n_batch, n_pixel]
        # [n_batch, img_sz[0], img_sz[1]]
        LTb = tf.reshape(LTb, [-1, self.__img_sz[0], self.__img_sz[1]])
        return LTb

    def __construct_threshold_ops(self):
        """
        threshold learnable parameters including momentV and alpha0
        """
        ops = []
        print(self.__learnable_vars[0].keys())
        for lv_layer in self.__learnable_vars:
            mV = lv_layer['momentV']
            threshold_op = mV.assign(
                tf.clip_by_value(mV, 0, 1, name='clip_momentV'))
            ops.append(threshold_op)
            #a0 = lv_layer['alpha0']
            #threshold_op = a0.assign(tf.maximum(a0, 0))
            ops.append(threshold_op)
        self.__threshold_ops = ops

    def __set_cost(self, cost_type):
        # compute reconstruction cost (type specified by cost_type)
        xout = self.__xs[-1]
        xout = tf.reshape(xout, [-1, self.__img_sz[0], self.__img_sz[1]])
        gt = self.__xgt_nrm
        self.__SAD_cost = tf.reduce_sum(tf.abs(xout - gt)) / self.__n_batch
        if cost_type == 'recon':
            self.__netcost = tf.reduce_sum(tf.abs(xout - gt)) / self.__n_batch
            if self.__regularize_act_func:
                self.__netcost = self.__netcost + self.D2_potf_reg*self.__lambda
            if self.__regularize_act_func_data:
                self.__netcost = self.__netcost + self.D2_potf_data * self.__lambda_data
        elif cost_type == 'recon_iter':
            self.__netcost = 0.
            recon_loss = 0.
            self.layer_loss = []
            if self.__use_temperature:
                loss_weights = tf.exp(
                    np.linspace(-1., 0., self.__n_layers) * self.weighting_temperature)
            else:
                #The scaling parameter (here, 5.) is changed for different experiments and ensembles
                loss_weights = np.exp(np.linspace(-5., 0., self.__n_layers))
            for i in range(self.__n_layers):
                curr_out = self.__xs[i+1]
                curr_out = tf.reshape(
                    curr_out, [-1, self.__img_sz[0], self.__img_sz[1]])

                if self.__aleatoric and i is (self.__n_layers-1):
                    #Creating neg log likelihood loss for aleatoric uncertainty computation
                    self.__bayes = 0.5 * tf.reduce_sum((tf.square(curr_out - gt) / self.ua + tf.log(self.ua))) / self.__n_batch
                    curr_loss = self.__bayes
                else:
                    #Regular VN
                    # L2 loss, comment if L1 desired
                    curr_loss = tf.reduce_sum(tf.square(curr_out - gt)) / self.__n_batch
                    #L1 loss, comment if L2 desired
                    # curr_loss = tf.reduce_sum(tf.abs(curr_out - gt)) / self.__n_batch
                recon_loss += loss_weights[i] * curr_loss
            self.layer_loss.append(curr_loss)
            if self.__regularize_act_func:
               self.__netcost = recon_loss + self.D2_potf_reg * self.__lambda
            if self.__regularize_act_func_data:
                self.__netcost = self.__netcost + self.D2_potf_data * self.__lambda_data
            else:
                self.__netcost = recon_loss
            if self.__KL:
                #Adding KL term to the loss for bayesian Variational Inference
                self.__KL_cost = self.__beta_KL * (self.__alpha_KL * tf.linalg.trace(tf.matmul(self.__st_dev_mat, tf.transpose(self.__st_dev_mat))) - 2 * tf.trace(tf.log(self.__st_dev_mat)))
                #Prenetcost used to logg state of the terms of the loss
                self.__prenetcost = [recon_loss + self.D2_potf_reg * self.__lambda, curr_loss, tf.reduce_any(tf.is_nan(curr_out)), recon_loss, 2 * tf.trace(tf.log(self.__st_dev_mat)), self.D2_potf_reg * self.__lambda]
                self.__netcost = self.__netcost + self.__KL_cost
            else:
                self.__prenetcost = tf.constant([1])
                self.__KL_cost = tf.constant([1])
        else:
            print('Unknown reconstruction cost')
            sys.exit(1)

    def get_all_model_params(self):
        placeholders = {'d_in': self.__din,
                        'gt_image': self.__img_gt,
                        'd_inpaint': self.__dinpaint_ori,
                        'usm': self.__usm,
                        'w_temp': self.weighting_temperature}
        L = self.__ray_mat
        layers = self.__layers
        net_cost = self.__netcost
        KL_cost = self.__KL_cost
        pre_netcost = self.__prenetcost
        init_img = self.__xs[0]
        recon_nrm = self.__xs[-1]
        xgt_nrm = self.__xgt_nrm
        recon_sos = self.__recon_sos
        layer_sos = self.__layer_sos
        threshold_ops = self.__threshold_ops
        SADbatch = self.__SAD_cost
        din_whiten = self.__din_whiten
        if self.__aleatoric:
            ua = self.ua
            ua_sos = self.ua_sos
            return {
                    'k': self.__k3,
                    'std': self.__std,
                    'xs': self.__xs,
                    'placeholders': placeholders,
                    'layers': layers,
                    'L': L,
                    'net_cost': net_cost,
                    'pre_net_cost': pre_netcost,
                    'KL_cost': KL_cost,
                    'init_img': init_img,
                    'xgt_nrm': xgt_nrm,
                    'recon_nrm': recon_nrm,
                    'recon_sos': recon_sos,
                    'layer_sos': layer_sos,
                    'threshold_ops': threshold_ops,
                    'SAD_train': SADbatch,
                    'din_whiten': din_whiten,
                    'ua': ua,
                    'ua_sos': ua_sos}
        else:
            return {
                'k': self.__k3,
                'std': self.__std,
                'xs': self.__xs,
                'placeholders': placeholders,
                'layers': layers,
                'L': L,
                'net_cost': net_cost,
                'pre_net_cost': pre_netcost,
                'KL_cost': KL_cost,
                'init_img': init_img,
                'xgt_nrm': xgt_nrm,
                'recon_nrm': recon_nrm,
                'recon_sos': recon_sos,
                'layer_sos': layer_sos,
                'threshold_ops': threshold_ops,
                'SAD_train': SADbatch,
                'din_whiten': din_whiten}

    def eval_batch_n_export(self, tf_sess, feed_dict, test=False):
        """
        export tensors from __layers and __learnable_vars
        """
        ops = []
        idxs = []
        names = []
        # get all parameters
        for i, layer in enumerate(self.__layers):
            for k, v in layer.items():
                idxs.append(i)
                names.append(k)
                ops.append(v)

        # for learnable variables
        names_v = []
        idxs_v = []
        ops_v = []
        for i, layer_lv in enumerate(self.__learnable_vars):
            for k_lv, val_lv in layer_lv.items():
                idxs_v.append(i)
                names_v.append(k_lv)
                ops_v.append(val_lv)
        # evaluate
        vals, vals_v = tf_sess.run([ops, ops_v], feed_dict=feed_dict)
        dict_mat = {}
        for i in range(len(vals)):
            k = names[i]
            if k not in dict_mat:
                dict_mat[k] = []
            vnpar = np.array(vals[i])
            dict_mat[k].append(vnpar)
        for i in range(len(vals_v)):
            k_lv = names_v[i]
            if k_lv not in dict_mat:
                dict_mat[k_lv] = []
            vnpar = np.array(vals_v[i])
            dict_mat[k_lv].append(vnpar)
        # store unrolled values and intermediate layers
        if not test:
            if not self.__aleatoric:
                xs, intermediate_sos, din_whiten, k, std, xgt_nrm, usm = tf_sess.run(
                    [self.__xs, self.__layer_sos, self.__din_whiten, self.__k3, self.__std, self.__xgt_nrm, self.__usm], feed_dict=feed_dict)
            elif self.__aleatoric:
                xs, intermediate_sos, din_whiten, k, std, xgt_nrm, usm, ua, ua_sos = tf_sess.run(
                    [self.__xs, self.__layer_sos, self.__din_whiten, self.__k3, self.__std, self.__xgt_nrm, self.__usm, self.ua, self.ua_sos],
                    feed_dict=feed_dict)
            dict_mat['xgt_nrm'] = xgt_nrm
        else:
            if not self.__aleatoric:
                xs, intermediate_sos, din_whiten, k, std, usm = tf_sess.run(
                    [self.__xs, self.__layer_sos, self.__din_whiten, self.__k3, self.__std, self.__usm], feed_dict=feed_dict)
            elif self.__aleatoric:
                xs, intermediate_sos, din_whiten, k, std, usm, ua,  ua_sos = tf_sess.run(
                    [self.__xs, self.__layer_sos, self.__din_whiten, self.__k3, self.__std, self.__usm, self.ua, self.ua_sos],
                    feed_dict=feed_dict)
        dict_mat['init_img'] = xs[0]
        dict_mat['xs'] = xs
        dict_mat['layer_sos'] = intermediate_sos
        dict_mat['recon'] = intermediate_sos[-1]
        dict_mat['k'] = k
        dict_mat['std'] = std
        dict_mat['din_whiten'] = din_whiten
        dict_mat['usm'] = usm
        if self.__aleatoric:
            dict_mat['ua_sos'] = ua_sos
            dict_mat['ua'] = ua
        return dict_mat

    def eval_param_n_export(self, sess):
        """
        evaluate and export learnable parameters
        """
        dict_mat = {}
        for i, layer_lv in enumerate(self.__learnable_vars):
            for k_lv, val_lv in layer_lv.items():
                if k_lv not in dict_mat:
                    dict_mat[k_lv] = []
                dict_mat[k_lv].append(val_lv.eval())
        return dict_mat

    def validate(self, sess, validation_iterator, N_val, mat_dir, i_iter, use_med, use_mean, save=True, logger=None, name='syn'):
        first = True
        done = False
        netout = self.__xs[-1]
        gt = self.__xgt_nrm
        netout_sos = self.__recon_sos
        gt_sos = 1 / self.__img_gt
        val_cost = 0
        val_cost_sos = 0
        while not done:
            try:
                batch_d, batch_inpaint, batch_mask, batch_img = next(
                    validation_iterator)
                feed_dict = {self.__din: batch_d,
                             self.__dinpaint_ori: batch_inpaint,
                             self.__usm: batch_mask,
                             self.__img_gt: batch_img}  # self.__ray_mat: rayValue
                recon = netout.eval(feed_dict=feed_dict)
                xgt_nrm = gt.eval(feed_dict=feed_dict)
                recon_sos = netout_sos.eval(feed_dict=feed_dict)
                xgt_sos = gt_sos.eval(feed_dict=feed_dict)
                val_cost += np.sum(np.abs(recon - xgt_nrm))
                val_cost_sos += np.sum(np.mean(np.abs(recon_sos -
                                                      xgt_sos), axis=(1, 2)))

                if first and save:
                    feed_dict = {self.__din: batch_d,
                                 self.__usm: batch_mask,
                                 self.__img_gt: batch_img,
                                 self.__dinpaint_ori: batch_inpaint}  # self.__ray_mat: rayValue
                    k, k2, k3 = sess.run(
                        [self.__k1, self.__k2, self.__k3], feed_dict=feed_dict)
                    if logger is not None:
                        logger.info('k {}'.format(k))
                        logger.info('k2 {}'.format(k2))
                        logger.info('k3 {}'.format(k3))
                        logger.info('true mean of images {}'.format(
                        np.mean(batch_img*self.__s1, axis=(1, 2))))
                    else:
                        print('k {}'.format(k))
                        print('k2 {}'.format(k2))
                        print('true mean of images {}'.format(
                            np.mean(batch_img, axis=(1, 2))))
                    dict_mat = self.eval_batch_n_export(sess, feed_dict)
                    dict_mat['din'] = batch_d
                    dict_mat['dinpaint'] = batch_inpaint
                    dict_mat['xgt_slowness'] = batch_img
                    dinpaint_val, xgt_nrm_val, recon_sos_val = sess.run(
                        [self.__dinpaint_ori, self.__xgt_nrm, self.__recon_sos],
                        feed_dict=feed_dict)
                    dict_mat['dinpaint'] = dinpaint_val
                    dict_mat['xgt_nrm'] = xgt_nrm_val
                    dict_mat['recon_sos'] = recon_sos_val
                    sio.savemat(os.path.join(
                        mat_dir, 'dm-val-{}-{}'.format(name, i_iter)), dict_mat)
                    del dict_mat
                    first = False
            except StopIteration:
                done = True
        val_cost = val_cost / N_val
        val_cost_sos = val_cost_sos / N_val
        return val_cost, val_cost_sos

    def check_whitening_step(self, sess, d_batch, usm_batch):
        feed_dict = {self.__din: d_batch, self.__usm: usm_batch}
        curr_din_whiten = self.__din_whiten.eval(feed_dict=feed_dict)
        curr_k = self.__k3.eval(feed_dict=feed_dict)
        curr_std = self.__std.eval(feed_dict=feed_dict)
        export_dict = {'din_whiten': curr_din_whiten,
                       'k': curr_k, 'std': curr_std}
        return export_dict

    def readjust_interp_resp(self, tf_sess):
        for i in self.__ada_interp_objects:
            i.readjust_responce_range(tf_sess)

    def get_interp_minmax_ops(self):
        return self.__ada_interp_minmax_ops

