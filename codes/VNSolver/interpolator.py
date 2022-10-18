"""
Fixed and Adaptive interpolator classes definitions. 
Code from Valery Vishnevskiy. 
"""

import numpy as np
import tensorflow as tf
import scipy


class AdaptiveInterpolator():
    """ 
    Adaptive interpolator.
    """
    def __init__(self, n_pix, n_flt, n_interp_knots, stddev_init, init_yk=None, scope='interpolation'):
        self.__n_interp_knots = n_interp_knots
        self.__n_pix = n_pix
        self.__n_flt = n_flt
        self.__scope = scope
        with tf.variable_scope(self.__scope):
            self.__flt_response_var = tf.get_variable('flt_resp_var',
                initializer=1., dtype=tf.float32, trainable=False)
            if init_yk is None:
                self.__yK = tf.get_variable('interp_knots', initializer = tf.truncated_normal(
                    [n_interp_knots, n_flt], stddev=stddev_init), dtype=tf.float32)
            else:
                if np.any(init_yk.shape != np.array([n_interp_knots, n_flt])):
                    sys.exit('Init values have incorrect shape')
                self.__yK = tf.get_variable('interp_knots',
                    initializer=init_yk, dtype=tf.float32)
            self.__interp_grid = tf.constant(np.tile(
                np.array(range(0, n_flt)).reshape([1, n_flt]), [n_pix, 1]), dtype=tf.float32)
            self.__minx = - self.__flt_response_var
            self.__maxx = self.__flt_response_var
            self.__w = (self.__maxx - self.__minx) / (n_interp_knots - 1)

    def register_input(self, xin):
        # this strange renaming is to restore from models
        # obtained with the old code version.
        if self.__scope == 'interpolation':
            scope_reg = 'interpolation_1'
        elif self.__scope == 'interpolation_2':
            scope_reg = 'interpolation_3'
        else:
            scope_reg = self.__scope
        with tf.variable_scope(scope_reg):
            flt_response_max = tf.get_variable('flt_resp_max',
               initializer= 1., dtype=tf.float32, trainable=False)
        # cur_max = 8 * tf.reduce_mean(tf.abs(xin))
        cur_max = tf.reduce_max(tf.abs(xin))
        op_maxabs = flt_response_max.assign(
            flt_response_max * 0.95 + cur_max * 0.05)
        self.__flt_response_max = flt_response_max
        return op_maxabs

    def apply_linear(self, xin):
        # xin npix - nflt
        xS = (xin - self.__minx) / self.__w
        xS = tf.clip_by_value(xS, 0.00001, self.__n_interp_knots - 1.00001)
        xF = tf.floor(xS)
        k = xS - xF
        idx_f = xF
        idx_c = xF + 1
        # tf.stack moves int32 to CPU, so use int(stack()) not stack(int())
        nd_idx1 = tf.to_int32(tf.stack([idx_f, self.__interp_grid], 2))
        nd_idx2 = tf.to_int32(tf.stack([idx_c, self.__interp_grid], 2))
        y_f = tf.gather_nd(self.__yK, nd_idx1)
        y_c = tf.gather_nd(self.__yK, nd_idx2)
        y = y_f * (1 - k) + k * y_c
        return y

    def get_knots_variable(self):
        return self.__yK

    def get_responce_vars(self):
        return self.__flt_response_var, self.__flt_response_max
    
    def D2_L1_regularization(self, deps = 1e-6):
        x0 = self.__yK[0:-3, :]
        x1 = self.__yK[1:-2, :]
        x2 = self.__yK[2:-1, :]
        ddx = x0 + x2 - 2. * x1
        return tf.reduce_sum(tf.sqrt(ddx**2 + deps))

    def readjust_responce_range(self, tf_sess):
        rng = self.__flt_response_var.eval()
        yK = self.__yK.eval()
        rng_new = self.__flt_response_max.eval()
        if rng < rng_new * 1.45 and rng > rng_new * 0.95:
            # print('No adjustment_needed at layer', i)
            return
        rng_new = np.maximum(rng_new, 1e-6)
        # rng_new = rng_new * 1.05
        rng_new = rng * 0.3 + rng_new * 0.7
        print('Layer from ', rng, ' to ', rng_new)
        x_old = np.linspace(-rng, +rng, self.__n_interp_knots)
        x_new = np.linspace(-rng_new, +rng_new, self.__n_interp_knots)

        yK_new = []
        for j in range(yK.shape[1]):
            finterp = scipy.interpolate.interp1d(
                x_old, yK[:, j], 'linear', fill_value=(yK[0, j], yK[-1, j]), bounds_error=False)
            yK_new.append(finterp(x_new))
            # yK_new.append(np.interp(x_new, x_old, yK[:, j]))
        yK_new = np.stack(yK_new, axis=1)
        #print('old ', yK[:, 0])
        #print('new ', yK_new[:, 0])
        self.__yK.load(yK_new)
        self.__flt_response_var.load(rng_new)


# class used to implement interpolated activation function
class FixedInterpolator():
    def __init__(self, n_pix, n_flt, minx, maxx, n_interp_knots, stddev_init, init_yk=None, scope = 'interpolation'):
        self.__n_interp_knots = n_interp_knots
        self.__n_pix = n_pix
        self.__n_flt = n_flt
        # length of each interval
        self.__w = (maxx - minx) / (n_interp_knots - 1)
        self.__minx = minx
        self.__maxx = maxx
        self.__scope = scope
        with tf.variable_scope(self.__scope):
            if init_yk is None:
                self.__yK = tf.get_variable('interp_knots', tf.truncated_normal(
                    [n_interp_knots, n_flt], stddev=stddev_init), dtype=tf.float32)
            else:
                if np.any(init_yk.shape != np.array([n_interp_knots, n_flt])):
                    sys.exit('Init values have incorrect shape')
                self.__yK = tf.get_variable('interp_knots',
                    initializer=init_yk, dtype=tf.float32)
        self.__interp_grid = tf.constant(np.tile(
            np.array(range(0, n_flt)).reshape([1, n_flt]), [n_pix, 1]), dtype=tf.float32)

    def apply_linear(self, xin):
        # xin npix - nflt
        # compute the current input lies in which interval
        xS = (xin - self.__minx) / self.__w
        # make sure indices do not exceed # of knots
        xS = tf.clip_by_value(xS, 0.00001, self.__n_interp_knots - 1.00001)
        xF = tf.floor(xS)
        k = xS - xF
        idx_f = xF  # Python index
        idx_c = xF + 1  # Python index
        # tf.stack moves int32 to CPU, so use int(stack()) not stack(int())
        # [npix, nflt, 2]: for every element in xin, select [knot, flt] pair
        nd_idx1 = tf.to_int32(tf.stack([idx_f, self.__interp_grid], 2))
        nd_idx2 = tf.to_int32(tf.stack([idx_c, self.__interp_grid], 2))
        y_f = tf.gather_nd(self.__yK, nd_idx1)
        y_c = tf.gather_nd(self.__yK, nd_idx2)
        y = y_f * (1 - k) + k * y_c
        return y  # npix - nflt

    def apply_cubic(self, xin):
        xS = (xin - self.__minx) / self.__w
        xS = tf.clip_by_value(xS, 0, self.__n_interp_knots - 0.001)
        xF = tf.floor(xS)
        k = xS - xF
        idx_1 = tf.clip_by_value(xF - 1, 0, self.__n_interp_knots - 1)
        idx_2 = tf.clip_by_value(xF - 0, 0, self.__n_interp_knots - 1)
        idx_3 = tf.clip_by_value(xF + 1, 0, self.__n_interp_knots - 1)
        idx_4 = tf.clip_by_value(xF + 2, 0, self.__n_interp_knots - 1)
        nd_idx1 = tf.to_int32(tf.stack([idx_1, self.__interp_grid], 2))
        nd_idx2 = tf.to_int32(tf.stack([idx_2, self.__interp_grid], 2))
        nd_idx3 = tf.to_int32(tf.stack([idx_3, self.__interp_grid], 2))
        nd_idx4 = tf.to_int32(tf.stack([idx_4, self.__interp_grid], 2))

        y_1 = tf.gather_nd(self.__yK, nd_idx1)
        y_2 = tf.gather_nd(self.__yK, nd_idx2)
        y_3 = tf.gather_nd(self.__yK, nd_idx3)
        y_4 = tf.gather_nd(self.__yK, nd_idx4)

        k1 = k * ((2 - k) * k - 1)
        k2 = (k * k * (3 * k - 5) + 2)
        k3 = k * ((4 - 3 * k) * k + 1)
        k4 = k * k * (k - 1)
        y = (k1 * y_1 + k2 * y_2 + k3 * y_3 + k4 * y_4) * 0.5
        return y

    def get_knots_variable(self):
        return self.__yK

