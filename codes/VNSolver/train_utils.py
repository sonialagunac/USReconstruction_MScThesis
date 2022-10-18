"""
Utils functions.
"""
import scipy.io as sio
import os

def saveBatchMatrices(model, feed_train, din, dinpaint, gt_image,
                      sess, i_iter, mat_dir, save=False):
        """ Wrapper function to save the output of 
        eval_batch_n_export defined in vn_sos_net
        """
        # eval. current training batch and save network parameters
        dict_mat = model.eval_batch_n_export(sess, feed_train)
        dict_mat['din'] = din
        dict_mat['dinpaint'] = dinpaint
        dict_mat['xgt_slowness'] = gt_image
        if save:
            sio.savemat(os.path.join(
                mat_dir, 'dm-train-'+str(i_iter)), dict_mat)
        del dict_mat