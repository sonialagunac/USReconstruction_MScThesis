"""
Code adapted from Weiye Li and Valery Vishnevskiy
Melanie Bernhardt - M.Sc. Thesis - ETH Zurich
"""

import numpy as np
from skimage.transform import resize as skresize
from skimage import img_as_bool as skbool
from codes.utils.image_utils import medfilt2d_usm, meanfilt2d_usm
import sys


def generate_patchy_random_mask(Nimgs, dim_lr, dim_hr, NA, usm_rate):
    mask = np.float32(np.random.rand(
        dim_lr[0]*dim_lr[1]*NA, Nimgs) >= usm_rate)
    mask = np.transpose(mask) # Nimgs, dim_lr[0]*dim_lr[1]*NA
    mask = np.reshape(mask, [-1, NA, dim_lr[0], dim_lr[1]])
    mask = np.transpose(mask, [0, 2, 3, 1])  # [Nimgs, dim_lr, dim_lr, NA]
    mask = np.float32(skbool(skresize(mask, (Nimgs, dim_hr[0], dim_hr[1], NA),
                                      mode='reflect',
                                      anti_aliasing=True)))  # [Nimgs, dim_hr, dim_hr, NA]
    mask = np.transpose(mask, [0, 3, 1, 2])
    mask = np.reshape(mask, [-1, NA*dim_hr[0]*dim_hr[1]])
    mask = np.transpose(mask)  # [Nreads, Nimgs]
    return mask


def prepare_input_sos(d, msz=[64, 64], na=2, noise_rate=0.01,
                      usm_rate=0.4, fixedMask=None,
                      random_mask_type='patchy',
                      use_med_filter=False,
                      use_mean_filter=False,
                      med_patch_sz=5):

    Nreads = d.shape[0]
    Nimgs = d.shape[1]
    dinpaint = d + 0
    noise = np.random.randn(Nreads, Nimgs) * noise_rate * \
        1e-7 * np.random.rand(1, Nimgs)
    dn = np.float32(d + noise)  # [Nreads, Nimgs]

    if fixedMask is not None:
        masksFixed = np.reshape(fixedMask, [Nreads, 1])
        masksFixed = np.float32(
            np.repeat(masksFixed, repeats=Nimgs, axis=1))
        # [Nreads, Nimgs]
        if random_mask_type == 'patchy':
            masksRandom = generate_patchy_random_mask(
                Nimgs, [16, 16], msz, na, usm_rate) * np.float32(~np.isnan(dn))

        elif random_mask_type == 'plain':
            masksRandom = np.float32(np.random.rand(
                Nreads, Nimgs) >= usm_rate) * np.float32(~np.isnan(dn))  # [Nreads, Nimgs]
        elif random_mask_type == 'test':
            masksRandom = np.float32(~np.isnan(dn))
        else:
            print('Unknown random undersampling pattern. Use patchy or plain')
            sys.exit(1)

    else:
        masksFixed = np.float32(np.ones([Nreads, Nimgs]))
        # random mask ONLY
        if random_mask_type == 'patchy':
            masks = generate_patchy_random_mask(
                Nimgs, [16, 16], msz, na, usm_rate)

        elif random_mask_type == 'plain':
            # usm = np.random.rand(1, Nimgs) * usm_rate
            masks = np.float32(np.random.rand(Nreads, Nimgs) >= usm_rate)
        else:
            print('Unknown random undersampling pattern. Use patchy or plain')
            raise

    masks = masksFixed * masksRandom
    # compute inpainting mask (i.e. only inpaint areas within fov and out of near-field)
    masksInpaint = np.float32(np.invert(np.int32(
        np.invert(np.int32(masksRandom)) * masksFixed)))
    din = dn
    din[masks == 0] = 0
    dinpaint[masksFixed == 0] = np.nan
    if use_med_filter:
        dinpaint = np.transpose(dinpaint)  # [Nimgs, Nreads]
        dinpaint = np.reshape(dinpaint, [-1, na, msz[0], msz[1]])
        masks2d = masksInpaint + 0  # pass inpainting mask to median filter
        masks2d = np.transpose(masks2d)  # [Nimgs, Nreads]
        masks2d = np.reshape(masks2d, [-1, na, msz[0], msz[1]])
        for i_d in range(dinpaint.shape[0]):
            for i_c in range(dinpaint.shape[1]):
                dinpaint[i_d, i_c, :, :] = medfilt2d_usm(
                    dinpaint[i_d, i_c, :, :], masks2d[i_d, i_c, :, :],
                    patch_size=med_patch_sz)
    elif use_mean_filter:
        dinpaint = np.transpose(dinpaint)  # [Nimgs, Nreads]
        dinpaint = np.reshape(dinpaint, [-1, na, msz[0], msz[1]])
        masks2d = masksInpaint + 0  # pass inpainting mask to median filter
        masks2d = np.transpose(masks2d)  # [Nimgs, Nreads]
        masks2d = np.reshape(masks2d, [-1, na, msz[0], msz[1]])
        for i_d in range(dinpaint.shape[0]):
            for i_c in range(dinpaint.shape[1]):
                dinpaint[i_d, i_c, :, :] = meanfilt2d_usm(
                    dinpaint[i_d, i_c, :, :], masks2d[i_d, i_c, :, :],
                    patch_size=med_patch_sz)
    dinpaint = np.reshape(dinpaint, [-1, Nreads])
    dinpaint = np.transpose(dinpaint)  # [Nreads, Nimgs]
    dinpaint = np.float32(dinpaint)
    dinpaint[masksFixed == 0] = 0
    # for memory issue
    del masksFixed
    del masksRandom
    assert np.sum(np.isnan(din)) == 0
    return din, dinpaint, masks
