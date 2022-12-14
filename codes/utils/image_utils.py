# Authors: Emmanuelle Gouillart <emmanuelle.gouillart@normalesup.org>
#          Gael Varoquaux <gael.varoquaux@normalesup.org>
#          Olivier Grisel
#          Vlad Niculae
# License: BSD 3 clause

# Adapted by Weiye Li - 02.05.2019 - ETH Zurich
# Modified places are marked as "MODIFIED"

from itertools import product
import numbers
import numpy as np
from scipy import sparse
from numpy.lib.stride_tricks import as_strided

###############################################################################
# Added by Weiye Li
# to implement median and mean filter inpainting

def medfilt2d_usm(image, mask, patch_size=3):
    """
    fill in masked region in input image with median-filtered values
    INPUT:
    image: 2d array, with NaN entries at locations where [mask==0]
    mask: binary mask with the same shape as image
    patch_size: integer, specifying height and width of square patch; default is 3
    OUTPUT:
    A: median-filtered image
    """

    A = image
    B = image
    B[mask==0] = np.nan
    # median filter B with the following steps
    # 1. extract patch at each pixel location
    pad_width = int((patch_size - 1) / 2)
    Bp = np.pad(array=B, pad_width=pad_width, mode='symmetric')
    patches = extract_patches_2d(image=Bp, patch_size=(patch_size, patch_size), max_patches=None) # set max_patches to None gives patch at every pixel location
    # patches of shape [num_pixel, patch_size, patch_size]
    # 2. compute median within patch ignoring NaNs
    medians = np.nanmedian(patches, axis=[1,2], keepdims=False)
    isNaN = np.unique(np.isnan(medians))
    # should all be False
    # in the extreme case where extracted patch only contains NaN, nanmedian will still give NaN
    # fix this issue by replacing NaN in medians with mean value of all non-NaN medians
    if len(isNaN) > 1:
        medians[np.isnan(medians)] = np.nanmean(medians)
        # medians[np.isnan(medians)] = 0
    assert(len(np.unique(np.isnan(medians)))==1)
    # medians of shape [num_pixel,]
    # 3. reshape to input image dimension
    Bf = np.reshape(medians, B.shape)
    # 4. fill in masked region in A
    A[mask==0] = Bf[mask==0]
    return A


def meanfilt2d_usm(image, mask, patch_size=3):
    """
    fill in masked region in input image with mean-filtered values
    INPUT:
    image: 2d array, with NaN entries at locations where [mask==0]
    mask: binary mask with the same shape as image
    patch_size: integer, specifying height and width of square patch; default is 3
    OUTPUT:
    A: mean-filtered image
    """

    A = image
    B = image
    B[mask==0] = np.nan
    # median filter B with the following steps
    # 1. extract patch at each pixel location
    pad_width = int((patch_size - 1) / 2)
    Bp = np.pad(array=B, pad_width=pad_width, mode='symmetric')
    patches = extract_patches_2d(image=Bp, patch_size=(patch_size, patch_size), max_patches=None) # set max_patches to None gives patch at every pixel location
    # patches of shape [num_pixel, patch_size, patch_size]
    # 2. compute mean within patch ignoring NaNs
    means = np.nanmean(patches, axis=(1,2), keepdims=False)
    isNaN = np.unique(np.isnan(means))
    # should all be False
    # in the extreme case where extracted patch only contains NaN, nanmedian will still give NaN
    # fix this issue by replacing NaN in medians with mean value of all non-NaN medians
    if len(isNaN) > 1:
        #means[np.isnan(means)] = np.nanmean(means)
        # 3. reshape to input image dimension
        Bf = np.reshape(means, B.shape)
        # 4. fill in masked region in A
        A[mask==0] = Bf[mask==0]
        Bp = np.pad(array=A, pad_width=pad_width, mode='symmetric')
        # 5. Retiterate
        patches = extract_patches_2d(image=Bp, patch_size=(patch_size, patch_size), max_patches=None) # set max_patches to None gives patch at every pixel location
        means = np.nanmean(patches, axis=(1,2), keepdims=False)
        isNaN = np.unique(np.isnan(means))
        if len(isNaN) > 1:
            # 3. reshape to input image dimension
            Bf = np.reshape(means, B.shape)
            # 4. fill in masked region in A
            A[mask==0] = Bf[mask==0]
            Bp = np.pad(array=A, pad_width=pad_width, mode='symmetric')
            # 5. Retiterate
            patches = extract_patches_2d(image=Bp, patch_size=(patch_size, patch_size), max_patches=None) # set max_patches to None gives patch at every pixel location
            means = np.nanmean(patches, axis=(1,2), keepdims=False)
            isNaN = np.unique(np.isnan(means))
            if len(isNaN) > 1:
                means[np.isnan(means)] = np.nanmean(means)
    assert(len(np.unique(np.isnan(means)))==1)
    # medians of shape [num_pixel,]
    # 3. reshape to input image dimension
    Bf = np.reshape(means, B.shape)
    # 4. fill in masked region in A
    A[mask==0] = Bf[mask==0]
    return A


###############################################################################
# From an image to a set of small image patches

def _compute_n_patches(i_h, i_w, p_h, p_w, max_patches=None):
    """Compute the number of patches that will be extracted in an image.

    Read more in the :ref:`User Guide <image_feature_extraction>`.

    Parameters
    ----------
    i_h : int
        The image height
    i_w : int
        The image with
    p_h : int
        The height of a patch
    p_w : int
        The width of a patch
    max_patches : integer or float, optional default is None
        The maximum number of patches to extract. If max_patches is a float
        between 0 and 1, it is taken to be a proportion of the total number
        of patches.
    """
    n_h = i_h - p_h + 1
    n_w = i_w - p_w + 1
    all_patches = n_h * n_w

    if max_patches:
        if (isinstance(max_patches, (numbers.Integral))
                and max_patches < all_patches):
            return max_patches
        elif (isinstance(max_patches, (numbers.Integral))
              and max_patches >= all_patches):
            return all_patches
        elif (isinstance(max_patches, (numbers.Real))
                and 0 < max_patches < 1):
            return int(max_patches * all_patches)
        else:
            raise ValueError("Invalid value for max_patches: %r" % max_patches)
    else:
        return all_patches


def extract_patches(arr, patch_shape=8, extraction_step=1):
    """Extracts patches of any n-dimensional array in place using strides.

    Given an n-dimensional array it will return a 2n-dimensional array with
    the first n dimensions indexing patch position and the last n indexing
    the patch content. This operation is immediate (O(1)). A reshape
    performed on the first n dimensions will cause numpy to copy data, leading
    to a list of extracted patches.

    Read more in the :ref:`User Guide <image_feature_extraction>`.

    Parameters
    ----------
    arr : ndarray
        n-dimensional array of which patches are to be extracted

    patch_shape : integer or tuple of length arr.ndim
        Indicates the shape of the patches to be extracted. If an
        integer is given, the shape will be a hypercube of
        sidelength given by its value.

    extraction_step : integer or tuple of length arr.ndim
        Indicates step size at which extraction shall be performed.
        If integer is given, then the step is uniform in all dimensions.


    Returns
    -------
    patches : strided ndarray
        2n-dimensional array indexing patches on first n dimensions and
        containing patches on the last n dimensions. These dimensions
        are fake, but this way no data is copied. A simple reshape invokes
        a copying operation to obtain a list of patches:
        result.reshape([-1] + list(patch_shape))
    """

    arr_ndim = arr.ndim

    if isinstance(patch_shape, numbers.Number):
        patch_shape = tuple([patch_shape] * arr_ndim)
    if isinstance(extraction_step, numbers.Number):
        extraction_step = tuple([extraction_step] * arr_ndim)

    patch_strides = arr.strides

    slices = tuple(slice(None, None, st) for st in extraction_step)
    indexing_strides = arr[slices].strides

    patch_indices_shape = ((np.array(arr.shape) - np.array(patch_shape)) //
                           np.array(extraction_step)) + 1

    shape = tuple(list(patch_indices_shape) + list(patch_shape))
    strides = tuple(list(indexing_strides) + list(patch_strides))

    patches = as_strided(arr, shape=shape, strides=strides)
    return patches


def extract_patches_2d(image, patch_size, max_patches=None, random_state=None):
    """Reshape a 2D image into a collection of patches

    The resulting patches are allocated in a dedicated array.

    Read more in the :ref:`User Guide <image_feature_extraction>`.

    Parameters
    ----------
    image : array, shape = (image_height, image_width) or
        (image_height, image_width, n_channels)
        The original image data. For color images, the last dimension specifies
        the channel: a RGB image would have `n_channels=3`.

    patch_size : tuple of ints (patch_height, patch_width)
        the dimensions of one patch

    max_patches : integer or float, optional default is None
        The maximum number of patches to extract. If max_patches is a float
        between 0 and 1, it is taken to be a proportion of the total number
        of patches.

    random_state : int, RandomState instance or None, optional (default=None)
        Pseudo number generator state used for random sampling to use if
        `max_patches` is not None.  If int, random_state is the seed used by
        the random number generator; If RandomState instance, random_state is
        the random number generator; If None, the random number generator is
        the RandomState instance used by `np.random`.

    Returns
    -------
    patches : array, shape = (n_patches, patch_height, patch_width) or
         (n_patches, patch_height, patch_width, n_channels)
         The collection of patches extracted from the image, where `n_patches`
         is either `max_patches` or the total number of patches that can be
         extracted.

    Examples
    --------

    >>> from sklearn.feature_extraction import image
    >>> one_image = np.arange(16).reshape((4, 4))
    >>> one_image
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])
    >>> patches = image.extract_patches_2d(one_image, (2, 2))
    >>> print(patches.shape)
    (9, 2, 2)
    >>> patches[0]
    array([[0, 1],
           [4, 5]])
    >>> patches[1]
    array([[1, 2],
           [5, 6]])
    >>> patches[8]
    array([[10, 11],
           [14, 15]])
    """
    i_h, i_w = image.shape[:2]
    p_h, p_w = patch_size

    if p_h > i_h:
        raise ValueError("Height of the patch should be less than the height"
                         " of the image.")

    if p_w > i_w:
        raise ValueError("Width of the patch should be less than the width"
                         " of the image.")

    # NOTE: MODIFIED, commented out to allow NaN values in images
    # image = check_array(image, allow_nd=True)
    image = image.reshape((i_h, i_w, -1))
    n_colors = image.shape[-1]

    extracted_patches = extract_patches(image,
                                        patch_shape=(p_h, p_w, n_colors),
                                        extraction_step=1)

    n_patches = _compute_n_patches(i_h, i_w, p_h, p_w, max_patches)
    if max_patches:
        rng = check_random_state(random_state)
        i_s = rng.randint(i_h - p_h + 1, size=n_patches)
        j_s = rng.randint(i_w - p_w + 1, size=n_patches)
        patches = extracted_patches[i_s, j_s, 0]
    else:
        patches = extracted_patches

    patches = patches.reshape(-1, p_h, p_w, n_colors)
    # remove the color dimension if useless
    if patches.shape[-1] == 1:
        return patches.reshape((n_patches, p_h, p_w))
    else:
        return patches


def reconstruct_from_patches_2d(patches, image_size):
    """Reconstruct the image from all of its patches.

    Patches are assumed to overlap and the image is constructed by filling in
    the patches from left to right, top to bottom, averaging the overlapping
    regions.

    Read more in the :ref:`User Guide <image_feature_extraction>`.

    Parameters
    ----------
    patches : array, shape = (n_patches, patch_height, patch_width) or
        (n_patches, patch_height, patch_width, n_channels)
        The complete set of patches. If the patches contain colour information,
        channels are indexed along the last dimension: RGB patches would
        have `n_channels=3`.

    image_size : tuple of ints (image_height, image_width) or
        (image_height, image_width, n_channels)
        the size of the image that will be reconstructed

    Returns
    -------
    image : array, shape = image_size
        the reconstructed image

    """
    i_h, i_w = image_size[:2]
    p_h, p_w = patches.shape[1:3]
    img = np.zeros(image_size)
    # compute the dimensions of the patches array
    n_h = i_h - p_h + 1
    n_w = i_w - p_w + 1
    for p, (i, j) in zip(patches, product(range(n_h), range(n_w))):
        img[i:i + p_h, j:j + p_w] += p

    for i in range(i_h):
        for j in range(i_w):
            # divide by the amount of overlap
            # XXX: is this the most efficient way? memory-wise yes, cpu wise?
            img[i, j] /= float(min(i + 1, p_h, i_h - i) *
                               min(j + 1, p_w, i_w - j))
    return img
