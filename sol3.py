##################################################################################
# This python file is ex3 in image processing course.
# the next script includes functions creating photo pyramid's and photos mixtures.
##################################################################################
# ---------------------------  imports   ---------------------------------------------
from imageio import imread
from skimage.color import rgb2gray
import numpy as np
import scipy as sp
import scipy.ndimage as spi


# --------------------------- constants ----------------------------------------------

MAX_SEGMENT = 255

# -------------------------- functions -----------------------------------------------


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    The function builds a gaussian pyramid.
    :param im:  a grayscale image with double values in [0, 1].
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter.
    :return: tuple of the resulting pyramid pyr as a standard python array with maximum length of max_levels,
    where each element of the array is a grayscale image and filter_vec which is row vector of shape (1, filter_size)
    used for the pyramid construction.
    """
    pyr = [im]
    conv_vec1 = np.ones((1, 2))
    conv_vec2 = np.ones((1, 2))
    for i in range(filter_size - 2):
        conv_vec1 = np.convolve(conv_vec1, conv_vec2)
    filter_vec = (1 / np.sum(conv_vec1)) * conv_vec1
    n = im.shape[0]
    new_image = np.copy(im)
    i = 1
    while n > 16 or i <= max_levels:
        filtered_image = spi.filters.convolve(spi.filters.convolve(new_image, filter_vec, mode='constant', cval=0.0),
                                              filter_vec.T, mode='constant', cval=0.0)
        new_image = np.copy(filtered_image[::2, ::2])
        pyr.append(new_image)
        i += 1
        n /= 4
    return pyr, filter_vec


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    The function builds a laplacian pyramid.
    :param im:  a grayscale image with double values in [0, 1].
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter.
    :return: tuple of the resulting pyramid pyr as a standard python array with maximum length of max_levels,
    where each element of the array is a grayscale image and filter_vec which is row vector of shape (1, filter_size)
    used for the pyramid construction.
    """
    gaus_pyr, gaus_filter = build_gaussian_pyramid(im, max_levels, filter_size)
    lpyr = []
    for i in range(1, len(gaus_pyr)):
        lpyr.append(gaus_pyr[i - 1] - expand(gaus_pyr[i], gaus_filter))
    return lpyr, gaus_filter * 2


def expand(im, im_filter):
    """
    A helper method gets a matrix and expand it's size and blur it.
    :param im: the matrix of the requested expand needed.
    :param im_filter: the filter used for the expand action.
    :return: expanded im.
    """




# from ex1 read image:
def read_image(filename, representation):
    """
    The next lines preform a image read to a matrix of numpy.float64 using
    imagio and numpy libraries.
    :param filename: a path to jpg image we would like to read.
    :param representation: 1 stands for grayscale , 2 for RGB.
    :return: image_mat - a numpy array represents the photo as described above.
    """
    image = imread(filename)
    if representation == 1:
        image_mat = np.array(rgb2gray(image))
    else:
        image_mat = np.array(image.astype(np.float64))
        image_mat /= MAX_SEGMENT
    return image_mat.astype(np.float64)