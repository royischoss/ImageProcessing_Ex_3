##################################################################################
# This python file is ex3 in image processing course.
# the next script includes functions creating photo pyramid's and photos mixtures.
##################################################################################
# ---------------------------  imports   ---------------------------------------------
from imageio import imread
from skimage.color import rgb2gray
import numpy as np
import scipy as sp
from scipy import signal
import scipy.ndimage as spi
import matplotlib.pyplot as plt

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
        conv_vec1 = signal.convolve(conv_vec1, conv_vec2)
    filter_vec = (1 / np.sum(conv_vec1)) * conv_vec1
    n = im.shape[0]
    m = im.shape[1]
    new_image = np.copy(im)
    i = max_levels - 1
    while m > 16 and n > 16 and i > 0:
        filtered_image = spi.filters.convolve(spi.filters.convolve(new_image, filter_vec),
                                              filter_vec.T)
        new_image = np.copy(filtered_image[::2, ::2])
        pyr.append(new_image)
        i -= 1
        n /= 2
        m /= 2
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
    lpyr.append(gaus_pyr[len(gaus_pyr) - 1])
    return lpyr, gaus_filter


def expand(im, im_filter):
    """
    A helper method gets a matrix and expand it's size and blur it.
    :param im: the matrix of the requested expand needed.
    :param im_filter: the filter used for the expand action.
    :return: expanded im.
    """
    return_mat = np.zeros((2 * im.shape[0], 2 * im.shape[1]), np.float64)
    return_mat[::2, ::2] = im
    return_mat = spi.filters.convolve(spi.filters.convolve(return_mat, 2 * im_filter),
                                      2 * im_filter.T)
    return return_mat


def laplacian_to_image(lpyr, filter_vec, coeff):
    """
    The next function reconstruct an image from a laplacioan pyramid.
    :param lpyr: - a list of images of the laplacian pyramid.
    :param filter_vec: the vector we use in expand blur.
    :param coeff: a list of weights for each level in the pyramid.
    :return: matrix of the image we construct.
    """
    for i in range(len(lpyr)):
        lpyr[i] *= coeff[i]
    temp_mat = lpyr[len(lpyr) - 1]
    for i in range(len(lpyr) - 1, 0, -1):
        temp_mat = expand(temp_mat, filter_vec) + lpyr[i - 1]
    image = temp_mat
    return image


def render_pyramid(pyr, levels):
    """
    The next function gets a pyramid and return matrix of 'levels' numbers of levels stacked horizontally.
    :param pyr: either a Gaussian or Laplacian pyramid.
    :param levels: number of levels to present.
    :return: an image of the the horizontally stacked levels.
    """
    stretch_pyr = []
    for i in range(len(pyr)):
        stretch_pyr.append(stretch(pyr[i], 1))
    rows = pyr[0].shape[0]
    render_pyr = []
    for i in range(levels):
        zeros = np.zeros((rows, pyr[i].shape[1]), np.float64)
        zeros[:pyr[i].shape[0], :] = stretch_pyr[i]
        render_pyr.append(zeros)
    big_image = np.hstack(render_pyr)
    return big_image


def stretch(im, high):
    """
    A helper method for stretching the matrix values linearity between [0,1].
    :param im: the image matrix.
    :param high: the max value for stretch.
    :return: a stretched values matrix.
    """
    low = im.min()
    diff = im.max() - low
    return (im - low) * high / diff


def display_pyramid(pyr, levels):
    """
    The next function display the render image of the pyramid.
    :param pyr: either a Gaussian or Laplacian pyramid.
    :param levels: number of levels included in the image.
    """
    image = render_pyramid(pyr, levels)
    plt.imshow(image, cmap='gray')
    plt.show()




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
