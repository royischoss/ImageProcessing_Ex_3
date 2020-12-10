##################################################################################
# This python file is ex3 in image processing course.
# the next script includes functions creating photo pyramid's and photos mixtures.
##################################################################################
# ---------------------------  imports   ---------------------------------------------
from imageio import imread
from skimage.color import rgb2gray
import numpy as np
from scipy import signal
import scipy.ndimage as spi
import matplotlib.pyplot as plt
import os

# --------------------------- constants ----------------------------------------------

MAX_SEGMENT = 255
EVERY_TWO = 2
ROWS = 0
COLUMNS = 1
MIN_WIDE = 16
MIN_HEIGHT = 16
END_LEVELS = 0
FIRST = 0
MAX_CLIP = 1
MIN_CLIP = 0

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
    if filter_size == 1:
        filter_vec = np.ones((1, 1))
    else:
        for i in range(filter_size - 2):
            conv_vec1 = signal.convolve(conv_vec1, conv_vec2)
        filter_vec = (1 / np.sum(conv_vec1)) * conv_vec1
    n = im.shape[ROWS]
    m = im.shape[COLUMNS]
    new_image = np.copy(im)
    i = max_levels - 1
    while m > MIN_HEIGHT and n > MIN_WIDE and i > END_LEVELS:
        filtered_image = spi.filters.convolve(spi.filters.convolve(new_image, filter_vec),
                                              filter_vec.T)
        new_image = np.copy(filtered_image[::EVERY_TWO, ::EVERY_TWO])
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
    return_mat = np.zeros((2 * im.shape[ROWS], 2 * im.shape[COLUMNS]), np.float64)
    return_mat[::EVERY_TWO, ::EVERY_TWO] = im
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
    rows = pyr[FIRST].shape[ROWS]
    render_pyr = []
    for i in range(levels):
        zeros = np.zeros((rows, pyr[i].shape[COLUMNS]), np.float64)
        zeros[:pyr[i].shape[ROWS], :] = stretch_pyr[i]
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
    plt.figure()
    image = render_pyramid(pyr, levels)
    plt.imshow(image, cmap=plt.cm.gray)
    plt.show()


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """
    The next function creates blending of the two images given using a mask indicates false where im2 will fill im 1.
    :param im1: image 1 matrix.
    :param im2: image 2 matrix.
    :param mask: a boolean matrix.
    :param max_levels: number of max levels of the pyramid/
    :param filter_size_im: the filter size for the images.
    :param filter_size_mask: the filter size for the mask.
    :return:
    """
    new_mask = mask.astype(np.float64)
    lap_1, filter_vec = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    lap_2, filter_vec = build_laplacian_pyramid(im2, max_levels,filter_size_im)
    gaus_mask, filter_vec_mask = build_gaussian_pyramid(new_mask, max_levels, filter_size_mask)
    lap_3 = []
    for i in range(len(lap_1)):
        ones = np.ones((gaus_mask[i].shape[ROWS], gaus_mask[i].shape[COLUMNS]), np.float64)
        lap_3.append(gaus_mask[i] * lap_1[i] + (ones - gaus_mask[i]) * lap_2[i])
    return np.clip(laplacian_to_image(lap_3, filter_vec, [1] * len(lap_3)), MIN_CLIP, MAX_CLIP)


def blending_example1():
    """
    A function that display a blend example.
    """
    im1_path = realpath("tzipi.jpg")
    im2_path = realpath("dunk.jpg")
    mask_path = realpath("mask1.jpg")
    images, representations = the_big_blend(im1_path, im2_path, mask_path,  10, 5, 5)
    display_images(images, representations)
    return images[0], images[1], images[2], images[3]


def blending_example2():
    """
    A function that display a blend example.
    """
    im1_path = realpath("gambit.jpg")
    im2_path = realpath("chess.jpg")
    mask_path = realpath("mask2.jpg")
    images, representations = the_big_blend(im1_path, im2_path, mask_path, 12, 5, 5)
    display_images(images, representations)
    return images[0], images[1], images[2], images[3]


def realpath(filename):
    """
    A function for controlling the relative path of our files.
    :param filename: the path of your file.
    :return: relative path of your file.
    """
    return os.path.join(os.path.dirname(__file__), filename)


def the_big_blend(im1_path, im2_path, mask_path, max_levels, filter_size_im, filter_size_mask):
    """

    :param im1_path:
    :param im2_path:
    :param mask_path:
    :param max_levels:
    :param filter_size_im:
    :param filter_size_mask:
    :return:
    """
    im1 = read_image(im1_path, 2)
    im2 = read_image(im2_path, 2)
    mask_float = np.round(read_image(mask_path, 1))
    mask_bool = mask_float.astype(np.bool)
    image = np.empty(im1.shape, np.float64)
    image[:, :, 0] = pyramid_blending(im1[:, :, 0], im2[:, :, 0], mask_bool, max_levels, filter_size_im,
                                      filter_size_mask)
    image[:, :, 1] = pyramid_blending(im1[:, :, 1], im2[:, :, 1], mask_bool, max_levels, filter_size_im,
                                      filter_size_mask)
    image[:, :, 2] = pyramid_blending(im1[:, :, 2], im2[:, :, 2], mask_bool, max_levels, filter_size_im,
                                      filter_size_mask)
    return [im1, im2, mask_bool, image], [2, 2, 1, 2]


def display_images(images, representations):
    """
    A helper function for displaying image.
    :param images: list of images to display [Image A, Image B, Mask, blend]
    :param representations: array of representations for the given images 1 is grayscale image and 2 is RGB.
    """
    figure, add = plt.subplots(nrows=2, ncols=2)
    for i in range(4):
        if representations[i] == 1:
            add[i // 2][i % 2].imshow(images[i], cmap=plt.cm.gray)
        else:
            add[i // 2][i % 2].imshow(images[i])
        if i == 0:
            add[0][0].set_title("Image A")
        elif i == 1:
            add[0][1].set_title("Image B")
        elif i == 2:
            add[1][0].set_title("Mask")
        else:
            add[1][1].set_title("Blend")
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
    return image_mat
