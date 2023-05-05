import numpy as np
from skimage import morphology
import cv2


def remove_blobs(img, area):
    wh_clean = morphology.remove_small_objects(img.astype(bool), min_size=area, connectivity=1)
    bl_clean = morphology.remove_small_objects(~wh_clean, min_size=area, connectivity=1)
    return ~bl_clean


def mean_filter(img, kernel_size):
    kernel = np.ones((kernel_size, kernel_size)) / kernel_size ** 2
    filtered_img = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    return filtered_img


def boxfilter(img, r):
    return cv2.boxFilter(img, -1, ksize=(r, r), borderType=cv2.BORDER_REPLICATE)


def guidedfilter(img, filtering_img, r, eps):
    """
    Guidance image: img (should be a grayscale/single channel image)
    Filtering input image: filtering_img (should be a grayscale/single channel image)
    Local window radius: r
    Regularization parameter: eps
    """

    N = boxfilter(np.ones_like(img), r)

    mean_input_img = boxfilter(img, r) / N
    mean_filtering_img = boxfilter(filtering_img, r) / N

    mean_imgfilter = boxfilter(img * filtering_img, r) / N
    cov_imgfilter = mean_imgfilter - (mean_input_img * mean_filtering_img)

    mean_imgimg = boxfilter(img * img, r) / N
    var_img = mean_imgimg - (mean_input_img * mean_input_img)

    a = cov_imgfilter / (var_img + eps)
    b = mean_filtering_img - (a * mean_input_img)

    mean_a = boxfilter(a, r) / N
    mean_b = boxfilter(b, r) / N

    return (mean_a * img) + mean_b