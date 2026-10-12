import math
from typing import List

from uwimg import Image, make_image

from src.hw1.process_image import get_pixel, set_pixel, hsv_to_rgb

TWOPI = 6.2831853


# ----------------------------- Resizing -----------------------------

def nn_interpolate(im: Image, x: float, y: float, c: int) -> float:
    # TODO
    # Performs nearest-neighbor interpolation at floating (x, y) for channel c.
    return 0.0


def nn_resize(im: Image, w: int, h: int) -> Image:
    # TODO Fill in (also fix the return line)
    # Uses nearest-neighbor interpolation to resize to (w, h).
    return make_image(1, 1, 1)


def bilinear_interpolate(im: Image, x: float, y: float, c: int) -> float:
    # TODO
    # Performs bilinear interpolation at floating (x, y) for channel c.
    return 0.0


def bilinear_resize(im: Image, w: int, h: int) -> Image:
    # TODO
    # Uses bilinear interpolation to resize to (w, h).
    return make_image(1, 1, 1)


# ----------------------------- Filtering -----------------------------

def l1_normalize(im: Image) -> None:
    # TODO
    # Divide each value by the sum of all values (in-place).
    return


def make_box_filter(w: int) -> Image:
    # TODO
    # Make a (w x w x 1) filter filled with 1s, then l1_normalize.
    return make_image(1, 1, 1)


def convolve_image(im: Image, filt: Image, preserve: int) -> Image:
    # TODO
    # Convolve im with filt. preserve=1 keeps channels, else outputs 1 channel.
    # Must assert (im.c == filt.c) or (filt.c == 1).
    return make_image(1, 1, 1)


def make_highpass_filter() -> Image:
    # TODO
    # Create a 3x3 highpass filter (1 channel).
    return make_image(1, 1, 1)


def make_sharpen_filter() -> Image:
    # TODO
    # Create a 3x3 sharpen filter (1 channel).
    return make_image(1, 1, 1)


def make_emboss_filter() -> Image:
    # TODO
    # Create a 3x3 emboss filter (1 channel).
    return make_image(1, 1, 1)


# Question 2.3.1: Which of these filters should we use preserve when we run our convolution and which ones should we not? Why?
# Answer: TODO

# Question 2.3.2: Do we have to do any post-processing for the above filters? Which ones and why?
# Answer: TODO


def make_gaussian_filter(sigma: float) -> Image:
    # TODO
    # Kernel size is next highest odd integer from 6*sigma (matches C: ((int)(6*sigma))|1)
    # Fill using 2D gaussian, then l1_normalize.
    return make_image(1, 1, 1)


def add_image(a: Image, b: Image) -> Image:
    # TODO
    # Assert same shape. Return a+b.
    return make_image(1, 1, 1)


def sub_image(a: Image, b: Image) -> Image:
    # TODO
    # Assert same shape. Return a-b.
    return make_image(1, 1, 1)


def make_gx_filter() -> Image:
    # TODO
    # Create a 3x3 Sobel Gx filter.
    return make_image(1, 1, 1)


def make_gy_filter() -> Image:
    # TODO
    # Create a 3x3 Sobel Gy filter.
    return make_image(1, 1, 1)


def feature_normalize(im: Image) -> None:
    # TODO
    # Normalize to [0,1] using (x-min)/(max-min); if max==min set all to 0.
    return


def sobel_image(im: Image) -> List[Image]:
    # TODO
    # Return [magnitude, direction] as two 1-channel images.
    return [make_image(1, 1, 1), make_image(1, 1, 1)]


def colorize_sobel(im: Image) -> Image:
    # TODO
    # Use sobel magnitude as S and V, direction as H, then hsv_to_rgb.
    return make_image(1, 1, 1)


# EXTRA CREDIT: Median filter

"""
def apply_median_filter(im: Image, kernel_size: int) -> Image:
    return make_image(1, 1, 1)
"""

# SUPER EXTRA CREDIT: Bilateral filter

"""
def apply_bilateral_filter(im: Image, sigma1: float, sigma2: float) -> Image:
    return make_image(1, 1, 1)
"""