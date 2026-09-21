import math
from uwimg import Image, make_image


def get_pixel(im: Image, x: int, y: int, c: int) -> float:
    # TODO Fill this in
    return 0.0


def set_pixel(im: Image, x: int, y: int, c: int, v: float) -> None:
    # TODO Fill this in
    return


def copy_image(im: Image) -> Image:
    copy = make_image(im.w, im.h, im.c)
    # TODO Fill this in
    return copy


def rgb_to_grayscale(im: Image) -> Image:
    assert im.c == 3
    gray = make_image(im.w, im.h, 1)
    # TODO Fill this in
    return gray


def shift_image(im: Image, c: int, v: float) -> None:
    # TODO Fill this in
    return


def clamp_image(im: Image) -> None:
    # TODO Fill this in
    return


# These might be handy
def three_way_max(a: float, b: float, c: float) -> float:
    return (a if a > b else b) if ((a if a > b else b) > c) else c


def three_way_min(a: float, b: float, c: float) -> float:
    return (a if a < b else b) if ((a if a < b else b) < c) else c


def rgb_to_hsv(im: Image) -> None:
    # TODO Fill this in
    return


def hsv_to_rgb(im: Image) -> None:
    # TODO Fill this in
    return