import os
import math
import numpy as np
from uwimg import Image, load_image, make_image
from src.hw3.harris_image import Point, structure_matrix, cornerness_response
from src.hw3.panorama_image import Match, project_point, compute_homography, make_point

EPS = 0.005
PRINT_PASSES = True

tests_total = 0
tests_fail = 0
_current_test_name = "unknown"

def avg_diff(a: Image, b: Image) -> float:
    diff = 0.0
    ad = a.data.reshape(-1)
    bd = b.data.reshape(-1)
    for i in range(a.w * a.h * a.c):
        diff += float(bd[i]) - float(ad[i])
    return diff / (a.w * a.h * a.c)

def feature_normalize2(im: Image) -> None:
    flat = im.data.reshape(-1)
    mn = float(flat[0])
    mx = float(flat[0])
    for i in range(im.w * im.h * im.c):
        v = float(flat[i])
        if v > mx:
            mx = v
        if v < mn:
            mn = v

    denom = mx - mn
    if denom == 0.0:
        return
    for i in range(im.w * im.h * im.c):
        flat[i] = (float(flat[i]) - mn) / denom

def within_eps(a: float, b: float, eps: float) -> bool:
    return (a - eps) < b and b < (a + eps)

def same_point(p: Point, q: Point, eps: float) -> bool:
    return within_eps(p.x, q.x, eps) and within_eps(p.y, q.y, eps)

def same_matrix(m: np.ndarray, n: np.ndarray) -> bool:
    if m.shape != n.shape:
        return False
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if not within_eps(m[i, j], n[i, j], EPS):
                return False
    return True

def same_image(a: Image, b: Image, eps: float) -> bool:
    if a.w != b.w or a.h != b.h or a.c != b.c:
        # print(f"Expected {b.w} x {b.h} x {b.c} image, got {a.w} x {a.h} x {a.c}")
        return False

    ad = a.data.reshape(-1)
    bd = b.data.reshape(-1)
    for i in range(a.w * a.h * a.c):
        thresh = (abs(float(bd[i])) + abs(float(ad[i]))) * eps / 2.0
        if thresh > eps:
            eps = thresh
        if not within_eps(float(ad[i]), float(bd[i]), eps):
            print(f"The value should be {bd[i]}, but it is {ad[i]}! ")
            return False
    return True

def TEST(cond: bool, expr: str) -> None:
    global tests_total, tests_fail
    tests_total += 1
    if not cond:
        tests_fail += 1
        import inspect
        fr = inspect.currentframe().f_back
        rel = os.path.relpath(fr.f_code.co_filename)
        line = fr.f_lineno
        print(f"failed: [{_current_test_name}] testing [{expr}] in {rel}, line {line}")
    else:
        if PRINT_PASSES:
            print(f"passed: [{_current_test_name}] testing [{expr}]")


# ---------------------------------------------------------
# HOMEWORK 3 TESTS
# ---------------------------------------------------------

def test_structure() -> None:
    global _current_test_name
    _current_test_name = "test_structure"

    im = load_image("data/dogbw.png")
    s = structure_matrix(im, 2)
    feature_normalize2(s)
    gt = load_image("figs/structure.png")
    TEST(same_image(s, gt, EPS), "same_image(s, gt, EPS)")

def test_cornerness() -> None:
    global _current_test_name
    _current_test_name = "test_cornerness"

    im = load_image("data/dogbw.png")
    s = structure_matrix(im, 2)
    c = cornerness_response(s)
    feature_normalize2(c)
    gt = load_image("figs/response.png")
    TEST(same_image(c, gt, EPS), "same_image(c, gt, EPS)")

def test_projection() -> None:
    global _current_test_name
    _current_test_name = "test_projection"

    # make_translation_homography in numpy
    H = np.eye(3, dtype=np.float64)
    H[0, 2] = 12.4
    H[1, 2] = -3.2

    TEST(same_point(project_point(H, make_point(0,0)), make_point(12.4, -3.2), EPS),
         "same_point(project_point(H, make_point(0,0)), make_point(12.4, -3.2), EPS)")

    # make_identity_homography with custom values
    H = np.eye(3, dtype=np.float64)
    H[0, 0] = 1.32;  H[0, 1] = -1.12; H[0, 2] = 2.52
    H[1, 0] = -.32;  H[1, 1] = -1.2;  H[1, 2] = .52
    H[2, 0] = -3.32; H[2, 1] = 1.87;  H[2, 2] = .112

    p = project_point(H, make_point(3.14, 1.59))
    TEST(same_point(p, make_point(-0.66544, 0.326017), EPS),
         "same_point(p, make_point(-0.66544, 0.326017), EPS)")

def test_compute_homography() -> None:
    global _current_test_name
    _current_test_name = "test_compute_homography"

    m = [Match() for _ in range(4)]
    m[0].p = make_point(0,0);     m[0].q = make_point(10,10)
    m[1].p = make_point(3,3);     m[1].q = make_point(13,13)
    m[2].p = make_point(-1.2,-3.4); m[2].q = make_point(8.8,6.6)
    m[3].p = make_point(9,10);    m[3].q = make_point(19,20)

    H = compute_homography(m, 4)

    d10 = np.eye(3, dtype=np.float64)
    d10[0, 2] = 10.0
    d10[1, 2] = 10.0
    TEST(same_matrix(H, d10), "same_matrix(H, d10)")

    m[0].p = make_point(7.2,1.3);  m[0].q = make_point(10,10.9)
    m[1].p = make_point(3,3);      m[1].q = make_point(1.3,7.3)
    m[2].p = make_point(-.2,-3.4); m[2].q = make_point(.8,2.6)
    m[3].p = make_point(-3.2,2.4); m[3].q = make_point(1.5,-4.2)

    H = compute_homography(m, 4)

    Hp = np.eye(3, dtype=np.float64)
    Hp[0, 0] = -0.1328042; Hp[0, 1] = -0.2910411; Hp[0, 2] = 0.8103200
    Hp[1, 0] = -0.0487439; Hp[1, 1] = -1.3077799; Hp[1, 2] = 1.4796660
    Hp[2, 0] = -0.0788730; Hp[2, 1] = -0.3727209; Hp[2, 2] = 1.0000000

    TEST(same_matrix(H, Hp), "same_matrix(H, Hp)")

def test_hw3() -> None:
    test_structure()
    test_cornerness()
    test_projection()
    test_compute_homography()
    print(f"{tests_total} tests, {tests_total - tests_fail} passed, {tests_fail} failed")

if __name__ == "__main__":
    test_hw3()