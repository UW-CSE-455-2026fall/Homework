import os
import sys
import numpy as np
import inspect

from src.matrix import *
from src.hw4.classifier import *

EPS = 0.005
tests_total = 0
tests_fail = 0

def within_eps(a, b, eps):
    return (a - eps) < b and b < (a + eps)

def same_matrix(m, n):
    if m.rows != n.rows or m.cols != n.cols:
        return 0

    diff = np.abs(m.data - n.data)
    if np.any(diff >= EPS):
        return 0
    return 1

def TEST(condition):
    """
    Replicates the exact behavior of the C macro:
    #define TEST(ex) do { ... fprintf(stderr, "TEST OK: %s\n", #ex); ... }
    Writes directly to stderr and captures the exact expression string.
    """
    global tests_total, tests_fail
    tests_total += 1

    frame = inspect.currentframe().f_back
    context = inspect.getframeinfo(frame).code_context[0].strip()

    start = context.find('TEST(')
    if start != -1:
        start += 5
        end = context.rfind(')')
        expr_str = context[start:end].strip()
    else:
        expr_str = "expression"

    if not condition:
        tests_fail += 1
        sys.stderr.write(f"FAILED: {expr_str}\n")
    else:
        sys.stderr.write(f"TEST OK: {expr_str}\n")

    sys.stderr.flush()

def load_matrix(fname):
    if not os.path.exists(fname):
        print(f"Error: Cannot find test data file {fname}.")
        sys.exit(1)

    with open(fname, "rb") as f:
        rows = np.fromfile(f, dtype=np.int32, count=1)[0]
        cols = np.fromfile(f, dtype=np.int32, count=1)[0]
        data = np.fromfile(f, dtype=np.float64, count=rows * cols).reshape((rows, cols))

        m = make_matrix(rows, cols)
        m.data = data
        return m

def save_matrix(m, fname):
    with open(fname, "wb") as f:
        np.array([m.rows], dtype=np.int32).tofile(f)
        np.array([m.cols], dtype=np.int32).tofile(f)
        m.data.astype(np.float64).tofile(f)

def test_activate_matrix():
    a = load_matrix("data/test/a.matrix")
    truth_alog = load_matrix("data/test/alog.matrix")
    truth_arelu = load_matrix("data/test/arelu.matrix")
    truth_alrelu = load_matrix("data/test/alrelu.matrix")
    truth_asoft = load_matrix("data/test/asoft.matrix")

    alog = copy_matrix(a)
    activate_matrix(alog, 'LOGISTIC')

    arelu = copy_matrix(a)
    activate_matrix(arelu, 'RELU')

    alrelu = copy_matrix(a)
    activate_matrix(alrelu, 'LRELU')

    asoft = copy_matrix(a)
    activate_matrix(asoft, 'SOFTMAX')

    TEST(same_matrix(truth_alog, alog))
    TEST(same_matrix(truth_arelu, arelu))
    TEST(same_matrix(truth_alrelu, alrelu))
    TEST(same_matrix(truth_asoft, asoft))

def test_gradient_matrix():
    a = load_matrix("data/test/a.matrix")
    y = load_matrix("data/test/y.matrix")
    truth_glog = load_matrix("data/test/glog.matrix")
    truth_grelu = load_matrix("data/test/grelu.matrix")
    truth_glrelu = load_matrix("data/test/glrelu.matrix")
    truth_gsoft = load_matrix("data/test/gsoft.matrix")

    glog = copy_matrix(a)
    grelu = copy_matrix(a)
    glrelu = copy_matrix(a)
    gsoft = copy_matrix(a)

    gradient_matrix(y, 'LOGISTIC', glog)
    gradient_matrix(y, 'RELU', grelu)
    gradient_matrix(y, 'LRELU', glrelu)
    gradient_matrix(y, 'SOFTMAX', gsoft)

    TEST(same_matrix(truth_glog, glog))
    TEST(same_matrix(truth_grelu, grelu))
    TEST(same_matrix(truth_glrelu, glrelu))
    TEST(same_matrix(truth_gsoft, gsoft))

def test_layer():
    a = load_matrix("data/test/a.matrix")
    w = load_matrix("data/test/w.matrix")
    dw = load_matrix("data/test/dw.matrix")
    v = load_matrix("data/test/v.matrix")
    delta = load_matrix("data/test/delta.matrix")

    truth_dx = load_matrix("data/test/truth_dx.matrix")
    truth_v = load_matrix("data/test/truth_v.matrix")
    truth_dw = load_matrix("data/test/truth_dw.matrix")

    updated_dw = load_matrix("data/test/updated_dw.matrix")
    updated_w = load_matrix("data/test/updated_w.matrix")
    updated_v = load_matrix("data/test/updated_v.matrix")

    truth_out = load_matrix("data/test/out.matrix")

    l = make_layer(64, 16, 'LRELU')
    l.w = w
    l.dw = dw
    l.v = v

    out = forward_layer(l, a)
    TEST(same_matrix(truth_out, out))

    dx = backward_layer(l, delta)
    TEST(same_matrix(truth_v, v))
    TEST(same_matrix(truth_dw, l.dw))
    TEST(same_matrix(truth_dx, dx))

    update_layer(l, .01, .9, .01)
    TEST(same_matrix(updated_dw, l.dw))
    TEST(same_matrix(updated_w, l.w))
    TEST(same_matrix(updated_v, l.v))

def test_hw4():
    test_activate_matrix()
    test_gradient_matrix()
    test_layer()
    print("%d tests, %d passed, %d failed" % (tests_total, tests_total - tests_fail, tests_fail))

if __name__ == "__main__":
    test_hw4()