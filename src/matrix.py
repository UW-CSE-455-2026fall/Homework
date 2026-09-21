import numpy as np
import math
import random

class matrix:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.shallow = 0
        self.data = np.zeros((rows, cols), dtype=np.float64)

class data:
    def __init__(self, X, y):
        self.X = X
        self.y = y

def make_matrix(rows, cols):
    return matrix(rows, cols)

def free_matrix(m):
    pass

def free_data(d):
    pass

def copy_matrix(m):
    c = make_matrix(m.rows, m.cols)
    c.data = np.copy(m.data)
    return c

def matrix_mult_matrix(a, b):
    assert a.cols == b.rows
    p = make_matrix(a.rows, b.cols)
    p.data = np.dot(a.data, b.data)
    return p

def axpy_matrix(a, x, y):
    assert x.cols == y.cols
    assert x.rows == y.rows
    p = make_matrix(x.rows, x.cols)
    p.data = a * x.data + y.data
    return p

def transpose_matrix(m):
    t = make_matrix(m.cols, m.rows)
    t.data = np.transpose(m.data)
    return t

def random_matrix(rows, cols, s):
    m = make_matrix(rows, cols)
    m.data = np.random.uniform(-s, s, (rows, cols))
    return m

def random_batch(d, n):
    indices = np.random.choice(d.X.rows, n, replace=False)
    b_X = make_matrix(n, d.X.cols)
    b_y = make_matrix(n, d.y.cols)
    b_X.data = d.X.data[indices]
    b_y.data = d.y.data[indices]
    return data(b_X, b_y)