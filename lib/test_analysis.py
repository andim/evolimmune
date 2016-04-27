import numpy as np
import numpy.testing as npt
from analysis import *

def test_polygon_from_boundary():
    # |\
    # | \
    # |__\
    xs = np.array([0.2, 0.1, 0.0])
    ys = 0.2-xs
    target = np.array([[ 0.2,  0. ],
                       [ 0.1,  0.1],
                       [ 0. ,  0.2],
                       [ 0. ,  0. ]])
    out = polygon_from_boundary(xs, ys)
    npt.assert_array_equal(out, target)

    # boundary along xmin -> remove
    xs = np.array([0.2, 0.1, 0.0, -0.1])
    ys = 0.2-xs
    out = polygon_from_boundary(xs, ys)
    npt.assert_array_equal(out, target)

    # |--/ 
    # | / 
    # |/
    xs = np.array([0.0, 0.5, 1.0])
    ys = xs
    target = np.array([[ 0.0,  0.0],
                       [ 0.5,  0.5],
                       [ 1.0,  1.0],
                       [ 0.0,  1.0]])
    out = polygon_from_boundary(xs, ys)
    npt.assert_array_equal(out, target)

    # x larger than xmax -> set to bounds
    xs = np.array([0.0, 0.5, 1.1])
    out = polygon_from_boundary(xs, ys)
    npt.assert_array_equal(out, target)

    # only a single value -> vspan
    out = polygon_from_boundary([0.5], [0.0])
    target = np.array([[0.5, 0.0],
                       [0.5, 1.0],
                       [0.0, 1.0],
                       [0.0, 0.0]])
    npt.assert_array_equal(out, target)


if __name__ == '__main__':
    test_polygon_from_boundary()
    npt.run_module_suite()
