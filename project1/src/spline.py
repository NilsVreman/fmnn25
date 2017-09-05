import matplotlib.pyplot as plt

import numpy as np
import scipy as sp
import pylab as pl

class spline:

    def __init__(self):
        pass

    def value(self, xi, d, u, p=3):
        """
        xi: grid points on u. Assume xi sorted and padded
        d:  control points {(x_i, y_i)}_{i=0}^L
        u:  the point in which we want to evaluate the spline
        p:  degree of the spline
        """
        I = np.searchsorted(xi, u) - 1
        d_i = np.array([d[i] for i in range(I-2, I+1+1)])

        for deg_lvl in range(0, p):
            for depth in range(p, deg_lvl, -1):
                alpha = (xi[I + depth - deg_lvl] - u) / (xi[I + depth - deg_lvl] - xi[I + depth - p])
                d_i[depth] = alpha*d_i[depth-1] + (1-alpha)*d_i[depth]

        return d_i[p]
        

if __name__ == '__main__':
    xi = np.array([0, 0, 0, 0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0])
    d = np.array([0.1, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9])

    f = spline()
    for i in np.arange(0.01, 0.99, 0.01):
        print(f.value(xi, d, i))
