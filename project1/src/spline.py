import matplotlib.pyplot as plt

import numpy as np
import scipy as sp
import pylab as pl

class spline:

    def __init__(self):
        pass

    def value(self, d, u, p=3):
        """
        xi: grid points on u. Assume xi sorted and padded
        d:  control points {(x_i, y_i)}_{i=0}^L
        u:  the point in which we want to evaluate the spline
        p:  degree of the spline
        """
        if u == 0: return d[0]
        elif u == 1: return d[-1]

        xi = np.zeros(len(d)-2+2*p)
        xi[-p:] = np.ones(p)
        xi[p:-p] = np.array([ i for i in np.linspace(0, 1, len(d)-2)])

        I = np.searchsorted(xi, u) - 1
        d_i = np.array([d[i-1] for i in range(I-2, I+1+1)])

        for deg_lvl in range(0, p):
            for depth in range(p, deg_lvl, -1):
                alpha = (xi[I + depth - deg_lvl] - u) / (xi[I + depth - deg_lvl] - xi[I + depth - p])
                d_i[depth] = alpha*d_i[depth-1] + (1-alpha)*d_i[depth]

        return d_i[p]

if __name__ == '__main__':
    d = np.array([i for i in np.linspace(0, 1, 6)])
    print(d)

    f = spline()
    for i in np.arange(0, 1.01, 0.1):
        print(f.value(d, i))
        print('')

