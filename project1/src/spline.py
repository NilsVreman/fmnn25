import matplotlib.pyplot as plt

import numpy as np
import scipy as sp
import pylab as pl

class spline:

    def __init__(self):
        print("hej")


    def value(self, I, u_ctrl, u, x0):
        """
        I: index of knot interval that contains x [u_I, u_{I+1}]
        u_ctrl: array of ctrl points
        u: the point in which we want to evaluate the spline
        x: array of knot points
        return: value s(u)
        """
        x = [x0[0], x0[0], x0[0], x0[0], x0[1], x0[2], x0[3], x0[4], x0[4], x0[4], x0[4]]
        deg = 3
        d = [u_ctrl[i + I - deg] for i in range(0, deg+1)]

        for deg_lvl in range(1, deg+1):
            for index in range(deg, deg_lvl-1, -1):
                alpha = (u - x[index+I-deg]) / (x[index+1+I-deg_lvl] - x[index+I-deg]) 
                print((x[index+1+I-deg_lvl] - x[index+I-deg]), index+1+I-deg_lvl, index+I-deg)
                d[index] = alpha*d[index] + (1.0 - alpha)*d[index-1]

        return d[deg]

if __name__ =='__main__':
    f = spline()
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([21, 42, 3, 14,19])
    x_ctrl = np.array([1, 3, 5, 7, 8, 2, 4, 6])
    y_ctrl = np.array([4, 6, 14, 53, 20, 40, 23, 19])
    I = 5
    sx = np.zeros([100])
    sy = np.zeros([100])
    for i in range(0, 100):
        sx[i] = f.value(I, x_ctrl, 3+i/100, x)
        sy[i] = f.value(I, y_ctrl, 3+i/100, y)

    plt.plot(sx, sy, 'b^', x_ctrl, y_ctrl, 'gs', x, y, '+')
    plt.show()
