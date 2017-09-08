import numpy as np
import matplotlib.pyplot as plt
import spline as spl

class plot_splines:
    def __init__(self):
        self.__sp = []

    def add_spline(self, s):
        """
        s: spline object
        """
        if not isinstance(s, spl.spline):
            raise Exception('Error: Instance not spline')
        
        self.__sp.append(s)

    def plot_all(self, points = None, steps=100, de_boor=True, ctrl_pol=True):
        """
        Calculate points on the spline at "steps" intervals and put them in a matrix.
        Plot the results.

        steps: Nbr of steps/points to evaluate the spline (resolution)
        de_boor: Boolean to plot the control points
        ctrl_pol: Boolean to plot the control polygon
        """
        x_low, x_up = 0, 0
        y_low, y_up = 0, 0

        for i in range(0, len(self.__sp)):
            results = self.__sp[i].get_points(steps)
            # Plots the spline
            plt.plot(results[:,0], results[:,1])
            d = self.__sp[i].get_ctrl_points()
            print(d)

            if de_boor:
                # Plots control points
                plt.plot(d[:, 0], d[:, 1], '*')

            if ctrl_pol:
                # Plots control polygon
                plt.plot(d[:,0], d[:,1])
            if points is not None:
                plt.plot(points[0,:], points[1,:], '+')

            # Sets axes to relate to the max and min control values.
            xmax, ymax = d.max(axis=0)
            xmin, ymin = d.min(axis=0)
            x_low = xmin if xmin < x_low else x_low
            x_up = xmax if xmax > x_up else x_up
            y_low = ymin if ymin < y_low else y_low
            y_up = ymax if ymax > y_up else y_up

        plt.axis([x_low-2, x_up+2, y_low-1, y_up+1])
        plt.show()

if __name__ == '__main__':
    """
    d = np.array([[5,2], [14, 2.1], [26,2], [27,1.5],
                  [27,1.5], [24,1.5], [24,1.5], [27,1.5],
                  [27,1.5], [26,1], [9,1], [9,1], [10,1],
                  [10,0], [5,0], [5,0.7], [5,0.7], [5,0],
                  [0,0], [0,1], [1.5,1.8], [5,2]]).astype(float)
    """
    d = np.array([[0,0], [0.5, -1], [1, 2], [2, -2], [2.5, 2], [3, -2], [3.5, 1], [4,0], [5, 1], [6, -1], [7, 1], [8, -1], [9,1], [10,-1], [11, 0]])
    #d1 = np.array([[0, 4], [0.5, 5], [1, 6], [2, 6], [3, 6], [3.5, 5], [4, 4]])

    s = spl.spline(d)
    #s1 = spl.spline(d1)

    p = plot_splines()

    p.add_spline(s)
    #p.add_spline(s1)
    p.plot_all(steps=100)
