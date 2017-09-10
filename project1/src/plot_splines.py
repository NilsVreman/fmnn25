import numpy as np
import matplotlib.pyplot as plt
import spline as spl

class plot_splines:
    def __init__(self):
        self.__sp = []

    def add_spline(self, s):
        """
        Adds a spline object to the list of splines to be plotted.
        Raise: Exception if s is not of instance spline

        s: spline object
        """
        if not s.__class__.__name__ == 'spline':
            raise Exception('Error: Instance not spline')

        self.__sp.append(s)

    def plot_all(self, interpolation=False, de_boor=True, ctrl_pol=True):
        """
        Calculate points on the spline and put them in a matrix.
        Plot the results.

        interpolation: Boolean to plot the interpolation points (if such exists for the spline)
        de_boor: Boolean to plot the control points
        ctrl_pol: Boolean to plot the control polygon
        """
        x_low, x_up = 0, 0
        y_low, y_up = 0, 0

        for i in range(0, len(self.__sp)):
            results = self.__sp[i].get_spline_values()

            # Plots the spline
            plt.plot(results[:,0], results[:,1], label=i)

            d = self.__sp[i].get_ctrl_points()
            interpol_points = self.__sp[i].get_interpolation_points()

            if de_boor:
                # Plots control points
                plt.plot(d[:, 0], d[:, 1], '*')

            if ctrl_pol:
                # Plots control polygon
                plt.plot(d[:,0], d[:,1])

            if interpolation and interpol_points is not None:
                # Plots the interpolation points if such exists
                plt.plot(interpol_points[:,0], interpol_points[:,1], '+')

            # Updates max and min value of the plot
            xmax, ymax = d.max(axis=0)
            xmin, ymin = d.min(axis=0)
            x_low = xmin if xmin < x_low else x_low
            x_up = xmax if xmax > x_up else x_up
            y_low = ymin if ymin < y_low else y_low
            y_up = ymax if ymax > y_up else y_up

        # Sets axes to relate to the max and min control values and also sets a legend
        plt.axis([x_low-2, x_up+2, y_low-1, y_up+1])
        plt.legend()
        plt.show()

if __name__ == '__main__':
    """
    d = np.array([[5,2], [14, 2.1], [26,2], [27,1.5],
                  [27,1.5], [24,1.5], [24,1.5], [27,1.5],
                  [27,1.5], [26,1], [9,1], [9,1], [10,1],
                  [10,0], [5,0], [5,0.7], [5,0.7], [5,0],
                  [0,0], [0,1], [1.5,1.8], [5,2]]).astype(float)
    """
    d = np.array([[0, 0], [0.5, 0.5], [1, 1], [2, 2], [2.5, 2.5], [3, 3], [3.5, 3.5]])
    d1 = np.array([[-2, -1], [0.5, -4], [1, 2], [2, -6], [2.5, 4], [3, -2], [5, 1]])

    s = spl.spline(d, steps=100)
    s1 = spl.spline(d1, steps=100)

    s2 = s + s1
    s3 = s1 + s

    p = plot_splines()
    p.add_spline(s)
    p.add_spline(s1)
    p.add_spline(s2)
    p.add_spline(s3)
    p.plot_all(de_boor=False, ctrl_pol=False)
