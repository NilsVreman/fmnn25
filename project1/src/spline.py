import matplotlib.pyplot as plt
import numpy as np

class spline:

    def __init__(self, d, p=3):
        """
        Creates a spline of degree p based on the points d=[[dx1,dy1], ..., [dxn, dyn]]

        d: The control points
        p: The spline degree
        """
        self.__d = d
        self.__p = p

        #Create knots vector (L = K-2) and pad it with p (degree) repetitions on each side
        xi = np.zeros(len(d)-2+2*p)
        xi[-p:] = np.ones(p)
        xi[p:-p] = np.array([ i for i in np.linspace(0, 1, len(d)-2)])
        self.__xi = xi

    def __find_interval(self, u):
        """
        Finds the interval in which u is located. Returns this index and the relevant control points

        u:  The point in which we want to evaluate the spline (0-1)
        return: A tuple: The interval index I, relevant control points d_i
        """
        I = np.searchsorted(self.__xi, u) - 1
        d_i = np.array([self.__d[i-1] for i in range(I-self.__p+1, I+1+1)])
        return I, d_i
    
    def value(self, u):
        """
        Calculate a value on the spline given the control points d and a position u (0-1)
    
        d:  Control points {(x_i, y_i)
        u:  The point in which we want to evaluate the spline (0-1)
        p:  Degree of the spline
        return: value of spline in u
        """
        #Collects the control points and data points from the attributes
        d = self.__d
        p = self.__p
        xi = self.__xi

        #Return endpoints if u = 0 or 1
        if u == 0: return d[0]
        elif u == 1: return d[-1]

        #Find the index of the knot interval where u is located.
        #Put surrounding p+1 control points that are influencing the final value in a vector
        I, d_i = self.__find_interval(u)

        #Evaluation
        for deg_lvl in range(0, p):
            for depth in range(p, deg_lvl, -1):
                alpha = (xi[I + depth - deg_lvl] - u) / (xi[I + depth - deg_lvl] - xi[I + depth - p])
                d_i[depth] = alpha*d_i[depth-1] + (1-alpha)*d_i[depth]

        return d_i[p]

    def get_points(self, steps):
        """
        Calculate points on the spline at "steps" intervals and put them in a matrix.

        steps: Nbr of steps/points to evaluate the spline (resolution)
        return: A vector of point tuples (x,y)
        """
        # Create a matrix to store all result points
        results = np.zeros([steps + 1, len(d[0])])
        # Evaluate for each step
        for i in range(0, steps + 1):
            results[i, :] = self.value(i / steps)
        return results

    def plot(self, steps, de_boor=True, ctrl_pol=True):
        """
        Calculate points on the spline at "steps" intervals and put them in a matrix.
        Plot the results.

        steps: Nbr of steps/points to evaluate the spline (resolution)
        de_boor: Boolean to plot the control points
        ctrl_pol: Boolean to plot the control polygon
        """
        results = self.get_points(steps)
        # Plots the spline
        plt.plot(results[:,0], results[:,1])
        if de_boor:
            # Plots control points
            plt.plot(self.__d[:, 0], self.__d[:, 1], '*')

        if ctrl_pol:
            # Plots control polygon
            plt.plot(self.__d[:,0], self.__d[:,1])

        # Sets axes to relate to the max and min control values.
        xmax, ymax = self.__d.max(axis=0)
        xmin, ymin = self.__d.min(axis=0)
        plt.axis([xmin-2, xmax+2, ymin-1, ymax+1])
        plt.show()



if __name__ == '__main__':
    # Create control points
    d = np.array([[5,2], [14, 2.1], [26,2], [27,1.5],
                  [27,1.5], [24,1.5], [24,1.5], [27,1.5],
                  [27,1.5], [26,1], [9,1], [9,1], [10,1],
                  [10,0], [5,0], [5,0.7], [5,0.7], [5,0],
                  [0,0], [0,1], [1.5,1.8], [5,2]]).astype(float)
    # Create spline object
    sp = spline(d)
    # Plot spline object with control polygon
    sp.plot(steps=100)