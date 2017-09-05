import matplotlib.pyplot as plt
import numpy as np

class spline:

    def __init__(self, d, p=3):
        self.__d = d
        self.__p = p

        #Create knots vector (L = K-2) and pad it with p (degree) repetitions on each side
        xi = np.zeros(len(d)-2+2*p)
        xi[-p:] = np.ones(p)
        xi[p:-p] = np.array([ i for i in np.linspace(0, 1, len(d)-2)])
        self.__xi = xi

    def __find_interval(self, u):
        """
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
        Calculate the points on the spline and put them in a matrix

        steps: Nbr of steps/points to evaluate the spline (resolution)
        return: A vector of point tuples (x,y)
        """

        # Create a matrix to store all result points
        results = np.zeros([steps + 1, len(d[0])])
        # Evaluate for each step
        for i in range(0, steps + 1):
            results[i, :] = sp.value(i / steps)
            print(results[i])

        return results

if __name__ == '__main__':
    # Create control points
    d = np.array([[5,2], [14, 2.1], [26,2], [27,1.5],
                  [27,1.5], [24,1.5], [24,1.5], [27,1.5],
                  [27,1.5], [26,1], [9,1], [9,1], [10,1],
                  [10,0], [5,0], [5,0.7], [5,0.7], [5,0],
                  [0,0], [0,1], [1.5,1.8], [5,2]]).astype(float)
    # Create spline object
    sp = spline(d)
    # Get "step" amount of points that form the spline and put them in a matrix
    results = sp.get_points(steps=100)

    plt.plot(results[:,0], results[:,1])    #Plot the spline
    plt.plot(d[:, 0], d[:, 1], '*')         #Plot control points
    plt.plot(d[:,0], d[:,1])                #Plot control polygon
    plt.axis([-2, 30, -0.7, 3.2])           #Custom axis
    plt.show()
