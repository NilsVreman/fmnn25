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


if __name__ == '__main__':
    d = np.array([[0,0], [5,0], [8, 3], [5,8], [0,10]]).astype(float)   #Control points
    sp = spline(d)
    steps = 100                                                 #Nbr of steps to evaluate
    results = np.zeros([steps+1, 2])                            #All results for each step

    #Calculate the spline and put the results in a matrix
    for i in range(0, steps+1):
        results[i,:] = sp.value(i/steps)
        print(results[i])

    plt.plot(results[:,0], results[:,1])    #Plot the spline
    plt.plot(d[:, 0], d[:, 1], '*')         #Plot control points
    plt.plot(d[:,0], d[:,1])                #Plot control polygon
    plt.show()
