import matplotlib.pyplot as plt
import numpy as np

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
        #Return endpoints if u = 0 or 1
        if u == 0: return d[0]
        elif u == 1: return d[-1]

        #Create knots vector (L = K-2) and pad it with p (degree) repetitions on each side
        xi = np.zeros(len(d)-2+2*p)
        xi[-p:] = np.ones(p)
        xi[p:-p] = np.array([ i for i in np.linspace(0, 1, len(d)-2)])

        #Find the index of the knot interval where u is located.
        #Put surrounding p+1 control points that are influencing the final value in a vector
        I = np.searchsorted(xi, u) - 1
        d_i = np.array([d[i+I-p] for i in range(0, p+1)])

        #Evaluation
        for deg_lvl in range(0, p):
            for depth in range(p, deg_lvl, -1):
                alpha = (xi[I + depth - deg_lvl] - u) / (xi[I + depth - deg_lvl] - xi[I + depth - p])
                d_i[depth] = alpha*d_i[depth-1] + (1-alpha)*d_i[depth]

        return d_i[p]

if __name__ == '__main__':

    sp = spline()
    d = np.array([[0,0], [5,0], [5,5], [0,10]]).astype(float)   #Control points
    steps = 100                                                 #Nbr of steps to evaluate
    results = np.zeros([steps+1, 2])                            #All results for each step

    #Calculate the spline
    for i in range(0, steps+1):
        results[i,:] = sp.value(d, i/steps)
        print(results[i])

    plt.plot(results[:,0], results[:,1])    #Plot the spline
    plt.plot(d[:, 0], d[:, 1], '*')         #Plot control points
    plt.plot(d[:,0], d[:,1])                #Plot control polygon
    plt.show()