import numpy as np
import scipy as sp

class spline:

    def __init__(self, d, xi=None, p=3):
        """
        Creates a spline of degree p based on the points d=[[dx1,dy1], ..., [dxn, dyn]]

        d: The control points
        p: The spline degree
        """
        self.__d = d
        self.__p = p

        if xi is None:
            #Create knots vector (L = K-2) and pad it with p (degree) repetitions on each side
            xi = np.zeros(len(d)-2)
            xi = np.array([ i for i in np.linspace(0, 1, len(d)-2)])
            self.__u_knots=xi[p:-p] # Knots without padding (Might not be needed)
            self.__xi = xi
            print(xi)
        else: self.__xi = xi

    def __find_interval(self, u):
        """
        Finds the interval in which u is located. Returns this index and the relevant control points

        u:  The point in which we want to evaluate the spline (0-1)
        return: A tuple: The interval index I, relevant control points d_i
        """
        I = np.searchsorted(self.__xi, u)
        if u not in self.__xi:
            I -= 1

        print(I)
        d_i = np.array([self.__d[i] for i in range(I-self.__p+1, I+1+1)])
        print(d_i)
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
                print(I + depth - p, I + depth - deg_lvl, u, xi[I], xi[I+1])
                alpha = (xi[I + depth - deg_lvl] - u) / (xi[I + depth - deg_lvl] - xi[I + depth - p])
                d_i[depth] = alpha*d_i[depth-1] + (1-alpha)*d_i[depth]

        print(d_i[p])
        return d_i[p]

    def interpolate(self, xi, points):
        if len(points[0]) < 4:
            raise valueError("Need atleast 4 points")
        if len(points[0]) != (len(xi) - 2):
            raise valueError("Number of points and knots doesnt match")
        
        L = len(points[0])
        NMat = np.zeros((L,L))
        
        for i  in range(0, L):
            G_abscissae = (xi[i] + xi[i+1] + xi[i+2])/3
            for j in range(0, L):
                N = self.__get_N_base(j, G_abscissae, xi, 3)
                NMat[i,j] = N
        
        dx = sp.linalg.solve_banded((3,3),NMat,points[0])
        dy = sp.linalg.solve_banded((3,3),NMat,points[1])
        
        print([dx,dy])
        
        
    def test(self):
        print("ok")
        xi = np.array([0,0,0,0.25,0.5,0.75,1,1,1])
        points = np.array([[1,2,3,4,5,6,7],[7,6,5,4,3,2,1]])
        
        self.interpolate(xi, points)
                

                
    def get_points(self, steps):
        """
        Calculate points on the spline at "steps" intervals and put them in a matrix.

        steps: Nbr of steps/points to evaluate the spline (resolution)
        return: A vector of point tuples (x,y)
        """
        # Create a matrix to store all result points
        results = np.zeros([steps, len(self.__d[0])])
        # Evaluate for each step
        counter = 0
        for i in np.linspace(self.__xi[2], self.__xi[-3]-1/steps, steps): #range(0, steps + 1):
            try:
                results[counter, :] = self.value(i)
                counter +=1
            except Exception as e:
                print(e)
                pass

        print(results)
        return results

    def get_ctrl_points(self):
        return self.__d

    def get_knots(self):
        return self.__u_knots

    def getN_i_k(self, xi, i):
        """
        Sets the knots and index

        u:  The point in which we want to evaluate the value of N
        return: the function to evaluate basis function N in u
        """
        self.__base_knots = xi
        self.__i = i
        return self.N_base

    def N_base(self,u):
        xi = self.__base_knots # The knots to create base functions at
        i = self.__i # The index of the base function
        return self.__get_N_base(i, u, xi, self.__p)

    def __get_N_base(self, i ,u, xi, k):
        """
        Calculate the N basis function by recursion

        i:  index of basis function
        u:  The point in which we want to evaluate the value of N
        xi: The knots that we want to create a spline through
        k:  Degree of the spline
        return: value of basis function N in u
        """
        try:
            if k == 0:
                if xi[i - 1] == xi[i]:
                    return 0
                elif (u >= xi[i-1] and u < xi[i]):
                    return 1
                else:
                    return 0
            else:
                return (self.__getMultVal(u - xi[i-1], xi[i + k - 1] - xi[i-1]) * self.__get_N_base(i, u, xi, k - 1) +
                        self.__getMultVal(xi[i+k] - u, xi[i+k] - xi[i]) * self.__get_N_base(i+1, u, xi, k - 1))
        except IndexError:
            return 0.0
        

    """
    Redefines divde by zero to 0/0 = 0
    """
    def __getMultVal(self,t,n):
        if(n == 0.0):
            return 0.0
        return t/n
