import numpy as np
import scipy as sp
import plot_splines as ps

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
            #Create knots vector (K=L+2)
            xi1 = np.zeros(p-1)
            xi2 = np.array([ i for i in np.linspace(0, 1, len(d)-2)])
            xi3 = np.ones(p - 1)
            
            self.__xi = np.hstack([xi1, xi2, xi3])
            
        else: self.__xi = xi

    def __find_interval(self, u):
        """
        Finds the interval in which u is located. Returns this index and the relevant control points

        u:  The point in which we want to evaluate the spline (0-1)
        return: A tuple: The interval index I, relevant control points d_i
        """
        I = np.searchsorted(self.__xi, u) - 1
        d_i = np.array([self.__d[i] for i in range(I-self.__p+1, I+1+1)])
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
        
        
        dx = sp.linalg.solve(NMat,points[0])
        dy = sp.linalg.solve(NMat,points[1])
        
        
        d = np.array([[0.0 for x in range(2)] for y in range(len(dx))]) 
        for x in range(0,len(dx)):
            d[x] = np.array([dx[x],dy[x]])
  
        spli = spline(d, xi, 3)
        p = ps.plot_splines()
        p.add_spline(spli)
        
        p.plot_all(points)
        
        
    def test(self):
        xi = np.linspace(0,1.,8)
        xi = np.hstack([0,0, xi, 1,1])
        points = np.array([[-8.18548387, -7.13709677, -2.82258065, -2.37903226,  1.00806452, 2.41935484,  4.87903226,  5.88709677,  6.93548387,  7.41935484], [4.18410042, -3.45188285,  5.75313808, -2.71966527,  8.21129707, -3.66108787,  4.55020921, -0.31380753,  7.4790795 , -3.9748954]])
        
        self.interpolate(xi, points)
                

                
    def get_points(self, steps):
        """
        Calculate points on the spline at "steps" intervals and put them in a matrix.

        steps: Nbr of steps/points to evaluate the spline (resolution)
        return: A vector of point tuples (x,y)
        """
        # Create a matrix to store all result points
        results = np.zeros([steps + 1, len(self.__d[0])])
        # Evaluate for each step
        for i in range(0, steps + 1):
            results[i, :] = self.value(i / steps)
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
        
        if k == 0:
            if self.getU(xi, i - 1) == self.getU(xi, i):
                return 0
            elif (u >= self.getU(xi,i-1) and u < self.getU(xi, i)):
                return 1
            else:
                return 0
        else:
            return (self.__getMultVal(u - self.getU(xi, i-1), self.getU(xi, i + k - 1) - self.getU(xi, i-1)) * self.__get_N_base(i, u, xi, k - 1) +
                    self.__getMultVal(self.getU(xi, i+k) - u, self.getU(xi,i+k) - self.getU(xi, i)) * self.__get_N_base(i+1, u, xi, k - 1))
        
        
    def getU(self, xi, i):
        if i < 0:
            return 0
        elif i >= len(xi):
            return 10
        else:
            return xi[i]   
    
    def __getMultVal(self,t,n):
        """
        Redefines divde by zero to 0/0 = 0
        """
        if(n == 0.0):
            return 0.0
        return t/n
