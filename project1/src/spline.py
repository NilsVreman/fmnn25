import numpy as np
import scipy.linalg as sp
import plot_splines as ps

class spline:

    def __init__(self, d, steps=100, xi=None, p=3):
        """
        Creates a spline of degree p based on the points d=[[dx1,dy1], ..., [dxn, dyn]]

        d: The control points
        p: The spline degree
        """
        self.__d = d.astype(float) # OBS! need to Type cast all values in d as float because value need to be able to calculate with float
        self.__steps = steps
        self.__p = p
        self.__interpolation_points = None

        if xi is None:
            #Create knots vector (K=L+2)
            xi1 = np.zeros(p-1)
            xi2 = np.array([ i for i in np.linspace(0, 1, len(d)-2)])
            xi3 = np.ones(p - 1)

            self.__xi = np.hstack([xi1, xi2, xi3])

        else: self.__xi = xi

    def __call__(self, interpolation, de_boor=True, ctrl_pol=True):
        """
        Plots the spline based on the boolean interpolation. If interpolation is True: Use the control points as interpolation points and calculate the new control points such that the spline goes through the interpolation points.

        interpolation:  Boolean that decides if our control points are interpolation points (True) or not (False)
        de_boor:        Boolean that decides if we want the control points plotted or not.
        ctrl_pol:       Boolean that decides if we want the control polygon plotted or not
        """
        if interpolation:
            self.interpolate()

        p = ps.plot_splines()
        p.add_spline(self)
        p.plot_all(interpolation, de_boor, ctrl_pol)


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

        d:  Control points {(x_i, y_i)}
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
        print(d_i)
        #Evaluation
        for deg_lvl in range(0, p):
            for depth in range(p, deg_lvl, -1):
                alpha = (xi[I + depth - deg_lvl] - u) / (xi[I + depth - deg_lvl] - xi[I + depth - p])
                d_i[depth] = alpha*d_i[depth-1] + (1-alpha)*d_i[depth]

        return d_i[p]

    def interpolate(self):
        """
        Finds control points d from already given points (interpolation points)

        return: control points, d
        """
        xi = self.__xi
        self.__interpolation_points = np.array(self.__d)
        points = self.__d.T

        #Check that we can compute the Interpolation
        if len(points[0]) < 4:
            raise valueError("Need atleast 4 points")
        if len(points[0]) != (len(xi) - 2):
            raise valueError("Number of points and knots doesnt match")

        L = len(points[0])
        NMat = np.zeros((L,L))

        #Some comments about the for loop
        for i  in range(0, L):
            G_abscissae = (xi[i] + xi[i+1] + xi[i+2])/3
            for j in range(0, L):
                N = self.__get_N_base(j, G_abscissae, xi, 3)
                NMat[i,j] = N

        dx = sp.solve(NMat,points[0])
        dy = sp.solve(NMat,points[1])


        #Transform the control points into a [nx2] matrix
        self.__d = np.array([dx, dy]).T
        
        return self.__d

    def __is_cyclic(self):
        # Returns boolean wether spline is cyclic or not

        min = self.__d[0, 0]
        max = self.__d[0, 0]
        for x in self.__d[:, 0]:
            if x >= max:
                max = x
            else:
                return True
        return False

    def __get_min_max(self):
        # Find interval min and max in x of the spline

        min = self.__d[0, 0]
        max = self.__d[0, 0]
        for x in self.__d[:, 0]:
            if x >= max:
                max = x
        return min, max

    def __get_u_min_max(self, min, max):
        u_min = 0
        for i in range(0, self.__steps + 1):
            if self.value(i / self.__steps)[0] >= min:
                u_min = i / self.__steps
                break

        u_max = 0
        for i in range(0, self.__steps + 1):
            if self.value(i / self.__steps)[0] >= max:
                u_max = i / self.__steps
                break

        return u_min, u_max

    def __add__(self, sp2):
        """
        calculates the addition between self and spline

        return: the new spline which is an addition between self and sp2
        """
        if not isinstance(sp2, spline): raise Exception("Spline 2 is not a spline")
        if self.__is_cyclic(): raise Exception("Spline 1 is cyclic")
        if sp2.__is_cyclic(): raise Exception("Spline 2 is cyclic")

        # Get interval in x of both splines
        min1, max1 = self.__get_min_max()
        min2, max2 = sp2.__get_min_max()

        # Get intersection interval of both splines
        min3 = max(min1, min2)
        max3 = min(max1, max2)

        # Get intersection interval of both splines in u instead of x
        u_min3_sp1, u_max3_sp1 = self.__get_u_min_max(min3, max3)
        u_min3_sp2, u_max3_sp2 = sp2.__get_u_min_max(min3, max3)

        # Create a vector with u values in the interval with "steps" many points
        u_vec_sp1 = None
        for i in np.linspace(0, 1, self.__steps + 1):
            if i < u_min3_sp1:
                continue
            u_vec_sp1 = np.linspace(i, u_max3_sp1, self.__steps + 1)
            break

        u_vec_sp2 = None
        for i in np.linspace(0, 1, self.__steps + 1):
            if i < u_min3_sp2:
                continue
            u_vec_sp2 = np.linspace(i, u_max3_sp2, self.__steps + 1)
            break

        #Find which spline has the min x value. Set left_most_sp to that.
        left_most_sp = None
        right_most_sp = None
        if self.value(u_vec_sp1[0])[0] <= sp2.value(u_vec_sp2[0])[0]:
            left_most_sp = self
            right_most_sp = sp2
        else:
            temp = np.array(u_vec_sp1)
            left_most_sp = sp2
            right_most_sp = self
            u_vec_sp1 = u_vec_sp2
            u_vec_sp2 = temp

        # Fix u_vec_sp2 so that for every x, f(u1) = f(u2), it aligns with u_vec_sp1
        for i in range(0, self.__steps):
            divide = 2
            while abs(left_most_sp.value(u_vec_sp1[i])[0] - right_most_sp.value(u_vec_sp2[i])[0]) > 1/self.__steps:
                if left_most_sp.value(u_vec_sp1[i])[0] < right_most_sp.value(u_vec_sp2[i])[0]:
                    while u_vec_sp2[i] - abs(u_vec_sp1[i] - u_vec_sp2[i]) / divide < 0:
                        divide += 1
                    u_vec_sp2[i] -= abs(u_vec_sp1[i] - u_vec_sp2[i]) / divide

                elif left_most_sp.value(u_vec_sp1[i])[0] > right_most_sp.value(u_vec_sp2[i])[0]:
                    while u_vec_sp2[i] + abs(u_vec_sp1[i] - u_vec_sp2[i]) / divide > 1:
                        divide += 1
                    u_vec_sp2[i] += abs(u_vec_sp1[i] - u_vec_sp2[i]) / divide

        # Create a matrix to store all result points
        results = np.zeros([self.__steps + 1, len(self.__d[0])])

        for i in range(0, self.__steps + 1):
            results[i, 0] = left_most_sp.value(u_vec_sp1[i])[0]
            results[i, 1] = left_most_sp.value(u_vec_sp1[i])[1] + right_most_sp.value(u_vec_sp2[i])[1]

        sp3 = spline(results, self.__steps)
        sp3.interpolate()

        return sp3

    def get_spline_values(self):
        """
        Calculate points on the spline at self.steps intervals and put them in a matrix.

        return: A vector of point tuples (x,y)
        """
        # Create a matrix to store all result points
        results = np.zeros([self.__steps + 1, len(self.__d[0])])

        # Evaluate for each step
        for i in range(0, self.__steps + 1):
            results[i, :] = self.value(i / self.__steps)
        return results

    def get_interpolation_points(self):
        # Return the interpolation points
        return self.__interpolation_points

    def get_ctrl_points(self):
        # Return the control points
        return self.__d

    def get_knots(self):
        # Return the grid points
        return self.__xi

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
            if self.__get_U(xi, i - 1) == self.__get_U(xi, i):
                return 0
            elif (u >= self.__get_U(xi,i-1) and u < self.__get_U(xi, i)):
                return 1
            else:
                return 0
        else:
            return (self.__getMultVal(u - self.__get_U(xi, i-1), self.__get_U(xi, i + k - 1) - self.__get_U(xi, i-1)) * self.__get_N_base(i, u, xi, k - 1) +
                    self.__getMultVal(self.__get_U(xi, i+k) - u, self.__get_U(xi,i+k) - self.__get_U(xi, i)) * self.__get_N_base(i+1, u, xi, k - 1))


    def __get_U(self, xi, i):
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

if __name__ == '__main__':
    d = np.array([[0,0], [1, 1], [2, -1], [3, 2], [4, -2], [5, 2], [6,-1], [7,1], [8,0]])
    d1 = np.array([[0,0], [1, 1], [2, -1], [3, 2], [4, -2], [5, 2], [6,-1], [7,1], [8,0]])
    s = spline(d)
    s1 = spline(d1)
    s(False)
    s1(True)
