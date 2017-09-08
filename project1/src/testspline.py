import unittest
from spline import spline
import numpy as np
import matplotlib.pyplot as plt

class TestSpline(unittest.TestCase):
    '''
    Test if S(u) = sum(d_i*N_i)
    Also plots it
    '''
    def test_s_equals_sum_dN(self):
        d =np.array([[0,0], [0.5, -1], [1, 2], [2, -2], [2.5, 2], [3, -2], [3.5, 1],
         [4,0], [5, 1], [6, -1], [7, 1], [8, -1], [9,1], [10,-1], [11, 0]])
        sp = spline(d, p= 3)
        u_knots = sp.get_knots() # returns the knots to Calculate the base functions at
        x = 0.0
        y = 0.0
        results = sp.get_points(1000)
        for u in np.linspace(u_knots[0],u_knots[-1],200):
            for i in range(0, len(u_knots)-2):
                N_i = sp.getN_i_k(u_knots, i) # gets the N_i function
                x+=N_i(u)*d[i][0]
                y+=N_i(u)*d[i][1]
            plt.plot(x, y, '*')
            self.assertEqual(round(sp.value(u)[0], 2), round(x, 2))
            self.assertEqual(round(sp.value(u)[1], 2), round(y, 2))
            x = 0
            y = 0
        plt.plot(d[:,0], d[:,1], '.', label="Control points")
        plt.plot(results[:,0], results[:,1], label="Spline")
        plt.legend(loc='best')
        plt.show()
        print("\nTEST: S(u)=Sum[d_i*N_i] OK\n")

    def test_splinecalc(self):
        d = np.array([[0,0], [5,0], [5,2], [8, 3], [5,8], [0,10]]).astype(float) #Control points
        sp = spline(d, p=3)
        sp.test()

    def plot_N(self):
        d = np.array([[0,0], [3,1], [4,1], [5,0], [5,2], [6,3],[8, 3], [8,4],[5,8], [0,10]]).astype(float) #Control pointss
        sp = spline(d, p= 3)
        s = np.linspace(0,2,300) # Resoluton of plot
        u_knots = sp.get_knots() # returns the knots to Calculate the base functions at
        for u in range(3, len(u_knots)-5):
            N_i = sp.getN_i_k(u_knots,u) # gets the N_i function
            plotArray=[]
            for i in s:
                plotArray.append(N_i(i))
            plt.plot(s, plotArray, label= "N" + str(u-2))
        plt.plot(u_knots, np.zeros(len(u_knots)), '*')
        plt.legend(loc='best')
        plt.show()

    def test_sum_of_N(self):
        d = np.array([[0,0], [3,1], [4,1], [5,0], [5,2], [6,3],[8, 3], [8,4],[5,8], [0,10]]).astype(float) #Control pointss
        sp = spline(d, p=3)
        x=0
        u_knots = sp.get_knots() # returns the knots to Calculate the base functions at
        for u in np.linspace(u_knots[2],u_knots[len(u_knots)-2],200):
            for i in range(0, len(u_knots)-2):
                N_i = sp.getN_i_k(u_knots,i) # gets the N_i function
                x+=(N_i(u))
            self.assertEqual(round(x,10),1.0)
            x = 0
        print("TEST: Sum of N OK")
        plot = input("Show plots of N? y/n(y)")
        if(plot!="n"):
            self.plot_N()


if __name__ == '__main__':
    unittest.main()
