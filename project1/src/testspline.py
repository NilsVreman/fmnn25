import unittest
from spline import spline
import numpy as np
import matplotlib.pyplot as plt

class TestSpline(unittest.TestCase):
    # Test if S(u) = sum(d_i*N_i)
    # def test_splinecalc(self):
    #     p = 3
    #     d = np.array([[0, 4], [0.5, 5], [1, 6], [2, 6], [3, 6], [3.5, 5], [4, 4]])
    #     sp = spline(d, p=3)
    #     s = np.linspace(0,2,300 )
    #     u_knots = sp.get_knots() # returns the knots to Calculate the base functions at
    #     print(u_knots)
    #     x0=0.0
    #     x1=0.0
    #     u1=0.3
    #     print(len(u_knots))
    #     results = sp.get_points(100)
    #     for u in np.linspace(0.26,0.74,200):
    #         plotArray1=[]
    #         plotArray2=[]
    #         for i in range(0, len(u_knots)-2):
    #             N_i = sp.getN_i_k(u_knots, i) # gets the N_i function
    #             x0+=N_i(u)*d[i][0]
    #             x1+=N_i(u)*d[i][1]
    #         plotArray1.append(N_i(u)*d[i][0])
    #         plotArray2.append(N_i(u)*d[i][1])
    #         plt.plot(x0,x1,'*')
    #         x0=0
    #         x1=0
    #         # plt.plot(u_knots, np.zeros(len(u_knots)), '*')
    #     plt.plot(d[:,0],d[:,1],'.',label="Control points")
    #     plt.plot(results[:,0],results[:,1],label="Spline")
    #     plt.legend(loc='best')
    #     plt.show()
    #     print("BASE",x0,x1)
    #     print("SPLINE",sp.value(u1))
    #     # self.assertEqual(round(sp.value(u1)[0],5),round(x0,5))
    #     # self.assertEqual(round(sp.value(u1)[1],5),round(x1,5))
    #
    #
    #
    def test_plot_N(self):
        d = np.array([[0,0], [3,1], [4,1], [5,0], [5,2], [6,3],[8, 3], [8,4],[5,8], [0,10]]).astype(float) #Control pointss
        sp = spline(d, p=3)
        x=0
        s = np.linspace(0,2,300) # Resoluton of plot
        u_knots = sp.get_knots() # returns the knots to Calculate the base functions at
        print(len(u_knots))
        for u in range(0, len(u_knots)-2):
            N_i = sp.getN_i_k(u_knots,u) # gets the N_i function
            plotArray=[]
            x+=(N_i(0.4))
            for i in s:
                plotArray.append(N_i(i))
            plt.plot(s, plotArray, label= "N" + str(u))
        plt.plot(u_knots, np.zeros(len(u_knots)), '*')
        print("SUM OF BASIS",x)
        plt.legend(loc='best')
        plt.show()
        
    def test_sum_of_N(self):
        d = np.array([[0,0], [3,1], [4,1], [5,0], [5,2], [6,3],[8, 3], [8,4],[5,8], [0,10]]).astype(float) #Control pointss
        sp = spline(d, p=3)
        x=0
        u_knots = sp.get_knots() # returns the knots to Calculate the base functions at
        for u in np.linspace(u_knots[2]+0.1,u_knots[len(u_knots)-2]-0.1,200):
            for i in range(0, len(u_knots)-2):
                N_i = sp.getN_i_k(u_knots,i) # gets the N_i function
                x+=(N_i(u))
            self.assertEqual(round(x,10),1.0)
            x=0
        print("Sum of N OK")

if __name__ == '__main__':
    unittest.main()
