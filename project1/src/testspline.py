import unittest
from spline import spline
import numpy as np
import matplotlib.pyplot as plt

class TestSpline(unittest.TestCase):
    # EXAMPLES
    # def test_upper(self):
    #     self.assertEqual('foo'.upper(), 'FOO')
    #
    # def test_isupper(self):
    #     self.assertTrue('FOO'.isupper())
    #     self.assertFalse('Foo'.isupper())

    def test_splinecalc(self):
        
        
        p = 3
        d = np.array([[0,0], [5,0], [5,2], [8, 3], [5,8], [0,10]]).astype(float) #Control points
        sp = spline(d, p=3)
        sp.test()

    def test_plot_N(self):
        d = np.array([[0,0], [3,1], [4,1], [5,0], [5,2], [6,3],[8, 3], [8,4],[5,8], [0,10]]).astype(float) #Control pointss
        sp = spline(d, p=3)
        s = np.linspace(0,2,300) # Resoluton of plot
        u_knots = sp.get_knots() # returns the knots to Calculate the base functions at
        for u in range(1, len(u_knots)-3):
            N_i =sp.getN_i_k(u_knots,u) # gets the N_i function
            plotArray=[]
            for i in s:
                plotArray.append(N_i(i))
            plt.plot(s, plotArray, label= "N" + str(u))
        plt.plot(u_knots, np.zeros(len(u_knots)), '*')
        plt.legend(loc='best')
        plt.show()


if __name__ == '__main__':
    unittest.main()
