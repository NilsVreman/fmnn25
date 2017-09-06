import unittest
from spline import spline
import numpy as np

class TestSpline(unittest.TestCase):

    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_splinecalc(self):

        d = np.array([[0,0], [5,0], [8, 3], [5,5], [0,10]]).astype(float) #Control points
        sp = spline(d,3)

        steps = 100                               #Nbr of steps to evaluate
        results = sp.get_points(steps)
        #Calculate the spline and put the results in a matrix
        # for i in range(0, steps+1):
        #     results[i,:] = sp.value(i/steps)
        # N_bases = sp.getN_i_k
        # for x in range(len(reults)):
        #     self.assertEqual(results[x],d[x]*N_bases[x])
        # print(results)
        sp.plot(100)

if __name__ == '__main__':
    unittest.main()
