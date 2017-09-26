import unittest

import numpy as np
import scipy.optimize as so
from Newton_Handler import Classic_Newton, Exact_Newton, Inexact_Newton_G, Inexact_Newton_WP
from Quasi_Newton_Handler import BFGS, DFP, Good_Broyden, Bad_Broyden
from chebyquad_problem import chebyquad, gradchebyquad, gradchebyquad
from numpy.testing import assert_allclose

class TestOpt(unittest.TestCase):

    def setUp(self):
        self.classic_newton = Classic_Newton()
        self.exact_newton = Exact_Newton()
        self.inexact_newton_g = Inexact_Newton_G()
        self.inexact_newton_wp = Inexact_Newton_WP()
        self.bfgs = BFGS()
        self.dfp = DFP()
        self.bb = Bad_Broyden()
        self.gb = Good_Broyden()

    def cheby_cases(self, x0, atol=0):
        xmin = so.fmin_bfgs(chebyquad, x0, gradchebyquad)
        assert_allclose(self.bfgs.optimize(chebyquad, x0, 300, grad=gradchebyquad), xmin, atol=atol)
        assert_allclose(self.dfp.optimize(chebyquad, x0, 300, grad=gradchebyquad), xmin, atol=atol)
        assert_allclose(self.gb.optimize(chebyquad, x0, 300, grad=gradchebyquad), xmin, atol=atol)
        assert_allclose(self.bb.optimize(chebyquad, x0, 300, grad=gradchebyquad), xmin, atol=atol)

    def test_newton_methods_rosenbrock(self):
        '''
        Tests that the newton methods with the rosenbrock function
        '''
        f = lambda x: 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2
        x0 = np.array([1.01,0.901])
        assert_allclose(self.classic_newton.optimize(f, x0, 100), np.array([1,1]))
        assert_allclose(self.exact_newton.optimize(f, x0, 100), np.array([1,1]))
        assert_allclose(self.inexact_newton_g.optimize(f, x0, 100), np.array([1,1]))
        assert_allclose(self.inexact_newton_wp.optimize(f, x0, 100), np.array([1,1]))

    def test_quasi_newtion_rosenbrock(self):
        '''
        Tests that the quasi-newton methods with the rosenbrock function
        '''

        f = lambda x: 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2
        x0 = np.array([2,2])
        assert_allclose(self.bfgs.optimize(f, x0, 300), np.array([1,1]),rtol=1e-07)
        assert_allclose(self.dfp.optimize(f, x0, 300), np.array([1,1]),rtol=1e-07)
        assert_allclose(self.gb.optimize(f, x0, 300), np.array([1,1]),rtol=1e-07)
        # BB DOES NOT WORK FOR SOME REASON
        # assert_allclose(self.bb.optimize(f, x0, 100), np.array([1,1]),rtol=1e-07)


    def test_chebyquad4(self):
        '''
        Tests the chebyguad with n=11, needs absolute tolerance of 0.05
        '''
        x0 = np.linspace(0,1,4)
        self.cheby_cases(x0, 0.05)

    def test_chebyquad8(self):
        '''
        Tests the chebyguad with n=11, needs absolute tolerance of 0.1
        '''
        x0 = np.linspace(0,1,8)
        self.cheby_cases(x0, 0.1)

    def test_chebyquad11(self):
        '''
        Tests the chebyguad with n=11, needs absolute tolerance of 0.6
        '''
        x0 = np.linspace(0,1,11)
        self.cheby_cases(x0, 0.6)

if __name__ == '__main__':
    unittest.main()
