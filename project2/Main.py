from Newton_Handler import Classic_Newton, Exact_Newton, Inexact_Newton_G, Inexact_Newton_WP
from Quasi_Newton_Handler import BFGS, DFP, Good_Broyden, Bad_Broyden
import numpy as np
from chebyquad_problem import chebyquad, gradchebyquad
import scipy.optimize as so

"""
Participants: Anton Göransson, Carl Nilsson, Alexander Arcombe, Nils Vreman
"""
if __name__ == '__main__':

    # f = lambda x: x[0] ** 2 + x[1] ** 2
    f = lambda x: 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

    classic_newton = Classic_Newton()
    exact_newton = Exact_Newton()
    inexact_newton_g = Inexact_Newton_G()
    inexact_newton_wp = Inexact_Newton_WP()

    x0 = np.array([1.01,0.901])

    print("Classic Newton:")
    print("\tAnswer:", classic_newton.optimize(f, x0, 100))
    print("\nExact Newton:")
    print("\tAnswer:", exact_newton.optimize(f, x0, 100))
    print("\nInexact c Newton G:")
    print("\tAnswer:", inexact_newton_g.optimize(f, x0, 100))
    print("\nInexact Newton WP:")
    print("\tAnswer:", inexact_newton_wp.optimize(f, x0, 100))

    x = np.linspace(0,1,4)
    # print("\nNewton CHEBY:")

    bfgs = BFGS()
    dfp = DFP()
    bb = Bad_Broyden()
    gb = Good_Broyden()

    x0 = np.array([2, 2])
    # print("BFGS:")
    # print("\tAnswer:", bfgs.optimize(f, x0, 100))
    # print("\nDFP:")
    # print("\tAnswer:", dfp.optimize(f, x0, 100))
    # print("\nGood_Broyden:")
    # print("\tAnswer:", gb.optimize(f, x0, 100))
    # print("\nBad_Broyden:")
    # print("\tAnswer:", bb.optimize(f, x0, 100))
    print("\tAnswer:", classic_newton.optimize(chebyquad, x, 100, grad=gradchebyquad))
    print("\tAnswer:", exact_newton.optimize(chebyquad, x, 100, grad=gradchebyquad))
    print("\tAnswer:", inexact_newton_g.optimize(chebyquad, x, 100, grad=gradchebyquad))
    print("\tAnswer:", inexact_newton_wp.optimize(chebyquad, x, 100, grad=gradchebyquad))
    # print("\tAnswer:", bfgs.optimize(chebyquad, x, 400, grad=gradchebyquad))
    # print("\tAnswer:", dfp.optimize(chebyquad, x, 400, grad=gradchebyquad))
    # print("\tAnswer:", bb.optimize(chebyquad, x, 400, grad=gradchebyquad))
    # print("\tAnswer:", gb.optimize(chebyquad, x, 400, grad=gradchebyquad))
    # print(so.fmin_bfgs(chebyquad,x,gradchebyquad1))
