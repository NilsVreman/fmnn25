from Newton_Handler import Classic_Newton, Exact_Newton, Inexact_Newton_G, Inexact_Newton_WP
import numpy as np

if __name__ == '__main__':

    f = lambda x: x[0] ** 2 + x[1] ** 2
    #f = lambda x: 100 * (x[0] - x[1] ** 2) ** 2 + (1 - x[0]) ** 2

    classic_newton = Classic_Newton()
    exact_newton = Exact_Newton()
    inexact_newton_g = Inexact_Newton_G()
    inexact_newton_wp = Inexact_Newton_WP()

    x0 = np.array([1,1])

    print("Classic Newton:")
    print("\tAnswer:", classic_newton.optimize(f, x0, 100))
    print("\nExact Newton:")
    print("\tAnswer:", exact_newton.optimize(f, x0, 100))
    print("\nInexact Newton G:")
    print("\tAnswer:", inexact_newton_g.optimize(f, x0, 100))
    print("\nInexact Newton WP:")
    print("\tAnswer:", inexact_newton_wp.optimize(f, x0, 100))