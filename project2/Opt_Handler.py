from abc import ABC, abstractmethod
import math as math
import numpy as np
import scipy.linalg as spl

class Opt_Handler(ABC):

    @abstractmethod
    def optimize(self):
        pass

    def __init__(self, grad=None):
        if grad is None:
            self.grad= self.__grad
        else:
            self.grad = grad

    def __grad(self, f, x):
        eps = 1.e-8

        if not hasattr(x, '__len__'): x = [x]

        g = np.zeros(len(x))
        for n in range(len(x)):
            e = np.zeros(len(x))
            e[n] = eps

            g[n] = (f(x + e) - f(x - e)) / (2.0 * eps)

        return g

    def hessian(self, func, point):

        e = 1.e-5
        n = len(point)
        G = np.zeros((n, n))
        g = np.zeros(n)

        g = self.grad(func, point)

        for x in range(0, n):
            new_point = np.copy(point)
            new_point[x] += e
            gplus = self.grad(func, new_point)
            new_point2 = np.copy(point)
            new_point2[x] -= e
            gminus = self.grad(func, new_point2)

            G[x] = (gplus - gminus) / (2 * e)

        G = ((G + G.T) / 2)
        try:
            c = spl.cholesky(G)
        except spl.LinAlgError as e:
            raise Exception("The matrix is not positive definite")

        return G

    def exact_LS_GS(self, f, x, s, a, b, n):
        """
        x: point to start line search in
        s: search direction
        a: min_interval
        b: max_interval
        n: nbr_iterations
        """
        f_alpha = lambda alpha: f(x + alpha * s)

        alpha = (math.sqrt(5) - 1) / 2
        a_k = a
        b_k = b
        # calculate where to put the next break point
        lambda_k = a_k + (1 - alpha) * (b_k - a_k)
        mu_k = a_k + alpha * (b_k - a_k)

        f_lambda = f_alpha(lambda_k)
        f_mu = f_alpha(mu_k)

        for i in range(0, n):
            if f_lambda < f_mu:
                b_k = mu_k
                mu_k = lambda_k
                lambda_k = a_k + (1 - alpha) * (b_k - a_k)
                f_mu = f_lambda
                f_lambda = f_alpha(lambda_k)
            else:
                a_k = lambda_k
                lambda_k = mu_k
                mu_k = a_k + alpha * (b_k - a_k)
                f_lambda = f_mu
                f_mu = f_alpha(mu_k)

        return a_k, b_k

    def inexact_LS(self, f, x, s, alpha_0, condition, rho=0.1, sigma=0.7, tau=0.1, chi=9):
        alpha_L = 0
        alpha_U = 10 ** 99

        rho = rho if rho <= 1 / 2 and rho >= 0 else 1 / 4
        sigma = sigma if sigma >= rho and sigma >= 0 and sigma <= 1 else rho + 1 / 4
        alpha = alpha_0

        f_alpha = lambda alpha: f(x + alpha * s)
        f_alpha_prim_L = self.grad(f_alpha, alpha_L)[0]
        f_alpha_prim_0 = self.grad(f_alpha, alpha)[0]
        f_alpha_L = f_alpha(alpha_L)
        f_alpha_0 = f_alpha(alpha)

        LC, RC = self.__inexact_LS_G(f_alpha_prim_0, f_alpha_prim_L, f_alpha_0, f_alpha_L, alpha, alpha_L, rho,
                                     sigma) if condition == 'G' else self.__inexact_LS_WP(f_alpha_prim_0,
                                                                                          f_alpha_prim_L,
                                                                                          f_alpha_0, f_alpha_L, alpha,
                                                                                          alpha_L, rho, sigma)

        while not LC or not RC:
            if not LC:
                factor = f_alpha_prim_L - f_alpha_prim_0
                if factor == 0:
                    #print('WARNING! DIVISION BY ZERO IN LINESEARCH')
                    return alpha, f_alpha(alpha)
                a_0 = (alpha - alpha_L) * f_alpha_prim_0 / factor
                a_0 = max(a_0, tau * (alpha - alpha_L))
                a_0 = min(a_0, chi * (alpha - alpha_L))
                alpha_L = alpha
                alpha = a_0 + alpha
            else:
                alpha_U = min(alpha, alpha_U)
                factor = 2 * (f_alpha_L - f_alpha_0 + (alpha - alpha_L) * f_alpha_prim_L)
                if factor == 0:
                    #print('WARNING! DIVISION BY ZERO IN LINESEARCH')
                    return alpha, f_alpha(alpha)
                a_0 = (alpha - alpha_L) ** 2 * f_alpha_prim_L / factor
                a_0 = max(a_0, alpha_L + tau * (alpha_U - alpha_L))
                a_0 = min(a_0, alpha_U - tau * (alpha_U - alpha_L))
                alpha = a_0

            f_alpha_prim_L = self.grad(f_alpha, alpha_L)[0]
            f_alpha_prim_0 = self.grad(f_alpha, alpha)[0]
            f_alpha_L = f_alpha(alpha_L)
            f_alpha_0 = f_alpha(alpha)

            LC, RC = self.__inexact_LS_G(f_alpha_prim_0, f_alpha_prim_L, f_alpha_0, f_alpha_L, alpha, alpha_L, rho,
                                         sigma) if condition == 'G' else self.__inexact_LS_WP(f_alpha_prim_0,
                                                                                              f_alpha_prim_L, f_alpha_0,
                                                                                              f_alpha_L, alpha, alpha_L,
                                                                                              rho, sigma)

        return alpha, f_alpha(alpha)

    """
    inexact line search using Goldstein criterion
    """

    def __inexact_LS_G(self, f_alpha_prim_0, f_alpha_prim_L, f_alpha_0, f_alpha_L, alpha, alpha_L, rho, sigma):
        return f_alpha_0 >= f_alpha_L + (1 - rho) * (alpha - alpha_L) * f_alpha_prim_L, f_alpha_0 <= f_alpha_L + rho * (
            alpha - alpha_L) * f_alpha_prim_L

    """
    inexact line search using Wolfe-Powell criterion
    """

    def __inexact_LS_WP(self, f_alpha_prim_0, f_alpha_prim_L, f_alpha_0, f_alpha_L, alpha, alpha_L, rho, sigma):
        return f_alpha_prim_0 >= sigma * f_alpha_prim_L, f_alpha_0 <= f_alpha_L + rho * (
        alpha - alpha_L) * f_alpha_prim_L
