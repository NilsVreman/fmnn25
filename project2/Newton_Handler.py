from abc import ABC, abstractmethod
from Opt_Handler import Opt_Handler
import numpy as np
import scipy.linalg as spl

class Newton_Handler(Opt_Handler, ABC):
    def optimize(self, f, x0, iterations, grad=None, tol=1.e-6):
        if not(grad is None):
            self.grad = grad

        x = x0.astype(float)

        g = self.grad(f, x)
        G = self.hessian(f, x)
        for i in range(1, iterations+1):
            c, lower = spl.cho_factor(G, lower = True)
            s = spl.cho_solve((c, lower), g)
            s = np.multiply(s, -1)
            x = x + self.alpha(f, x, s) * s
            g = self.grad(f, x)
            G = self.hessian(f, x)
            # print("\tg, norm(g):", g, np.linalg.norm(g))
            if np.linalg.norm(g) < tol:
                break

        print("\tIterations:", i)
        return x

    @abstractmethod
    def alpha(self): pass

class Classic_Newton(Newton_Handler):
    def alpha(self, f, x, s):
        return 1.0

class Exact_Newton(Newton_Handler):
    def alpha(self, f, x, s):
        a = 0.8
        b = 1.2
        a, b = self.exact_LS_GS(f, x, s, a, b, 100)
        alpha = (a + b) / 2
        return alpha

class Inexact_Newton_G(Newton_Handler):
    def alpha(self, f, x, s):
        alpha, _ = self.inexact_LS(f, x, s, 500, 'G')
        return alpha

class Inexact_Newton_WP(Newton_Handler):
    def alpha(self, f, x, s):
        alpha, _ = self.inexact_LS(f, x, s, 500, 'WP')
        return alpha
