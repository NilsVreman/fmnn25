from abc import ABC, abstractmethod
from Opt_Handler import Opt_Handler
import numpy as np
import scipy.linalg as spl

class Newton_Handler(Opt_Handler, ABC):
    def optimize(self, f, guess, iterations, tol=1.e-6):
        x = guess.astype(float)

        for i in range(0, iterations):
            g = self.grad(f, x)
            G = self.hessian(f, x)
            c, lower = spl.cho_factor(G, lower = True)
            s = spl.cho_solve((c, lower), g)
            s = np.multiply(s, -1)
            x = x + self.alpha(f, x, s) * s
            if np.linalg.norm(g) < tol:
                break

        print("\tIterations:", i)
        return x

    @abstractmethod
    def alpha(self): pass

class Classic_Newton(Newton_Handler):
    def alpha(self, f, x, s):
        return 1

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