from abc import ABC, abstractmethod
from Opt_Handler import Opt_Handler
import numpy as np
import scipy.linalg as spl

class Quasi_Newton_Handler(Opt_Handler, ABC):

    def optimize(self, f, x0, iterations, tol=1.e-6):

        x = x0.astype(float)
        x_old = x

        g = self.grad(f, x)
        H = np.eye(len(g))

        for i in range(1, iterations+1):

            s = -1*np.matmul(H, g)

            #a, b = self.exact_LS_GS(f, x, s, 0.8, 1.2, 100)
            #alpha = (a+b)/2
            alpha, _ = self.inexact_LS(f, x, s, 0.1, 'WP')
            x = x + alpha * s

            #Update variables
            g = self.grad(f, x)
            #WARNINGS SUPPRESSED IF DIVISION BY ZERO OR NAN
            np.seterr(divide='ignore', invalid='ignore')
            H = self.update(f, x_old, x, H)
            x_old = x

            #print("g, norm(g):", g, '\n', np.linalg.norm(g))
            if np.linalg.norm(g) < tol:
                break

            if np.isnan(H).any() or np.isinf(H).any():
                print('WARNING! H is NaN or Inf')
                break

        print("\tIterations:", i)
        return x

    @abstractmethod
    def update(self): pass

class BFGS(Quasi_Newton_Handler):
    def update(self, f, x, x_new, H):

        delta = (x_new - x)
        gamma = (self.grad(f, x_new)-self.grad(f, x))
        dTg = delta@gamma

        return H + (1 + gamma@H@gamma/dTg)*(np.outer(delta,delta)/dTg) - (np.outer(delta,gamma)@H + H@np.outer(gamma,delta))/dTg

class DFP(Quasi_Newton_Handler):
    def update(self, f, x, x_new, H):
        delta = (x_new - x)#.reshape(-1, 1)
        gamma = (self.grad(f, x_new)-self.grad(f, x))#.reshape(-1, 1)

        return H + (np.outer(delta,delta)/(delta@gamma)) - (H@np.outer(gamma,gamma)@H)/(gamma@H@gamma)        

class Good_Broyden(Quasi_Newton_Handler):
    def update(self, f, x, x_new, H):

        delta = (x_new - x)
        gamma = (self.grad(f, x_new) - self.grad(f, x))

        u = delta - (H@gamma)
        a = 1/(np.dot(u,gamma))
        return H + (a*np.outer(u,u))

class Bad_Broyden(Quasi_Newton_Handler):
    def update(self, f, x, x_new, H):
        delta = (x_new - x)
        gamma = (self.grad(f, x_new) - self.grad(f, x))

        try:
            c, lower = spl.cho_factor(H, lower=True)
            B = spl.cho_solve((c,lower), np.eye(len(H)))

            u = gamma - B@delta
            a = 1/(u.T@delta)

            B = B + a * (u@u.T)

            c, lower = spl.cho_factor(B, lower=True)
            H = spl.cho_solve((c, lower), np.eye(len(H)))

        except:
            raise Exception('Warning! Positive definite matrix')

        return H



if __name__ == '__main__':

    f = lambda x: x[0] ** 2 + x[1] ** 2 + 1
    #f = lambda x: 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

    bfgs = BFGS()
    dfp = DFP()
    bb = Bad_Broyden()
    gb = Good_Broyden()

    x0 = np.array([2, 2])

    print("BFGS:")
    print("\tAnswer:", bfgs.optimize(f, x0, 100))
    print("\nDFP:")
    print("\tAnswer:", dfp.optimize(f, x0, 100))
    print("\nGood_Broyden:")
    print("\tAnswer:", gb.optimize(f, x0, 100))
    print("\nBad_Broyden:")
    print("\tAnswer:", bb.optimize(f, x0, 100))
