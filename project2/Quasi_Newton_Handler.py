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
            
            print(s)
            alpha, _ = self.inexact_LS(f, x, s, 400, 'G')
            x = x + alpha * s

            #Update variables
            g = self.grad(f, x)
            #WARNINGS SUPPRESSED IF DIVISION BY ZERO OR NAN
            np.seterr(divide='ignore', invalid='ignore')
            H = self.update(f, x_old, x, H)
            x_old = x

            print("g, norm(g):", g, np.linalg.norm(g))
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
        delta = (x_new - x).reshape(-1, 1)
        gamma = (self.grad(f, x_new)-self.grad(f, x)).reshape(-1, 1)

        return H + (1 + np.matmul(gamma.T, np.matmul(H, gamma))/np.matmul(delta.T, gamma))*np.matmul(delta, delta.T)/np.matmul(delta.T, gamma) - (np.matmul(delta, np.matmul(gamma.T, H)) + np.matmul(H, np.matmul(gamma, delta.T)))/np.matmul(delta.T, gamma)

class DFP(Quasi_Newton_Handler):
    def update(self, f, x, x_new, H):
        delta = (x_new - x).reshape(-1, 1)
        gamma = (self.grad(f, x_new)-self.grad(f, x)).reshape(-1, 1)

        return H + np.matmul(delta, delta.T)/np.matmul(delta.T, gamma) - np.matmul(np.matmul(H, gamma), np.matmul(gamma.T,H))/np.matmul(gamma.T, np.matmul(H,gamma))

class Bad_Broyden(Quasi_Newton_Handler):
    def update(self, f, x, x_new, H):
        delta = (x_new - x).reshape(-1, 1)
        gamma = (self.grad(f, x_new) - self.grad(f, x)).reshape(-1, 1)
        u = delta - np.matmul(H, gamma)
        a = 1/np.matmul(u.T, gamma)

        return H + a[0][0] * np.matmul(u, u.T)

class Good_Broyden(Quasi_Newton_Handler):
    def update(self, f, x, x_new, H):
        delta = (x_new - x).reshape(-1, 1)
        gamma = (x_new - np.matmul(H, x)).reshape(-1, 1)
        print(delta, '\n', gamma)

        return H + np.matmul(gamma, delta.T)/np.matmul(delta.T, delta)



if __name__ == '__main__':

    f = lambda x: x[0] ** 3 + x[1] ** 2 + 1
    #f = lambda x: 100 * (x[0] - x[1] ** 2) ** 2 + (1 - x[0]) ** 2

    bfgs = BFGS()
    dfp = DFP()
    bb = Bad_Broyden()
    gb = Good_Broyden()

    x0 = np.array([1,1])

    print("BFGS:")
    print("\tAnswer:", bfgs.optimize(f, x0, 100))
    print("\nDFP:")
    print("\tAnswer:", dfp.optimize(f, x0, 100))
    print("\nGood_Broyden:")
    print("\tAnswer:", gb.optimize(f, x0, 100))
    print("\nBad_Broyden:")
    print("\tAnswer:", bb.optimize(f, x0, 100))
