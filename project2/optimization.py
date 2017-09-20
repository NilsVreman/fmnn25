from abc import ABC
import math as math
import numpy as np
import scipy.linalg as spl

class optimization(ABC):

    def __init__(self, f, g=None):
        self.__f = f
        self.__gradient = g

    def classical_newton(self, x0, hessian, nbrIters):
        #FAR FROM DONE

        try:
            si = spl.cholesky(hessian)
        except LinAlgError:
            raise Exception('Hessian is not positive definite.')

        for i in range(0, nbrIters):
            pass

    """
    inexact line search using Goldstein criterion
    """
    def inexact_LS_G(self, x, s, rho, alpha_L=0, alpha_U=10**99):

        rho = rho if rho <= 1/2 and rho >= 0 else 1/4
        alpha = 1
        f_alpha = lambda alpha: self.__f(x+alpha*s)

        #Checks whether we are outside the feasible area (to the right)
        while f_alpha(alpha) > f_alpha(0):
            alpha = alpha/2

        #Checks whether we are to close to origin or function is to constant
        while f_alpha(alpha) < f_alpha(0)+max(0.01*abs(f_alpha(0)), 10^-4):
            alpha = alpha*2

        #Check for positive derivative
        h = 10**-6*alpha
        f_alpha_prim_L = (f_alpha(alpha_L+h) + f_alpha(alpha_L))/h

        if f_alpha_prim_L > 0:
            print('Warning! Positive derivative. Set to zero')
            f_alpha_prim_L = 0

        #Create a conditional function
        f_cond_L = lambda alpha: f_alpha(alpha_L)+(1-rho)*(alpha - alpha_L)*f_alpha_prim_L
        f_cond_U = lambda alpha: f_alpha(alpha_L)+rho*(alpha - alpha_L)*f_alpha_prim_L

        #Check conditions
        while f_alpha(alpha) <= f_cond_U(alpha):
            alpha = alpha*2
            
        while f_alpha(alpha) >= f_cond_L(alpha):
            alpha = alpha/2

        if math.isnan(self.__f(x+alpha*s)) or self.__f(x+alpha*s) > self.__f(x):
            print('Warning! Line search did a bad job')

        return alpha

    """
    inexact line search using Wolfe-Powell criterion
    """
    def inexact_LS_WP(self, x, s, rho, sigma, alpha_L=0, alpha_U=10**99):

        rho = rho if rho <= 1/2 and rho >= 0 else 1/4
        sigma = sigma if sigma <= 1 and sigma >= 0 and sigma > rho else rho+1/4
        alpha = 1
        f_alpha = lambda alpha: self.__f(x+alpha*s)

        #Checks whether we are outside the feasible area (to the right)
        while f_alpha(alpha) > f_alpha(0):
            alpha = alpha/2

        #Checks whether we are to close to origin or function is to constant
        while f_alpha(alpha) < f_alpha(0)+max(0.01*abs(f_alpha(0)), 10^-4):
            alpha = alpha*2

        #Check for positive derivative
        h = 10**-6*alpha
        f_alpha_prim_L = (f_alpha(alpha_L+h) - f_alpha(alpha_L))/h

        if f_alpha_prim_L > 0:
            print('Warning! Positive derivative. Set to zero')
            f_alpha_prim_L = 0

        #Create a conditional function
        f_cond_L = f_alpha_prim_L*sigma
        f_cond_U = lambda alpha: f_alpha(alpha_L)+rho*(alpha - alpha_L)*f_alpha_prim_L

        #Check conditions
        while f_alpha(alpha) <= f_cond_U(alpha):
            alpha = alpha*2
            
        while (f_alpha(alpha+h) - f_alpha(alpha))/h >= f_cond_L:
            alpha = alpha/2

        if math.isnan(self.__f(x+alpha*s)) or self.__f(x+alpha*s) > self.__f(x):
            print('Warning! Line search did a bad job')

        return alpha

    def grad(self, f, x):

        if not hasattr(x, '__len__'): x = [x]
        
        xplus = np.array([i+10**-8 for i in x])
        xminus = np.array([j-10**-8 for j in x])
        g = np.divide((f(xplus)-f(xminus)), 2*10**-8)
        
        return g



if __name__ == '__main__':
    f = lambda x: 100*(x[0]-x[1]**2)**2+(1-x[0])**2
    g = lambda x: 2*x
    o = optimization(f, g)
    alpha = o.inexact_LS_G(np.array([0,0]).astype(float), np.array([1, 0]).astype(float), 0.01)
    alpha2 = o.inexact_LS_WP(np.array([0,0]).astype(float), np.array([1, 0]).astype(float), 0.01, 0.1)
    print(alpha, alpha2)
    temp = o.grad(f, np.array([1,1]))
    t = lambda x: x**2
    print(o.grad(t, 1))
    print(temp)
