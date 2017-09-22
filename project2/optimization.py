from abc import ABC
import math as math
import numpy as np
import scipy.linalg as spl
import scipy.optimize as opt

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
    Exact line search using golden section
    """
    def exact_LS_GS(self, x, s, a, b, n):
        """
        x: point to start line search in
        s: search direction
        a: min_interval
        b: max_interval
        n: nbr_iterations
        """
        f_alpha = lambda alpha: self.__f(x+alpha*s)

        alpha = (math.sqrt(5)-1)/2
        a_k = a
        b_k = b
        # calculate where to put the next break point
        lambda_k = a_k + (1 - alpha)*(b_k - a_k)
        mu_k = a_k + alpha*(b_k-a_k)

        f_lambda = f_alpha(lambda_k)
        f_mu = f_alpha(mu_k)

        for i in range(0, n):
            if f_lambda < f_mu:
                b_k = mu_k
                mu_k = lambda_k
                lambda_k = a_k + (1 - alpha)*(b_k - a_k)
                f_mu = f_lambda
                f_lambda = f_alpha(lambda_k)
            else:
                a_k = lambda_k
                lambda_k = mu_k
                mu_k = a_k + alpha*(b_k-a_k)
                f_lambda = f_mu
                f_mu = f_alpha(mu_k)

        return a_k, b_k


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
        eps = 1.e-8
        
        if not hasattr(x, '__len__'): x = [x]
        
        g = np.zeros(len(x))
        for n in range(len(x)):
            e = np.zeros(len(x))
            e[n] = eps

            g[n] = ( f(x+e) - f(x-e) ) / ( 2.0 * eps)
            
        return g 
    
    def hessian(self, func, point):
        
        e = 1.e-5
        n = len(point)
        G = np.zeros((n ,n))
        g = np.zeros(n)
        
        g = self.grad(func, point)
        
        for x in range(0, n):
    
            new_point = np.copy(point)
            new_point[x] += e
            gplus = self.grad(func, new_point)
            new_point2 = np.copy(point)
            new_point2[x] -= e
            gminus = self.grad(func, new_point2)
            
            G[x] = (gplus - gminus) / (2*e)
        try:
            c = spl.cholesky(G)
        except spl.LinAlgError as e:
            raise Exception("The matrix is not positive definite")
        
        return ((G + G.T) / 2)


    def classic_Newton_method(self, func, guess, iteration, tol = 1.e-8):
        x = guess
        
        while (iteration > 0):
            g = self.right_grad(func, x)
            G = self.hessian(func, x)
            c, lower = spl.cho_factor(G, lower = True)
            s = spl.cho_solve((c, lower), g)
            x = x - s
            if np.linalg.norm(g) < tol:
                break
            iteration -= 1
        
        return x
    
    def exact_Newton_method(self, func, guess, iteration, tol = 1.e-8):
        x = guess
        a = 0.8
        b = 1.2
        self.__f = func
        
        while (iteration > 0):
            g = self.right_grad(func, x)
            G = self.hessian(func, x)
            c, lower = spl.cho_factor(G, lower = True)
            s = spl.cho_solve((c, lower), g)
            a, b = self.exact_LS_GS(x, s, a, b, 100)
            alpha = (a + b) / 2
            x = x - alpha * s
            if np.linalg.norm(g) < tol:
                break
            iteration -= 1
        
        print(100 - iteration)
        return x
    
    def inexact_Newton_method_G(self, func, guess, iteration, tol = 1.e-8):
        x = guess
        self.__f = func
        
        while (iteration > 0):
            g = self.right_grad(func, x)
            G = self.hessian(func, x)
            c, lower = spl.cho_factor(G, lower = True)
            s = spl.cho_solve((c, lower), g)
            
            alpha = self.inexact_LS_G(x, s, 0.1)
            x = x - alpha * s
            if np.linalg.norm(g) < tol:
                break
            iteration -= 1
        
        print(100 - iteration)
        return x
    
    def inexact_Newton_method_WP(self, func, guess, iteration, tol = 1.e-8):
        x = guess
        self.__f = func
        
        while (iteration > 0):
            g = self.right_grad(func, x)
            G = self.hessian(func, x)
            c, lower = spl.cho_factor(G, lower = True)
            s = spl.cho_solve((c, lower), g)
            alpha = self.inexact_LS_WP(x, s, 0.1, 0.7)
            x = x - alpha * s
            if np.linalg.norm(g) < tol:
                break
            iteration -= 1
        
        print(100 - iteration)
        return x

    def update_DFP(self, f, x, x_new, H):
        delta = (x_new - x).reshape(-1, 1)
        gamma = (self.grad(f, x_new)-self.grad(f, x)).reshape(-1, 1)

        return H + np.matmul(delta, delta.T)/np.matmul(delta.T, gamma) - np.matmul(np.matmul(H, gamma), np.matmul(gamma.T,H))/np.matmul(gamma.T, np.matmul(H,gamma))

    def update_BFGS(self, f, x, x_new, H):
        delta = (x_new - x).reshape(-1, 1)
        gamma = (self.grad(f, x_new)-self.grad(f, x)).reshape(-1, 1)

        return H + (1 + np.matmul(gamma.T, np.matmul(H, gamma))/np.matmul(delta.T, gamma))*np.matmul(delta, delta.T)/np.matmul(delta.T, gamma) - (np.matmul(delta, np.matmul(gamma.T, H)) + np.matmul(H, np.matmul(gamma, delta.T)))/np.matmul(delta.T, gamma)

    def update_GB(self, f, x, x_new, H):
        delta = (x_new - x).reshape(-1, 1)
        gamma = (self.grad(f, x_new)-self.grad(f, x)).reshape(-1, 1)
        u = delta - np.matmul(H, gamma)
        a = 1/np.matmul(u.T, gamma)

        return H + np.matmul(a, np.matmul(u, u.T))

    def update_BB(self, f, x, x_new, H):
        pass
        
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

    h = lambda x: np.power(x, 2)
    opt = optimization(h)
    a, b = opt.exact_LS_GS(1, -2, 0, 10, 10)
    print('Golden section', (a+b)/2)
    
    f2 = lambda x: x[0]**2 + x[1]**2
    f = lambda x: 100*(x[0]-x[1]**2)**2+(1-x[0])**2

    op = optimization(f)
    
    point = np.zeros(2)
    point[0] = 3.0
    point[1] = 2.0
    
    w = op.inexact_Newton_method_WP(f, point, 100)
    print(w)

    w = op.inexact_Newton_method_WP(f2, point, 100)
    print(w)



    A = op.update_DFP(lambda x: x[0]**2 + x[1]**2, np.array([1, 1]), np.array([0, 0]),  np.array([[2, 0], [0, 2]]))
    print(A)

