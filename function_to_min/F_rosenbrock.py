import numpy as np
from numpy.linalg import inv
from function_to_min import InterfaceFunctionToMin
from util import util

class Rosenbrock(InterfaceFunctionToMin):
    def __init__(self):
        InterfaceFunctionToMin.__init__(self)

    def rosen_hess(self,x):
        x = np.asarray(x)
        H = np.diag(-400 * x[:-1], 1) - np.diag(400 * x[:-1], -1)
        diagonal = np.zeros_like(x)
        diagonal[0] = 1200 * x[0] ** 2 - 400 * x[1] + 2
        diagonal[-1] = 200
        diagonal[1:-1] = 202 + 1200 * x[1:-1] ** 2 - 400 * x[2:]
        H = H + np.diag(diagonal)
        return H

    def f(self,x):
        return np.sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)
    def df(self,x):
        return util.partial_derivative(self.f,x)
    def ddf(self,x):
        hessian=self.rosen_hess(x)
        part_deriv_rosen=self.df(x)
        part_deriv_rosen2=np.dot(inv(hessian),part_deriv_rosen)
        return part_deriv_rosen2