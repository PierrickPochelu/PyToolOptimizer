import numpy as np

from function_to_min.Abstract_function_to_min import function_to_min


class cosinusoidal(function_to_min):
    def __init__(self):
        function_to_min.__init__(self)

    def f(self,x):
        return 4*np.cos(4*x) + x*(x-1) + 4

    def df(self, x):
        return -16*np.sin(4*x)+2*x-1

    def ddf(self, x):
        return 2 - 64 * np.cos(4*x)