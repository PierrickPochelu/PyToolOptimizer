import numpy as np
from function_to_min.Abstract_function_to_min import function_to_min
class absolute(function_to_min):
    def __init__(self):
        function_to_min.__init__(self)

    def f(self,x):
        return np.abs(x)

    def df(self, x):
        return np.sign(x)

    def ddf(self, x):
        return 1e-20