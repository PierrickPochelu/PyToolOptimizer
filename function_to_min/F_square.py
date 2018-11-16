import numpy as np
from function_to_min.Abstract_function_to_min import function_to_min
class square(function_to_min):
    def __init__(self):
        function_to_min.__init__(self)

    def f(self,x):
        return (x)**2

    def df(self, x):
        return 2*x

    def ddf(self, x):
        return 2.