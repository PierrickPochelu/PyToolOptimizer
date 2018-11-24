import numpy as np
from function_to_min import InterfaceFunctionToMin
class square(InterfaceFunctionToMin):
    def __init__(self):
        InterfaceFunctionToMin.__init__(self)

    def f(self,x):
        return (x)**2

    def df(self, x):
        return 2*x

    def ddf(self, x):
        return 2.