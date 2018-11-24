import numpy as np

from function_to_min import InterfaceFunctionToMin


class absolute(InterfaceFunctionToMin):
    def __init__(self):
        InterfaceFunctionToMin.__init__(self)

    def f(self, x):
        return np.abs(x)

    def df(self, x):
        return np.sign(x)

    def ddf(self, x):
        return 1e-20
