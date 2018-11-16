import numpy as np
import numdifftools as nd
from scipy.misc import derivative

def partial_function(f___,input,pos,value):
    tmp  = input[pos]
    input[pos] = value
    ret = f___(input)#ret = f___(*input)
    input[pos] = tmp
    return ret

def partial_derivative(f,input):
    nbinput=input.shape[0]
    ret = np.empty(nbinput)
    for i in range(nbinput):
        fg = lambda x:partial_function(f,input,i,x)
        ret[i] = nd.Derivative(fg)(input[i])
    return ret