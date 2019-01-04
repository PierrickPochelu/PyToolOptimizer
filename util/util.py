import numpy as np
import numdifftools as nd

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

def random_dir(nb_dimensions, sigma):
    x=np.zeros((nb_dimensions,))
    for i in range(nb_dimensions):
        rand_vec=np.random.uniform(size=nb_dimensions)
        rand_vec/=np.linalg.norm( rand_vec )
        rand_vec*=np.random.normal(0,sigma)
        x[i]=np.linalg.norm( rand_vec )
    return x