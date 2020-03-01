import numpy as np
from display import *
from function_to_min import *
from optimizer import *

from function_to_min.tensorflow_util.DB_CIFAR10 import CIFAR10

# call function to minimize
d=CIFAR10("./function_to_min/tensorflow_util/")
np_d=d.get_np_dataset()
function_to_min = F_deeplearning.Deeplearning(np_d,bs=32)

w=function_to_min.get_w()




for i in range(100):
    for j in range(50000//function_to_min.batch_size):
        df=function_to_min.df(w)
        ddf=function_to_min.ddf(w)

        for i in range(len((w))):
            w[i]=w[i]-0.1*(df[i]/(ddf[i]+1e-10))#newton method
            #w[i]=w[i] - 0.1*df[i] #SGD
        function_to_min.set_w(w)


        print(function_to_min.f(w))
        function_to_min.next_batch()


    print(function_to_min.accuracy(w))
    function_to_min.shuffle()



"""
for i in range(100):
    for j in range(50000//128):
        g=function_to_min.df(w)
        assert(len(g)==len(w))
        for i, (gi,wi) in enumerate(zip(g, w)):
            w[i]= -0.1 * gi + wi
        function_to_min.set_w(w)

        function_to_min.next_batch()
    

    print(function_to_min.accuracy(w))
"""
exit()

# build optimizer
"""
def a_priori(x):
    return np.random.normal(x, .1)
opt = MCMC(PI=a_priori, lambd=10., iter_mcmc=512)
"""

def a_priori(x):
    return np.random.normal(x, .1)
opt = SPSA(lr=.1,epsilon_length=1e-3,PI=a_priori)

# init search position
x = np.array([4., 5.])

# search
position_history = [x]
print("current position : %.3f,%.3f loss= %.3f" % (x[0], x[1], function_to_min.f(x)))
for i in range(100):
    x = opt.run_one_step(x, function_to_min)
    print("current position : %.3f,%.3f loss= %.3f" % (x[0], x[1], function_to_min.f(x)))
    position_history.append(x)

# draw path
plot2D(position_history, function_to_min)
