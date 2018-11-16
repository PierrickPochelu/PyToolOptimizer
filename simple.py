import numpy as np

from function_to_min import *
from optimizer import *
from display import *

# call function to minimize
function_to_min = Rosenbrock()


# build optimizer
def a_priori(x):
    return np.random.normal(x, .1)
opt = MCMC(PI=a_priori, lambd=10., iter_mcmc=512)

# init search position
x = np.array([5., 5.])

# search
position_history = []
print("current position : %.3f,%.3f loss= %.3f" % (x[0], x[1], function_to_min.f(x)))
for i in range(10):
    x = opt.run_one_step(x, function_to_min)
    print("current position : %.3f,%.3f loss= %.3f" % (x[0], x[1], function_to_min.f(x)))
    position_history.append(x)

# draw path
plot2D(position_history, function_to_min)