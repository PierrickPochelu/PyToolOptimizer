import numpy as np
from display import *
from function_to_min import *
from optimizer import *

def log_run(opt,function_to_min,title):
    x = np.array([5.])
    position_history = [x]
    print("current position : %.3f loss= %.3f" % (x[0], function_to_min.f(x)))
    for i in range(30):
        x = opt.run_one_step(x, function_to_min)
        print("current position : %.3f loss= %.3f" % (x[0], function_to_min.f(x)))
        position_history.append(x)
    plot1D(position_history, function_to_min, title)

function_to_min = cosinusoidal()


opt = MCMC(sigma=1., lambd=1., iter_mcmc=1)
log_run(opt,function_to_min,"MCMC sigma=1 lambda=1")
opt = GD(lr=0.1)
log_run(opt,function_to_min,"GD lr=0.1")
