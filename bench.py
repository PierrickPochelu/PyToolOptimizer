import display
from function_to_min.F_absolute import absolute
from function_to_min.F_square import square
from function_to_min.F_rosenbrock import Rosenbrock
from function_to_min.F_deeplearning import Deeplearning
from functools import partial

from function_to_min.tensorflow_util.CNN_defaultCNNs import CNN_2conv,CNN_3conv

from optimizer.GD import GD
from optimizer.SPSA import SPSA
from optimizer.MCMC import MCMC
from optimizer import PSO
from optimizer.Newton import Newton

from function_to_min.tensorflow_util.DB_CIFAR10 import CIFAR10
from function_to_min.tensorflow_util.DB_MNIST import MNIST
import numpy as np
import time

#####################################################
import numpy as np
import matplotlib.pyplot as plt


def random_dir(nb_dimensions, sigma):
    x=np.zeros((nb_dimensions,))
    for i in range(nb_dimensions):
        rand_vec=np.random.uniform(size=nb_dimensions)
        rand_vec/=np.linalg.norm( rand_vec )
        rand_vec*=np.random.normal(0,sigma)
        x[i]=np.linalg.norm( rand_vec )
    return x

def draw(samples_time,samples_x,samples_title):
    for i in range(len(samples_title)):
        this_strategy_x=samples_x[i]
        this_strategy_name=samples_title[i]
        this_strategy_time=samples_time[i]
        plt.plot(this_strategy_time, this_strategy_x,label=this_strategy_name)
    plt.title('title')
    plt.xlabel("time (sec.)")
    plt.ylabel("crossentropy")
    plt.legend()
    plt.show()
###################################################


# FUNCTION TO MIN
batch_size_algo=5000
#db=CIFAR10(path=".\\function_to_min\\tensorflow_util")
db=MNIST(path=".\\function_to_min\\tensorflow_util")
np_dataset=db.get_np_dataset()
cnn=CNN_3conv(db.get_input_shape(), db.get_output_shape())
function_to_min=Deeplearning(np_dataset=np_dataset,
                             simpleCNN=cnn,
                             batch_size_accumulation=2500,
                             batch_size_algo=batch_size_algo,
                             mini_batch_mode=False)
nbDim=function_to_min.tensorflowObject.count_trainable_variables()



print(nbDim)


# INIT X
x = function_to_min.glorout_init_and_get_weights()
np.save("x.npy",x)


strategy_time=300

# OPTIMIZER
def a_priori(x):
    return np.random.normal(x, .1)
def a_priori2(x):
    return random_dir(x.shape[0], 0.1*x.shape[0])


opt_swarm=PSO(20, init_x_swarm=x, apriori_nn=partial(function_to_min.apriori_nn, sigma=4.),
                                    scale_init_vel=.1, momentum=0.5, my_vel_contrib=1., social_vel_contrib=1.)
opt_mcmc=MCMC(PI=function_to_min.apriori_nn,sigma=0.1, lambd=1e6, iter_mcmc=32)
opt_spsa=SPSA(lr=1.,epsilon_length=1.,PI=partial(function_to_min.apriori_nn,sigma=1.))
opt_gd=GD(lr=.1)
strategies=[opt_mcmc,opt_spsa,opt_swarm,opt_gd]

"""
MCMC 0.01
MCMC
15.5 ; 0.3246399 ; 0.3247813 ; 0.1017
30.8 ; 0.3229126 ; 0.3229401 ; 0.1098
46.1 ; 0.3212414 ; 0.3211697 ; 0.1636
61.5 ; 0.3188556 ; 0.3187045 ; 0.1781
77.0 ; 0.3162180 ; 0.3160482 ; 0.2177
92.4 ; 0.3119343 ; 0.3116348 ; 0.2485
107.9 ; 0.3078367 ; 0.3075043 ; 0.3358
123.7 ; 0.3035824 ; 0.3031427 ; 0.4030
139.4 ; 0.2989159 ; 0.2983868 ; 0.4076
154.9 ; 0.2935901 ; 0.2929737 ; 0.4239
170.4 ; 0.2862086 ; 0.2854106 ; 0.4830
185.9 ; 0.2792921 ; 0.2784518 ; 0.4810
201.4 ; 0.2701611 ; 0.2691756 ; 0.4958
216.9 ; 0.2610669 ; 0.2596090 ; 0.5442
232.4 ; 0.2533192 ; 0.2518361 ; 0.5310
247.9 ; 0.2437351 ; 0.2418278 ; 0.5959
263.4 ; 0.2335613 ; 0.2314807 ; 0.5979
278.9 ; 0.2238015 ; 0.2211333 ; 0.6142
294.4 ; 0.2156996 ; 0.2130682 ; 0.6421



1. 1.
30.3 ; 0.3257361 ; 0.3257309 ; 0.0819
45.5 ; 0.3256366 ; 0.3256299 ; 0.0881
60.6 ; 0.3255622 ; 0.3255545 ; 0.0921
75.7 ; 0.3254958 ; 0.3254880 ; 0.0958
90.8 ; 0.3254403 ; 0.3254312 ; 0.0983
106.0 ; 0.3253965 ; 0.3253874 ; 0.1010
121.2 ; 0.3253602 ; 0.3253507 ; 0.1010

10 1
less good

1 0.1
less good

1 10
less good

"""


mpl_strategy_name=[]
mpl_strategy_perf=[]
mpl_strategy_time=[]


# training this strategy
for opt in strategies:
    print(opt.get_name())
    mpl_strategy_name.append(opt.get_name())
    this_strategy_perf=[]
    this_strategy_time=[]
    x = np.load("x.npy")
    i=0
    start_time=time.clock()
    last_time_check=0
    while time.clock() - start_time < strategy_time:
        i+=1

        x = opt.run_one_step(x, function_to_min)


        # run each epoch
        if time.clock()-last_time_check > 15 and i!=0:
            last_time_check=time.clock()
            t=time.clock() - start_time
            losstrain=function_to_min.loss_trainDataset(x)
            losstest=function_to_min.loss_testDataset(x)
            acc=function_to_min.accuracy(x)
            this_strategy_perf.append(losstrain)
            this_strategy_time.append(t)
            print("%.1f ; %.7f ; %.7f ; %.4f" % (t,losstrain, losstest, acc))

    mpl_strategy_time.append(this_strategy_time)
    mpl_strategy_perf.append(this_strategy_perf)


draw(mpl_strategy_time,mpl_strategy_perf,mpl_strategy_name)

"""

swarm 2. 1.
18.5 ; 0.3249054 ; 0.3246822 ; 0.1081
36.9 ; 0.3213616 ; 0.3210662 ; 0.1240
55.4 ; 0.3214896 ; 0.3213346 ; 0.1994
73.8 ; 0.3198902 ; 0.3195977 ; 0.1826
92.2 ; 0.3180746 ; 0.3178751 ; 0.2418
110.5 ; 0.3129189 ; 0.3124284 ; 0.2350
128.9 ; 0.3113710 ; 0.3109285 ; 0.2405
147.2 ; 0.3042293 ; 0.3035573 ; 0.2407
165.5 ; 0.3004220 ; 0.2997361 ; 0.2801
183.8 ; 0.2986722 ; 0.2982668 ; 0.3041
202.2 ; 0.2946269 ; 0.2939098 ; 0.2990
220.5 ; 0.2899177 ; 0.2896000 ; 0.3381
239.0 ; 0.2864918 ; 0.2857572 ; 0.3752
257.4 ; 0.2860969 ; 0.2852899 ; 0.3667
275.6 ; 0.2830070 ; 0.2820473 ; 0.3512
293.8 ; 0.2800093 ; 0.2794352 ; 0.3722


"""