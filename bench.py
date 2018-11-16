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
from optimizer.Particleswarm import ParticleSwarmOptimization
from optimizer.Newton import Newton

from function_to_min.tensorflow_util.DB_CIFAR10 import CIFAR10
from function_to_min.tensorflow_util.DB_MNIST import MNIST
import numpy as np
import time

#####################################################
import numpy as np
import matplotlib.pyplot as plt

def draw(samples_time,samples_x,samples_title):
    for i in range(len(samples_title)):
        this_strategy_x=samples_x[i]
        this_strategy_name=samples_title[i]
        this_strategy_time=samples_time[i]
        plt.plot(this_strategy_time, this_strategy_x,label=this_strategy_name)
    plt.title('title')
    plt.xlabel("iterations")
    plt.ylabel("crossentropy")
    plt.legend()
    plt.show()
###################################################


# FUNCTION TO MIN
batch_size_algo=10000
#db=CIFAR10(path=".\\function_to_min\\tensorflow_util")
db=MNIST(path=".\\function_to_min\\tensorflow_util")
np_dataset=db.get_np_dataset()
cnn=CNN_3conv(db.get_input_shape(), db.get_output_shape())
function_to_min=Deeplearning(np_dataset=np_dataset,
                             simpleCNN=cnn,
                             batch_size_memory=2500,
                             batch_size_algo=batch_size_algo,
                             mini_batch_mode=False)
nbDim=function_to_min.tensorflowObject.count_trainable_variables()



print(nbDim)


# INIT X
x = function_to_min.glorout_init_and_get_weights()
np.save("x.npy",x)


strategy_time=300

# OPTIMIZER
PI=partial(function_to_min.apriori_nn,sigma=0.01,dof=2)
opt_swarm=ParticleSwarmOptimization(20, init_x_swarm=x, apriori_nn=partial(function_to_min.apriori_nn, sigma=4., dof=30),
                                    scale_init_vel=.1, momentum=0.5, my_vel_contrib=1., social_vel_contrib=1.)
opt_mcmc=MCMC(PI=PI, lambd=1e10, iter_mcmc=16)
opt_spsa=SPSA(lr=1e-1,epsilon_length=1e-1,PI=PI)
opt_gd=GD(lr=.1)
strategies=[opt_swarm,opt_mcmc,opt_spsa,opt_gd]




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