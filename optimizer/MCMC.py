from optimizer import Interface_optimizer
import numpy as np
from functools import partial

class MCMC(Interface_optimizer):
    def __init__(self, sigma, PI=None, lambd=1e-3, iter_mcmc=1):
        Interface_optimizer.__init__(self)
        self.lambd=lambd
        self.fx=None

        if PI is None:
            def PI(x,sigma):
                return np.random.normal(x,sigma)
        self.init_sigma=sigma
        self.sigma=sigma
        self.PI=partial(PI,sigma=self.sigma)

        self.last_it_accept=0
        self.iterMCMC=iter_mcmc


    def run_one_step(self,x,function_to_min):
        #if self.fx is None: # first time
        self.fx=function_to_min.f(x)


        self.last_it_accept=0

        for i in range(self.iterMCMC):
            xp = self.PI(x)
            fx_prop = function_to_min.f(xp)

            # accept or reject ?
            accept_rate = -self.lambd * (fx_prop - self.fx)
            accept = np.log(np.random.uniform(0., 1.)) < accept_rate

            # update x
            if accept:
                x = xp
                self.fx=fx_prop
                self.last_it_accept+=1

        if self.last_it_accept==0 and self.iterMCMC>1:
            self.sigma*=0.1
            self.PI=partial(function_to_min.apriori_nn,sigma=self.sigma)
            print("sigma=" + str(self.sigma))



        return x

    def next_batch(self):
        """
        When we change batch we must reset self.fx to not compare loss between two different batch
        :return:
        """
        self.fx=None
    def reset_sigma(self):
        self.sigma=self.init_sigma