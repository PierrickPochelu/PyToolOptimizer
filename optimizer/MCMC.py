from optimizer.optimizer_interface import optimizer_interface
import numpy as np

class MCMC(optimizer_interface):
    def __init__(self, PI=None, lambd=1e-3, iter_mcmc=1):
        optimizer_interface.__init__(self)
        self.lambd=lambd
        self.fx=None
        self.PI=PI
        self.last_it_accept=None
        self.iterMCMC=iter_mcmc

    def run_one_step(self,x,function_to_min):
        #if self.fx is None: # first time
        self.fx=function_to_min.f(x)

        # compute proposal
        #if self.PI is None:
        #    xp = np.random.standard_t(self.dof, x.shape) * self.sigma + x
        #else:

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


        return x

    def next_batch(self):
        """
        When we change batch we must reset self.fx to not compare loss between two different batch
        :return:
        """
        self.fx=None