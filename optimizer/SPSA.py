from optimizer.optimizer_interface import optimizer_interface
import numpy as np

class SPSA(optimizer_interface):
    def __init__(self,lr,epsilon_length,PI):
        optimizer_interface.__init__(self)
        self.lr=lr
        self.epsilon_length=epsilon_length
        self.PI=PI

    def run_one_step0(self,x,function_to_min):
        fx=function_to_min.f(x)
        xa_epsilon = np.abs(self.PI(np.zeros(x.shape), sigma=self.epsilon_length, dof=30))
        xa_epsilon=(xa_epsilon/np.linalg.norm(xa_epsilon))
        xa_epsilon=self.epsilon_length*xa_epsilon

        #xb_epsilon = np.abs(self.PI(x, sigma=self.epsilon_length, dof=30))
        #xb_epsilon=(xb_epsilon/np.linalg.norm(xb_epsilon))
        xb_epsilon=-1*xa_epsilon

        x_proposed=x+xa_epsilon

        fx_proposed=function_to_min.f(x_proposed)
        coef_direction=(fx_proposed-fx)/(self.epsilon_length)# 0> if it's better

        new_x = x - self.lr * coef_direction

        return new_x

    def run_one_step(self,x,function_to_min):
        #fx=function_to_min.f(x)

        # compute delta value
        x_epsilon = np.abs(self.PI(np.zeros(x.shape)))/2.
        x_epsilon=(x_epsilon/np.linalg.norm(x_epsilon))
        x_epsilon=self.epsilon_length*x_epsilon



        fxa_proposed=function_to_min.f(x+x_epsilon)
        fxb_proposed=function_to_min.f(x-x_epsilon)

        coef_direction=(fxa_proposed-fxb_proposed)/(2*self.epsilon_length)

        new_x = x - self.lr * coef_direction

        return new_x