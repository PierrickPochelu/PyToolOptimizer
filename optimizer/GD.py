from optimizer import Interface_optimizer


class GD(Interface_optimizer):
    def __init__(self,lr):
        Interface_optimizer.__init__(self)
        self.lr=lr

    def run_one_step(self,x,function_to_min):
        return x-self.lr*function_to_min.df(x)