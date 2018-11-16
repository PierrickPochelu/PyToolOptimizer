from optimizer.optimizer_interface import optimizer_interface


class GD(optimizer_interface):
    def __init__(self,lr):
        optimizer_interface.__init__(self)
        self.lr=lr

    def run_one_step(self,x,function_to_min):
        return x-self.lr*function_to_min.df(x)