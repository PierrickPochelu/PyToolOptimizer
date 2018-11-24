from optimizer import Interface_optimizer


class Newton(Interface_optimizer):
    def __init__(self):
        Interface_optimizer.__init__(self)

    def run_one_step(self,x,function_to_min):
        ddfx=function_to_min.ddf(x)
        dfx=function_to_min.df(x)
        return x-(dfx/ddfx)