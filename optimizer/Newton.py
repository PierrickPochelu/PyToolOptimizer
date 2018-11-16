from optimizer.optimizer_interface import optimizer_interface


class Newton(optimizer_interface):
    def __init__(self):
        optimizer_interface.__init__(self)

    def run_one_step(self,x,function_to_min):
        ddfx=function_to_min.ddf(x)
        dfx=function_to_min.df(x)
        return x-(dfx/ddfx)