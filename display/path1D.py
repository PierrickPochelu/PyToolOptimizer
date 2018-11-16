import numpy as np
import matplotlib.pyplot as plt







def plot1D(position_history,function_to_min,title=""):


    plt.figure(1)

    # plot search
    value_history=[function_to_min.f(x) for x in position_history]
    #plt.plot(position_history, value_history, '-o')
    for i, (x, y) in enumerate(zip(position_history, value_history)):
        plt.plot(x[0], y,'-o',color="red")
        plt.text(x[0], y, str(i), color="red", fontsize=12)

    # plot function
    function_x = np.arange(-6.0, 6.0, 0.01)
    function_y = [function_to_min.f(np.array([x])) for x in function_x]
    plt.plot(function_x, function_y)

    # initialisation
    """
    init_x=position_history[0]
    plt.plot(init_x, function_to_min.f(init_x), 'bo')
    plt.annotate('initialization', xy=(init_x, function_to_min.f(init_x)),
                 xytext=(init_x, function_to_min.f(init_x)+10),
                arrowprops=dict(facecolor='black', shrink=0.01),
                )
    """

    plt.title(title, fontsize=20)
    plt.show()

from function_to_min import F_square
if __name__=="__main__":
    position_history=[np.array([5.]),np.array([3.]),np.array([2.])]
    f=F_square.square()
    plot1D(position_history,f)