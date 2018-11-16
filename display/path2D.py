from itertools import product

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def plot2D(position_history, function_to_min, title=""):
    matplotlib.rcParams['xtick.direction'] = 'out'
    matplotlib.rcParams['ytick.direction'] = 'out'

    # draw 2D function
    X = np.arange(-1., +7., 0.05)
    Y = np.arange(-1., +7., 0.05)
    #X, Y = np.meshgrid(X, Y)

    couple_ij = product(range(X.shape[0]), range(Y.shape[0]))
    Z = np.zeros((X.shape[0],Y.shape[0]))
    for i, j in couple_ij:
        x=np.array([X[i], Y[j]])
        Z[i, j] = function_to_min.f(x)

    # draw isolines
    levels = np.array([8 ** i for i in range(20)])
    CS = plt.contour(X,
                     Y,
                     Z,
                     levels=levels,
                     )
    plt.clabel(CS, inline=0.1, fontsize=8)

    # draw path
    if isinstance(position_history,list):
        position_history=np.array(position_history)
    plt.plot(position_history[:, 0], position_history[:, 1], 'go-')

    plt.title(title)

    plt.show()


def PATH_nondeter(positionsx):
    matplotlib.rcParams['xtick.direction'] = 'out'
    matplotlib.rcParams['ytick.direction'] = 'out'

    X = np.arange(-1., +7., 0.05)
    Y = np.arange(-1., +7., 0.05)
    X, Y = np.meshgrid(X, Y)
    Z = (np.ones([np.shape(X)[0], np.shape(X)[1]]) - X) ** 2 + 100 * (Y - (X) ** 2) ** 2

    levels = np.array([0.125 * 2 ** i for i in range(50)])

    fig, ax = plt.subplots()

    CS = plt.contour(X,
                     Y,
                     Z,
                     levels=levels,
                     )

    plt.gca().set_color_cycle(['red', 'green', 'blue', 'yellow'])

    for i, posx in enumerate(positionsx[0:4]):
        posx_arr = np.array(posx)
        ax.plot(posx_arr[:, 0], posx_arr[:, 1], 'C' + str(i + 1), label='mcmc ' + str(i))

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title('')

    plt.show()


from function_to_min import F_rosenbrock
if __name__ == "__main__":
    plot2D(np.array([[0, 0], [0.1, 0.1], [0.2, 0.2], [0.4, 0.3], [0.8, 0.4]]), F_rosenbrock.Rosenbrock())
