import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np

class MyPlot(object):

    @staticmethod
    def plot_pair_curves(cx1, cy1, cx2, cy2, l1=None, l2=None):
        plt.plot(cx1, cy1, label=l1)
        plt.plot(cx2, cy2, label=l2)
        plt.legend()
        plt.show()

    @staticmethod
    def plot_curve(x, y, l):
        plt.plot(x, y, label=l)
        plt.legend()
        plt.show()


if __name__ == '__main__':
    X = np.linspace(-np.pi, np.pi, 100)
    MyPlot.plot_pair_curves(X, np.sin(X), X, np.cos(X), l1='sin', l2='cos')