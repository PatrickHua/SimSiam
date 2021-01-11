import matplotlib
matplotlib.use('Agg') #https://stackoverflow.com/questions/49921721/runtimeerror-main-thread-is-not-in-main-loop-with-matplotlib-and-flask
import matplotlib.pyplot as plt
from collections import OrderedDict
from torch import Tensor

class Plotter(object):
    def __init__(self):
        self.logger = OrderedDict()
    def update(self, ordered_dict):
        for key, value in ordered_dict.items():
            if isinstance(value, Tensor):
                ordered_dict[key] = value.item()
            if self.logger.get(key) is None:
                self.logger[key] = [value]
            else:
                self.logger[key].append(value)

    def save(self, file, **kwargs):
        fig, axes = plt.subplots(nrows=len(self.logger), ncols=1, figsize=(8,2*len(self.logger)))
        fig.tight_layout()
        for ax, (key, value) in zip(axes, self.logger.items()):
            ax.plot(value)
            ax.set_title(key)

        plt.savefig(file, **kwargs)
        plt.close()








