import matplotlib
matplotlib.use('Agg') #https://stackoverflow.com/questions/49921721/runtimeerror-main-thread-is-not-in-main-loop-with-matplotlib-and-flask
import matplotlib.pyplot as plt
from collections import OrderedDict


class PlotLogger(object):
    def __init__(self, params=['loss']):
        self.logger = OrderedDict({param:[] for param in params})
    def update(self, ordered_dict):
        # self.logger.keys()
        assert set(ordered_dict.keys()).issubset(set(self.logger.keys()))
        for key, value in ordered_dict.items():
            self.logger[key].append(value)

    def save(self, file, **kwargs):
        fig, axes = plt.subplots(nrows=len(self.logger), ncols=1)
        fig.tight_layout()
        for ax, (key, value) in zip(axes, self.logger.items()):
            ax.plot(value)
            ax.set_title(key)

        plt.savefig(file, **kwargs)
        plt.close()





if __name__ == "__main__":
    logger = PlotLogger(params=['loss', 'accuracy', 'epoch'])
    import random
    epochs = 100
    n_iter = 1000
    for epoch in range(epochs):
        for idx in range(n_iter):
            stuff = {'loss': random.random(), 'accuracy':random.random(), 'epoch': epoch}
            logger.update(stuff)

    logger.save('./logger.png')


