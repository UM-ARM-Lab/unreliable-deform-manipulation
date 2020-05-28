from typing import Callable

import matplotlib.pyplot as plt
import matplotlib.widgets
import mpl_toolkits.axes_grid1
import numpy as np
from matplotlib.animation import FuncAnimation


class Player(FuncAnimation):
    def __init__(self, fig: plt.Figure,
                 func: Callable,
                 max_index: int,
                 min_index: int = 0,
                 repeat: bool = True,
                 pos=(0.125, 0.92),
                 init_func=None,
                 fargs=None,
                 save_count=None,
                 **kwargs):
        """
        :param fig:
        :param func:
        :param max_index: exclusive
        :param min_index: inclusive
        :param repeat:
        :param pos:
        :param init_func:
        :param fargs:
        :param save_count:
        :param kwargs:
        """
        self.i = 0
        self.min = min_index
        self.max = max_index
        self.runs = True
        self.forwards = True
        self.fig = fig
        self.repeat = repeat
        self.func = func
        self.button_oneback = None
        self.button_back = None
        self.button_stop = None
        self.button_forward = None
        self.button_oneforward = None
        self.setup(pos)
        FuncAnimation.__init__(self,
                               self.fig,
                               self.func,
                               frames=self.play(),
                               init_func=init_func,
                               fargs=fargs,
                               save_count=save_count,
                               **kwargs)

    def play(self):
        while self.runs:
            self.i = self.i + self.forwards - (not self.forwards)
            if self.min < self.i < self.max:
                yield self.i
            elif self.repeat:
                self.i = self.min
                yield self.i
            else:
                self.stop()
                yield self.i

    def start(self):
        self.runs = True
        self.event_source.start()

    def stop(self, event=None):
        self.runs = False
        self.event_source.stop()

    def forward(self, event=None):
        self.forwards = True
        self.start()

    def backward(self, event=None):
        self.forwards = False
        self.start()

    def oneforward(self, event=None):
        self.forwards = True
        self.onestep()

    def onebackward(self, event=None):
        self.forwards = False
        self.onestep()

    def onestep(self):
        if self.forwards:
            if self.min <= self.i < self.max - 1:
                self.i += 1
        else:
            if self.min < self.i < self.max:
                self.i -= 1
        self.func(self.i)
        self.fig.canvas.draw_idle()

    def setup(self, pos):
        playerax = self.fig.add_axes([pos[0], pos[1], 0.22, 0.04])
        divider = mpl_toolkits.axes_grid1.make_axes_locatable(playerax)
        bax = divider.append_axes("right", size="80%", pad=0.05)
        sax = divider.append_axes("right", size="80%", pad=0.05)
        fax = divider.append_axes("right", size="80%", pad=0.05)
        ofax = divider.append_axes("right", size="100%", pad=0.05)
        self.button_oneback = matplotlib.widgets.Button(playerax, label=u'$\u29CF$')
        self.button_back = matplotlib.widgets.Button(bax, label=u'$\u25C0$')
        self.button_stop = matplotlib.widgets.Button(sax, label=u'$\u25A0$')
        self.button_forward = matplotlib.widgets.Button(fax, label=u'$\u25B6$')
        self.button_oneforward = matplotlib.widgets.Button(ofax, label=u'$\u29D0$')
        self.button_oneback.on_clicked(self.onebackward)
        self.button_back.on_clicked(self.backward)
        self.button_stop.on_clicked(self.stop)
        self.button_forward.on_clicked(self.forward)
        self.button_oneforward.on_clicked(self.oneforward)


if __name__ == '__main__':
    fig, ax = plt.subplots()
    x = np.linspace(0, 6 * np.pi, num=100)
    y = np.sin(x)

    ax.plot(x, y)
    point, = ax.plot([], [], marker="o", color="crimson", ms=15)


    def update(i):
        point.set_data(x[i], y[i])


    ani = Player(fig, update, max_index=len(y) - 1)

    plt.show()
