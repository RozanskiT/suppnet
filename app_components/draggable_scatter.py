#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np


class DraggableScatter():

    epsilon = 5

    def __init__(self, ax, x, y, parent):

        self.parent = parent
        self._ispl = self.parent.spline
        self._ispl.set_knots(x, y)
        x, y = self._ispl.get_knots()
        xl, yl = self.get_line_data(x)

        self.scatter = ax.scatter(x, y, marker='o', c='r', zorder=50)
        self.line, = ax.plot(xl, yl, 'r', zorder=40)

        self._ind = None
        self.ax = ax
        self.toolbar = self.ax.figure.canvas.toolbar
        self.canvas = self.ax.figure.canvas
        self.canvas.mpl_connect('button_press_event',
                                self.button_press_callback)
        self.canvas.mpl_connect('button_release_event',
                                self.button_release_callback)
        self.canvas.mpl_connect('motion_notify_event',
                                self.motion_notify_callback)

    def get_line_data(self, knots_x):
        if len(knots_x) > 1:
            self.x_line = np.linspace(knots_x[0], knots_x[-1], 10*len(knots_x))
        else:
            self.x_line = []
        return self.x_line, self._ispl(self.x_line)

    def get_ind_under_point(self, event):
        xy = np.asarray(self.scatter.get_offsets())
        xyt = self.ax.transData.transform(xy)
        xt, yt = xyt[:, 0], xyt[:, 1]

        d = np.sqrt((xt - event.x)**2 + (yt - event.y)**2)
        ind = d.argmin()

        if d[ind] >= self.epsilon:
            ind = None

        return ind

    def button_press_callback(self, event):
        if self.toolbar.mode != '':
            # If zooming, etc. just return
            return
        if event.inaxes is None:
            return
        if self._ispl.is_empty():
            return
        if event.inaxes == self.ax:
            if event.button == 1:
                self._ind = self.get_ind_under_point(event)
            elif event.button == 2:
                self.add_point(event)
            elif event.button == 3:
                self.remove_nearest_point(event)

    def remove_nearest_point(self, event):
        self._ispl.remove_nearest_knot(event.xdata)
        self.parent.update_plots_and_data()

    def add_point(self, event):
        self._ispl.add_knot(event.xdata, event.ydata)
        self.parent.update_plots_and_data()

    def update_plot(self):
        x, y = self._ispl.get_knots()
        if len(x) > 1:
            xy = list(zip(x, y))
            self.x_line = np.linspace(x[0], x[-1], 30*len(x))
        else:
            xy = np.empty((0, 2))
        self.scatter.set_offsets(xy)

        xl, yl = self.get_line_data(x)
        self.line.set_data(xl, yl)
        self.canvas.draw_idle()

    def button_release_callback(self, event):
        if self.toolbar.mode != '':
            # If zooming, etc. just return
            return
        if event.button != 1:
            return
        self._ind = None

    def motion_notify_callback(self, event):
        if self.toolbar.mode != '':
            # If zooming, etc. just return
            return
        if self._ind is None:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        if event.inaxes == self.ax:
            x, y = event.xdata, event.ydata
            xy = np.asarray(self.scatter.get_offsets())

            self._ind = self._ispl.replace_knot(x, y, self._ind)
            self.parent.update_plots_and_data()


def main():
    pass


if __name__ == '__main__':
    main()
