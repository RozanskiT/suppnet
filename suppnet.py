import os
import sys
import time
import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline, InterpolatedUnivariateSpline
from tensorflow.keras.models import load_model

from PySide2.QtWidgets import QApplication, QMainWindow, QFileDialog, QLabel
from PySide2.QtCore import QFile, QThreadPool
from app_components.main_window_qt import Ui_MainWindow
from app_components.worker import Worker
from app_components.app_logic import Logic
from app_components.draggable_scatter import DraggableScatter

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import SpanSelector


class MainWindow(QMainWindow):
    def __init__(self, path=None, show_segmentation=False):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.filename_label = QLabel("")
        self.ui.statusbar.addPermanentWidget(self.filename_label)
        self.logic = Logic()
        self.spline = self.logic.spline
        self.show_segmentation = show_segmentation

        if path is None:
            self.threadpool = QThreadPool()
            self.for_threading()

        self.configure_slider()
        self.ui.slider_value.setText("1.00")
        self.ui.update_normalization.clicked.connect(
            self.on_update_normalization)

        self.addmpl()

        self.ui.actionOpen_spectrum.triggered.connect(self.load_spectrum)
        self.ui.actionSave_normed_spectrum.triggered.connect(
            self.save_normed_spectrum)
        self.ui.actionSave_results.triggered.connect(self.save_all_results)
        self.ui.actionOpen_processed_spectrum.triggered.connect(
            self.on_load_processed_spectrum)
        self.ui.actionClose.triggered.connect(QApplication.quit)

        self.ui.action_normalize.triggered.connect(self.normalize)

        if path is not None:
            self.logic.read_processed_spectrum(path)
            self.update_plots_and_data(resize=True)

    def addmpl(self):
        self.dpi = 100
        self.fig = Figure(dpi=self.dpi)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas,
                                         self.ui.centralwidget,
                                         coordinates=True
                                         )
        self.create_plots()
        self.ui.mplvl.addWidget(self.canvas)
        self.ui.mplvl.addWidget(self.toolbar)

        self.canvas.draw()

    def configure_slider(self):
        self.slider_minimum = 0.0
        self.slider_maximum = 2.0
        self.slider_number_of_steps = 200
        self.slider_ticks = self.slider_number_of_steps//10
        self.ui.horizontalSlider.setMinimum(0)
        self.ui.horizontalSlider.setMaximum(self.slider_number_of_steps)
        self.ui.horizontalSlider.setTickInterval(self.slider_ticks)
        self.ui.horizontalSlider.setSingleStep(1)

        self.ui.horizontalSlider.setValue(100)

        self.ui.horizontalSlider.valueChanged.connect(self.update_label)

    def update_label(self, position):
        slider_value = (position/self.slider_number_of_steps) * \
            (self.slider_maximum-self.slider_minimum)+self.slider_minimum
        self.ui.slider_value.setText(f"{slider_value:.2f}")

    def on_update_normalization(self):
        position = self.ui.horizontalSlider.value()
        slider_value = (position/self.slider_number_of_steps) * \
            (self.slider_maximum-self.slider_minimum)+self.slider_minimum
        self.logic.on_adjust_smooth_factor(slider_value)
        self.update_plots_and_data(resize=False)

    def for_threading(self):
        worker = Worker(*self.logic.get_model())
        worker.signals.result.connect(self.load_model)
        worker.signals.finished.connect(self.thread_complete)

        # Execute
        self.threadpool.start(worker)

    def thread_complete(self):
        self.ui.action_normalize.setEnabled(True)
        self.ui.statusbar.showMessage("Model loaded")

    def load_model(self, model):
        self.logic.set_model(model)

    # READ/WRITE FILES
    def load_spectrum(self):
        filename, filetype = QFileDialog.getOpenFileName(self, 'OpenFile')
        if filename:
            print(f"Reading spectrum: {filename}")
            self.logic.read_spectrum(filename)
            self.update_plots_and_data(resize=True)
            self.filename_label.setText(self.logic.opened_file_name)

    def on_load_processed_spectrum(self):
        filename, filetype = QFileDialog.getOpenFileName(self, 'OpenFile')
        if filename:
            print(f"Reading normed spectrum: {filename}")
            self.logic.read_processed_spectrum(filename)
            self.update_plots_and_data(resize=True)

    def save_all_results(self):
        default_fn = os.path.splitext(self.logic.opened_file_name)[0]+'.all'
        filename, filetype = QFileDialog.getSaveFileName(
            self, 'SaveFile', default_fn)
        if filename:
            print(f"Saving result to file: {filename}")
            self.logic.save_all_results(filename)
            print(f"{filename} saved!")

    def save_normed_spectrum(self,):
        default_fn = os.path.splitext(self.logic.opened_file_name)[0]+'.norm'
        filename, filetype = QFileDialog.getSaveFileName(
            self, 'SaveFile', default_fn)
        if filename:
            print(f"Saving result to file: {filename}")
            self.logic.save_normed_spectrum(filename)
            print(f"{filename} saved!")

    # PLOTTING

    def update_plots_and_data(self, resize=False):
        self.logic.update_all()
        spectrum = self.logic.get_plotting_data()
        wave = spectrum["wave"].values
        flux = spectrum["flux"].values
        self.line11.set_data(wave, flux)

        if self.logic.continuum is not None:
            self.line12.set_data(wave, self.logic.continuum)
        else:
            self.line12.set_data([], [])

        self.ds.update_plot()

        if self.logic.normed_flux is not None:
            self.line21.set_data(wave, self.logic.normed_flux)
            self.line22.set_data([wave[0], wave[-1]], [1, 1])

        else:
            self.line21.set_data([], [])
            self.line22.set_data([], [])

        if self.show_segmentation:
            if self.logic.segmentation is not None:
                self.line31.set_data(wave, self.logic.segmentation)
            else:
                self.line31.set_data([], [])

        if resize:
            self.ax1.set_autoscale_on(True)
            self.ax1.relim()
            self.ax1.autoscale_view(True, True, True)
            self.toolbar.update()
        self.canvas.draw()

    def normalize(self):
        self.ui.statusbar.showMessage("Normalising...")
        self.logic.compute_continuum()
        self.update_plots_and_data(resize=False)
        self.ui.statusbar.showMessage("Normalisation done.")

    def create_plots(self):
        self.fig.subplots_adjust(
            wspace=0.0, hspace=0.0, top=0.95, bottom=0.05, left=0.05, right=0.95)

        if self.show_segmentation:
            gs = GridSpec(6, 1)
        else:
            gs = GridSpec(5, 1)

        self.ax1 = self.fig.add_subplot(gs[:3])
        self.ax1.grid(True)
        self.line11, = self.ax1.plot([], [], 'k-', zorder=20)
        self.line12, = self.ax1.plot([], [], 'b-', zorder=30)
        self.ds = DraggableScatter(self.ax1, [], [], self)

        self.ax2 = self.fig.add_subplot(gs[3:5], sharex=self.ax1)
        self.ax2.grid(True)
        self.line21, self.line22, = self.ax2.plot([], [], 'k', [], [], 'b--')

        self.ax1.set_autoscaley_on(True)
        self.ax2.set_ylim([0.1, 2.0])

        if self.show_segmentation:
            self.ax3 = self.fig.add_subplot(gs[5:], sharex=self.ax1)
            self.ax3.grid(True)
            self.line31, = self.ax3.plot([], [], 'k', zorder=20)
            self.ax3.set_ylim([-0.1, 1.1])


def run_window_app(path=None, show_segmentation=False):
    app = QApplication(sys.argv)

    window = MainWindow(path=path, show_segmentation=show_segmentation)
    window.show()

    sys.exit(app.exec_())


def argument_parser():
    import argparse
    from argparse import RawTextHelpFormatter

    description = "\n".join([
        "Code for stellar spectrum normalisation based on neural network SUPPNet",
        " ",
        "Usage scenarios:",
        "1. Spectrum-by-spectrum normalisation using interactive app:", " ",
        "    python suppnet.py [--segmentation]",
        " ",
        "2. Normalisation of group of spectra without any supervision:", " ",
        "    python suppnet.py --quiet [--skip number_of_rows_to_skip] path_to_spectrum_1.txt [path_to_spectrum_2.txt ...]",
        " ",
        "3. Manual inspection and correction of previously normalised spectrum, SUPPNet will not be loaded (often used in pair with 2.):", " ",
        "    python suppnet.py [--segmentation] --path path_to_processing_results.all",
        " ",
    ])

    parser = argparse.ArgumentParser(
        description=description, formatter_class=RawTextHelpFormatter)

    parser.add_argument('--quiet',
                        dest='without_window_app',
                        action='store_false',
                        help='Do not open window app for manual normalisation.',
                        required=False
                        )

    if '--quiet' in sys.argv:
        parser.add_argument('file_names',
                            type=str,
                            nargs='+',
                            help='Spectra to be normed.')
        parser.add_argument('--skip',
                            nargs=1,
                            dest='skipRows',
                            required=False,
                            default=[0],
                            help="No of rows to skip.",
                            type=int)
    else:
        parser.add_argument('--segmentation',
                            dest='show_segmentation',
                            action='store_true',
                            required=False,
                            help='Display segmentation on the plot.')
        parser.add_argument('--path',
                            type=str,
                            default=[None],
                            nargs=1,
                            help='Normed spectrum to be automatically loaded for manual correction. Neural network will not be loaded!')

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = argument_parser()

    if args.without_window_app:
        run_window_app(path=args.path[0],
                       show_segmentation=args.show_segmentation)
    else:
        from suppnet.NN_utility import process_all_spectra
        process_all_spectra(args.file_names, skip_rows=args.skipRows[0])
