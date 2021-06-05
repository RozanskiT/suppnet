#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import sys
import os
from scipy.interpolate import UnivariateSpline
from tensorflow.keras.models import load_model

from suppnet.NN_utility import ProcessSpectrum, MinMaxNormalizer
from app_components.interactive_spline import InteractiveSpline
from suppnet.SUPPNet import get_suppnet_model


class Logic:
    def __init__(self):
        self.spectrum = None
        self.result = None
        self.nn = None
        self.opened_file_name = None

        self.continuum = None
        self.continuum_error = None
        self.smoothed_continuum = None
        self.normed_flux = None
        self.normed_flux_err = None
        self.segmentation = None
        self.segmentation_error = None

        self.normed_spectrum_noise = None

        self.step_size = 256
        self.window_len = 8192

        self.spline = InteractiveSpline()
        self.smooth_factor = 1.0

    def get_model(self):
        return get_suppnet_model, False

    def read_spectrum(self, filename):
        self.spectrum = pd.read_csv(filename,
                                    index_col=None,
                                    delim_whitespace=True,
                                    comment="#"
                                    )
        self.opened_file_name = os.path.basename(filename)
        self.normed_spectrum_noise = None
        self.continuum = None
        self.continuum_error = None
        self.spline.set_knots([], [])
        self.normed_flux = None
        self.normed_flux_err = None
        self.segmentation = None
        self.segmentation_error = None
        if len(self.spectrum.columns) == 2:
            self.spectrum.columns = ['wave', 'flux']
        elif len(self.spectrum.columns) == 3:
            self.spectrum.columns = ['wave', 'flux', 'error']
        else:
            self.spectrum.columns = [
                'wave', 'flux', 'error'] + [x for x in self.spectrum.columns[3:]]

    def read_processed_spectrum(self, filename):
        data = pd.read_csv(filename,
                           index_col=None,
                           delim_whitespace=True,
                           comment="#"
                           )
        # COLUMNS:
        # wave flux normed_flux normed_error continuum continuum_err segmentation segmentation_err
        self.spectrum = data[["wave", "flux"]]
        self.opened_file_name = os.path.basename(filename)
        self.continuum = data["continuum"].values
        self.continuum_error = data["continuum_err"].values
        self.segmentation = data["segmentation"].values
        self.segmentation_error = data["segmentation_err"].values

        self.normed_spectrum_noise = None
        self.spline.set_knots([], [])
        self.normed_flux = None
        self.normed_flux_err = None

        self.fit_spline()
        print(f"Normed spectrum noise = {self.normed_spectrum_noise:.4f}")

    def update_all(self):
        self.compute_normed_spectrum()

    def set_model(self, model):
        self.nn = ProcessSpectrum(model,
                                  MinMaxNormalizer(),
                                  step_size=self.step_size,
                                  window_len=self.window_len
                                  )

    def on_adjust_smooth_factor(self, smooth_factor=1.0):
        self.smooth_factor = smooth_factor
        self.fit_spline()

    def save_normed_spectrum(self, filename):
        wave = self.spectrum['wave']
        flux = self.spectrum['flux']
        df = pd.DataFrame({
            "wave": wave,
            "flux": flux,
            "normed_flux": self.normed_flux,
            "normed_error": self.normed_flux_err
        })
        mask = (df['flux'] == 0)
        df.loc[mask, df.columns != 'wave'] = 0.
        df.to_csv(filename, sep=' ', index=False)

    def save_all_results(self, filename):
        wave = self.spectrum['wave']
        flux = self.spectrum['flux']
        df = pd.DataFrame({"wave": wave,
                           "flux": flux,
                           "normed_flux": self.normed_flux,
                           "normed_error": self.normed_flux_err,
                           "smoothed_continuum": self.smoothed_continuum,
                           "continuum": self.continuum,
                           "continuum_err": self.continuum_error,
                           "segmentation": self.segmentation,
                           "segmentation_err": self.segmentation_error
                           })
        mask = (df['flux'] == 0)
        df.loc[mask, df.columns != 'wave'] = 0.
        df.to_csv(filename, sep=' ', index=False)

    def compute_continuum(self):
        self.continuum, self.continuum_error, self.segmentation, self.segmentation_error = self.nn.normalize(
            self.spectrum["wave"].values, self.spectrum["flux"].values)
        self.fit_spline()
        return

    def fit_spline(self):
        knots_x, knots_y = self.fit_smoothing_spline(
            self.spectrum["wave"].values, self.continuum, self.continuum_error*self.smooth_factor)
        self.spline.set_knots(knots_x, knots_y)
        self.update_all()

    def noise_value(self, v):
        # signal = np.nanmedian(v)
        noise = 1.482602 / \
            np.sqrt(6.0) * np.nanmedian(np.abs(2*v -
                                               np.roll(v, -2) - np.roll(v, +2)))
        return noise

    def compute_normed_spectrum(self):
        self.smoothed_continuum = self.spline(self.spectrum['wave'])
        if len(self.smoothed_continuum) != 0:
            self.normed_flux = self.spectrum['flux']/self.smoothed_continuum
            self.normed_flux_err = self.continuum_error/self.smoothed_continuum
            self.normed_spectrum_noise = self.noise_value(self.normed_flux)

    def fit_smoothing_spline(self, wave, continuum, continuum_std):
        mask = ~(np.isclose(continuum_std, 0) | np.isclose(
            continuum, 0) | np.isnan(continuum_std) | np.isnan(continuum))
        wave = wave[mask]
        continuum = continuum[mask]
        continuum_std = continuum_std[mask]
        weights = 1./continuum_std
        spl = UnivariateSpline(wave,
                               continuum,
                               w=weights,
                               bbox=[None, None],
                               k=3,
                               s=None,
                               ext=0,
                               check_finite=False)
        knots = spl.get_knots()
        return knots, spl(knots)

    def get_plotting_data(self):
        return self.spectrum


def main():
    pass


if __name__ == '__main__':
    main()
