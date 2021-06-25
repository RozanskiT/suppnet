import pandas as pd
from scipy.interpolate import UnivariateSpline
from numpy import ma
import numpy as np
import os
from scipy.interpolate import interp1d
from suppnet.SUPPNet import get_suppnet_model


class ProcessSpectrum:

    def __init__(self, model, normalizer, step_size=64, window_len=8192, resampling_step=0.05):
        self.normalizer = normalizer
        self.model = model

        self.window_len = window_len
        self.step_size = step_size
        self.only_norm = self.model.norm_only
        self.resampling_step = resampling_step

    def prepare_data(self, y):
        shifts = np.arange(0, self.window_len, self.step_size)

        y_shape = y.shape[0]
        pad_number = 2*self.window_len - y_shape % self.window_len
        padded_y = np.pad(y, (0, pad_number), mode='constant',
                          constant_values=np.nan)

        padded_all = np.stack([np.roll(padded_y, shift) for shift in shifts])
        for_processing = padded_all.reshape((-1, self.window_len))
        return for_processing, shifts

    def get_results(self, processed, shifts):
        reshaped = np.array([np.roll(part, shift=-shift).flatten()
                             for part, shift in zip(np.split(processed, shifts.shape[0]), shifts)])
        w = self.generate_weights(sigma=3, length=reshaped.shape[0])
        return self.weighted_avg_and_std(reshaped, w)
        # return np.nanmean(reshaped, axis=0), np.nanstd(reshaped, axis=0)

    def weighted_avg_and_std(self, values, weights):
        values = np.ma.masked_array(values, np.isnan(values))
        average = np.ma.average(values, weights=weights, axis=0)
        variance = np.ma.average((values-average)**2, weights=weights, axis=0)
        return average, np.sqrt(variance)

    def generate_weights(self, sigma, length):
        x = np.linspace(-3, 3, length)
        weights = np.exp(-(x/sigma)**2)
        return weights

    def resample(self, wave, flux):
        wave = np.array(wave)
        flux = np.array(flux)
        no_samples = int((wave[-1]-wave[0])/self.resampling_step)
        new_wave = np.linspace(wave[0], wave[-1], num=no_samples)
        mask_nnan = np.isnan(flux)
        flux[mask_nnan] = 0.
        new_flux = np.interp(new_wave, wave, flux)
        return new_wave, new_flux

    def normalize(self, wave, flux):
        new_wave, new_flux = self.resample(wave, flux)
        flux_prepared, shifts = self.prepare_data(new_flux)

        normed_flux = self.normalizer.normalize(flux_prepared)
        normed_flux = normed_flux[..., None]
        result = self.model.predict(normed_flux)
        length = new_wave.shape[0]
        if self.only_norm:
            continuum, continuum_std = self.process_signal(
                result, shifts, length, norm=True)
            f_cont = interp1d(new_wave, continuum, kind='linear')
            f_cont_err = interp1d(new_wave, continuum_std, kind='linear')
            return f_cont(wave), f_cont_err(wave)
        else:
            continuum, continuum_std = self.process_signal(
                result["cont"], shifts, length, norm=True)
            segmentation, segmentation_std = self.process_signal(
                result["seg"], shifts, length, norm=False)
            f_seg = interp1d(new_wave, segmentation, kind='linear')
            f_seg_err = interp1d(new_wave, segmentation_std, kind='linear')
            f_cont = interp1d(new_wave, continuum, kind='linear')
            f_cont_err = interp1d(new_wave, continuum_std, kind='linear')
            return f_cont(wave), f_cont_err(wave), f_seg(wave), f_seg_err(wave)

    def process_signal(self, result, shifts, length, norm=True):
        processed = np.squeeze(result)
        if norm:
            processed = self.normalizer.denormalize(processed)

        processed, continuum_std = self.get_results(processed, shifts)
        processed = processed[:length]
        processed_std = continuum_std[:length]
        processed[processed < 0.] = 0.
        processed[np.isnan(processed)] = 0.
        return processed, processed_std


class MinMaxNormalizer:
    def __init__(self):
        self._max = None
        self._min = None

    def normalize(self, X, Y=None):
        masked_values = (X > 0)
        X_masked = ma.masked_array(X, mask=~masked_values)
        self._min = ma.min(X_masked, axis=1, keepdims=True)
        self._max = ma.max(X_masked, axis=1, keepdims=True)
        X_normed = np.where(masked_values, (X-self._min) /
                            (self._max-self._min), 0)
        if Y is not None:
            return X_normed, (Y-self._min)/(self._max-self._min)
        else:
            return X_normed

    def denormalize(self, Y):
        return Y*(self._max - self._min) + self._min


# Implement normalization script:


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Normalise some stellar spectra.')
    parser.add_argument('file_names', type=str, nargs='+',
                        help='Spectra to be normed')
    parser.add_argument('-s', '--skip', nargs=1, dest='skipRows',
                        required=False, default=[0], help="No of rows to skip.", type=int)

    args = parser.parse_args()

    process_all_spectra(args.file_names, skip_rows=args.skipRows[0])


def process_all_spectra(paths, skip_rows):
    print("==================================\nNeural network loading\n==================================\n")
    nn = load_nn()
    print("==================================\nNeural network loaded\n==================================\n")

    for sfn in paths:
        out_path = os.path.splitext(sfn)[0]+'.all'
        print(f"Processing {sfn} -> {out_path}")
        spectrum = pd.read_csv(sfn,
                               index_col=None,
                               header=None,
                               delim_whitespace=True,
                               skiprows=skip_rows,
                               comment="#")
        process_spectrum(spectrum, out_path, nn)


def load_nn():
    from tensorflow.keras.models import load_model
    model = get_suppnet_model(norm_only=False)
    nn = ProcessSpectrum(model,
                         MinMaxNormalizer(),
                         step_size=256,
                         window_len=8192
                         )
    return nn

def get_suppnet(resampling_step=0.05, step_size=256, norm_only=True):
    """
    Returns ProcessSpectrum object that can be used for pseudo-continuum prediction:
    continuum, continuum_error = nn.normalize(wave, flux)
    when norm_only=False:
    continuum, continuum_error, segmentation, segmentation_error = nn.normalize(wave, flux)
    """
    model = get_suppnet_model(norm_only=norm_only)
    nn = ProcessSpectrum(model,
                         MinMaxNormalizer(),
                         step_size=step_size,
                         window_len=8192,
                         resampling_step=resampling_step
                         )
    return nn 


def process_spectrum(spectrum, filename, nn):
    wave = spectrum[0].values
    flux = spectrum[1].values
    if nn.only_norm:
        cont, cont_err = nn.normalize(wave, flux)
    else:
        cont, cont_err, seg, seg_err = nn.normalize(wave, flux)
    cont_smo = get_smoothed_continuum(wave, cont, cont_err)
    normed_flux = flux/cont_smo
    normed_flux_error = cont_err/cont_smo
    if nn.only_norm:
        save_results_norm(filename, wave, flux, normed_flux,
                          normed_flux_error, cont_smo, cont, cont_err)
    else:
        save_results_both(filename, wave, flux, normed_flux,
                          normed_flux_error, cont_smo, cont, cont_err, seg, seg_err)


def backend_normed_spectrum(wave, flux, nn):
    if nn.only_norm:
        cont, cont_err = nn.normalize(wave, flux)
        return cont, cont_err
    else:
        cont, cont_err, seg, seg_err = nn.normalize(wave, flux)
        return cont, cont_err, seg, seg_err


def get_smoothed_continuum(wave_orig, continuum, continuum_std):
    mask = ~(np.isclose(continuum_std, 0) | np.isclose(continuum, 0)
             | np.isnan(continuum_std) | np.isnan(continuum))
    wave = wave_orig[mask]
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

    return spl(wave_orig)


def save_results_norm(filename, wave, flux, normed_flux, normed_flux_err, cont_smo, continuum, continuum_err):
    df = pd.DataFrame({"wave": wave,
                       "flux": flux,
                       "normed_flux": normed_flux,
                       "normed_error": normed_flux_err,
                       "smoothed_continuum": cont_smo,
                       "continuum": continuum,
                       "continuum_err": continuum_err,
                       })
    mask = (df['flux'] == 0)
    df.loc[mask, df.columns != 'wave'] = 0.
    df.to_csv(filename, sep=' ', index=False)


def save_results_both(filename, wave, flux, normed_flux, normed_flux_err, cont_smo, continuum, continuum_err, seg, seg_err):
    df = pd.DataFrame({"wave": wave,
                       "flux": flux,
                       "normed_flux": normed_flux,
                       "normed_error": normed_flux_err,
                       "smoothed_continuum": cont_smo,
                       "continuum": continuum,
                       "continuum_err": continuum_err,
                       "segmentation": seg,
                       "segmentation_err": seg_err
                       })
    mask = (df['flux'] == 0)
    df.loc[mask, df.columns != 'wave'] = 0.
    df.to_csv(filename, sep=' ', index=False)


if __name__ == "__main__":
    # eg. python NN_utility.py example_data/*dat
    main()
