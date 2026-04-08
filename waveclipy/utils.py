"""utility functions: corner frequency, butterworth filter, husid plot."""

import numpy as np
from scipy.signal import butter, sosfilt


def corner_frequency(mw, beta=3500.0, d_sig=10e6):
    """estimate corner frequency from magnitude (Boore, 2003).

    equation 2 of Edwards et al. (2025):
        fc = 0.4906 * beta * (d_sig / 10^(1.5*Mw + 9.051))^(1/3)

    Args:
        mw    (float): moment magnitude
        beta  (float): shear velocity at source [m/s]
        d_sig (float): stress parameter [Pa]

    Returns:
        float: corner frequency fc [Hz]
    """
    fc = 0.4906 * beta * (d_sig / 10 ** (1.5 * mw + 9.051)) ** (1.0 / 3.0)
    return fc


def segment_duration(fc, cl=10.0):
    """compute segment duration (eq. 1).

    Eq 1: dt_p = cl / fc

    Args:
        fc (float): corner frequency [Hz]
        cl (float): segment length constant

    Returns:
        float: segment duration [s]
    """
    return cl / fc


def butterworth_lowpass(data, cutoff, fs, order=4):
    """apply a 4th-order butterworth low-pass filter.

    Args:
        data   (ndarray): input signal
        cutoff (float):   cutoff frequency [Hz]
        fs     (float):   sampling rate [Hz]
        order  (int):     filter order

    Returns:
        ndarray: filtered signal
    """
    nyq  = 0.5 * fs
    norm = cutoff / nyq
    if norm >= 1.0:
        return data.copy()
    sos = butter(order, norm, btype="low", output="sos")
    return sosfilt(sos, data)


def butterworth_lowpass_freq(spectrum, freqs, cutoff, order=4):
    """apply butterworth low-pass in frequency domain.

    convolves spectrum with complex frequency-domain 4th-order
    butterworth response.

    Args:
        spectrum (ndarray): complex Fourier coefficients
        freqs    (ndarray): frequency axis [Hz]
        cutoff   (float):   cutoff frequency [Hz]
        order    (int):     filter order

    Returns:
        ndarray: filtered spectrum
    """
    s    = 1j * freqs / cutoff
    resp = 1.0 / (1.0 + s ** (2 * order))
    h    = np.abs(resp)
    return spectrum * h


def husid_plot(acc, dt):
    """compute husid plot (cumulative arias intensity vs time).

    Args:
        acc (ndarray): acceleration time series
        dt  (float):   time step [s]

    Returns:
        time (ndarray): time axis [s]
        hsd  (ndarray): normalized cumulative arias intensity [0-1]
    """
    ai   = np.cumsum(acc ** 2) * dt * np.pi / (2.0 * 9.81)
    hsd  = ai / ai[-1] if ai[-1] > 0 else ai
    time = np.arange(len(acc)) * dt
    return time, hsd


def fix_wraparound_clipping(data, clip_flag):
    """assign clipped samples the polarity of the most recent unclipped sample.

    handles wrap-around clipping as a preprocessing step.

    Args:
        data      (ndarray): waveform in counts
        clip_flag (ndarray): boolean, True = clipped

    Returns:
        ndarray: corrected waveform
    """
    out  = data.copy().astype(float)
    last = 1.0
    for i in range(len(out)):
        if not clip_flag[i]:
            last = np.sign(out[i]) if out[i] != 0 else last
        else:
            out[i] = last * np.abs(out[i])
    return out
