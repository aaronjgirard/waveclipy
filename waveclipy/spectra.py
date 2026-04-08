"""spectral computations: PSA, FAS, PSD with uncertainty propagation.

computes 5% damped pseudo-spectral acceleration, fourier amplitude
spectrum, and power spectral density, with uncertainty bounds derived
from temporal uncertainty of the CWRdt reconstruction.
"""

import numpy as np
from scipy.signal import lfilter


## standard periods for PSA (Edwards et al., 2025)
PSA_PERIODS = np.array([
    0.01,  0.02,  0.025, 0.03,  0.04,  0.05,  0.075,
    0.1,   0.12,  0.125, 0.15,  0.17,  0.175, 0.2,
    0.25,  0.3,   0.4,   0.5,   0.6,   0.7,   0.75,
    0.85,  1.0,
])


def _sdof_response(acc, dt, period, damp=0.05):
    """compute SDOF response for a single period via Duhamel integral.

    uses the recursive convolution method (piecewise-linear excitation)
    which is unconditionally stable.

    Args:
        acc    (ndarray): acceleration time series
        dt     (float):   time step [s]
        period (float):   natural period [s]
        damp   (float):   damping ratio (default 0.05)

    Returns:
        float: peak absolute acceleration (pseudo-spectral acceleration)
    """
    if period < 1e-10:
        return np.max(np.abs(acc))

    w   = 2.0 * np.pi / period
    wd  = w * np.sqrt(1.0 - damp ** 2)
    w2  = w * w
    n   = len(acc)

    ## precompute recurrence coefficients (piecewise linear)
    e_dt = np.exp(-damp * w * dt)
    s_dt = np.sin(wd * dt)
    c_dt = np.cos(wd * dt)

    a11 = e_dt * (c_dt + damp * w / wd * s_dt)
    a12 = e_dt * s_dt / wd
    a21 = -e_dt * w2 / wd * s_dt
    a22 = e_dt * (c_dt - damp * w / wd * s_dt)

    ## load interpolation coefficients for piecewise linear
    zw  = damp * w
    b11 = e_dt * (
        (2 * zw / (w2 * dt) + (1 - 2 * damp**2) / (wd * dt) * s_dt
         - (1 + 2 * zw / (w2 * dt)) * c_dt)
    ) + 2 * zw / (w2 * dt)

    b12 = -e_dt * (
        (2 * zw / (w2 * dt)) * (1 - c_dt)
        + (1 - 2 * damp**2) / (wd * dt) * s_dt
    ) + 1.0 / w2 - 2 * zw / (w2 * dt)

    ## step through time
    u  = 0.0
    v  = 0.0
    sd = 0.0

    for i in range(1, n):
        u_new = a11 * u + a12 * v + b11 * (-acc[i-1]) + b12 * (-acc[i])
        v_new = a21 * u + a22 * v

        u = u_new
        v = v_new
        if abs(u) > sd:
            sd = abs(u)

    return sd * w2


def compute_psa(acc, dt, periods=None, damp=0.05):
    """compute 5% damped pseudo-spectral acceleration.

    Args:
        acc     (ndarray): acceleration time series
        dt      (float):   time step [s]
        periods (ndarray): oscillator periods [s] (default: paper's 23 periods)
        damp    (float):   damping ratio

    Returns:
        periods (ndarray): periods [s]
        psa     (ndarray): PSA values at each period
    """
    if periods is None:
        periods = PSA_PERIODS.copy()

    psa = np.array([_sdof_response(acc, dt, t, damp) for t in periods])
    return periods, psa


def compute_psa_with_uncertainty(acc_mean, acc_std, dt, periods=None,
                                 damp=0.05, n_real=50):
    """compute PSA with uncertainty propagation.

    propagates temporal uncertainty into PSA bounds by evaluating
    PSA for multiple realizations of the acceleration time history.

    Args:
        acc_mean (ndarray): mean acceleration
        acc_std  (ndarray): std of acceleration
        dt       (float):  time step [s]
        periods  (ndarray): oscillator periods [s]
        damp     (float):  damping ratio
        n_real   (int):    number of realizations

    Returns:
        periods  (ndarray): periods [s]
        psa_mean (ndarray): mean PSA
        psa_lo   (ndarray): lower 2-sigma bound
        psa_hi   (ndarray): upper 2-sigma bound
    """
    if periods is None:
        periods = PSA_PERIODS.copy()

    rng  = np.random.default_rng(42)
    psa_all = np.zeros((n_real, len(periods)))

    for i in range(n_real):
        noise   = rng.normal(0, 1, len(acc_mean))
        acc_r   = acc_mean + noise * acc_std
        _, psa_all[i] = compute_psa(acc_r, dt, periods, damp)

    psa_mean = np.mean(psa_all, axis=0)
    psa_std  = np.std(psa_all, axis=0)
    psa_lo   = psa_mean - 2 * psa_std   # 95% confidence
    psa_hi   = psa_mean + 2 * psa_std

    return periods, psa_mean, psa_lo, psa_hi


def compute_fas(data, dt):
    """compute fourier amplitude spectrum.

    Args:
        data (ndarray): time series
        dt   (float):   time step [s]

    Returns:
        freq (ndarray): frequency axis [Hz]
        fas  (ndarray): fourier amplitude spectrum
    """
    n    = len(data)
    spec = np.fft.rfft(data)
    fas  = np.abs(spec) * dt
    freq = np.fft.rfftfreq(n, d=dt)
    return freq, fas


def compute_fas_with_uncertainty(data_mean, data_std, dt, n_real=50):
    """compute FAS with uncertainty bounds.

    Args:
        data_mean (ndarray): mean time series
        data_std  (ndarray): std of time series
        dt        (float):   time step [s]
        n_real    (int):     number of realizations

    Returns:
        freq     (ndarray): frequency axis [Hz]
        fas_mean (ndarray): mean FAS
        fas_lo   (ndarray): lower 2-sigma bound
        fas_hi   (ndarray): upper 2-sigma bound
    """
    rng     = np.random.default_rng(42)
    fas_all = []

    for i in range(n_real):
        noise = rng.normal(0, 1, len(data_mean))
        data_r = data_mean + noise * data_std
        _, fas = compute_fas(data_r, dt)
        fas_all.append(fas)

    fas_all  = np.array(fas_all)
    fas_mean = np.mean(fas_all, axis=0)
    fas_std  = np.std(fas_all, axis=0)
    freq     = np.fft.rfftfreq(len(data_mean), d=dt)

    return freq, fas_mean, fas_mean - 2 * fas_std, fas_mean + 2 * fas_std


def compute_psd(data, dt):
    """compute power spectral density.

    Args:
        data (ndarray): time series
        dt   (float):   time step [s]

    Returns:
        freq (ndarray): frequency axis [Hz]
        psd  (ndarray): power spectral density
    """
    freq, fas = compute_fas(data, dt)
    n         = len(data)
    psd       = (fas ** 2) / (n * dt)
    return freq, psd


def compute_psd_with_uncertainty(data_mean, data_std, dt, n_real=50):
    """compute PSD with uncertainty bounds.

    Args:
        data_mean (ndarray): mean time series
        data_std  (ndarray): std of time series
        dt        (float):   time step [s]
        n_real    (int):     number of realizations

    Returns:
        freq     (ndarray): frequency axis [Hz]
        psd_mean (ndarray): mean PSD
        psd_lo   (ndarray): lower 2-sigma bound
        psd_hi   (ndarray): upper 2-sigma bound
    """
    rng     = np.random.default_rng(42)
    psd_all = []

    for i in range(n_real):
        noise  = rng.normal(0, 1, len(data_mean))
        data_r = data_mean + noise * data_std
        _, psd = compute_psd(data_r, dt)
        psd_all.append(psd)

    psd_all  = np.array(psd_all)
    psd_mean = np.mean(psd_all, axis=0)
    psd_std  = np.std(psd_all, axis=0)
    freq     = np.fft.rfftfreq(len(data_mean), d=dt)

    return freq, psd_mean, psd_mean - 2 * psd_std, psd_mean + 2 * psd_std
