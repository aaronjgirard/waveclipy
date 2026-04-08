"""benchmark module: artificial clipping and misfit metrics.

implements benchmark testing from Edwards et al. (2025), equations 9-10.
"""

import numpy as np
from waveclipy.spectra import compute_psa, PSA_PERIODS


def artificial_clip(data, clip_pct):
    """artificially clip waveform at a percentage of peak amplitude.

    Args:
        data     (ndarray): original waveform
        clip_pct (float):   clip level as fraction of peak (e.g. 0.3 = 30%)

    Returns:
        clipped  (ndarray): clipped waveform
        clip_lvl (float):   absolute clip level used
    """
    data     = np.asarray(data, dtype=float)
    peak     = np.max(np.abs(data))
    clip_lvl = clip_pct * peak

    clipped = data.copy()
    clipped[clipped > clip_lvl]  = clip_lvl
    clipped[clipped < -clip_lvl] = -clip_lvl

    return clipped, clip_lvl


def time_domain_misfit(orig, recon, clip_flag):
    """compute time-domain misfit on clipped samples only (eq. 9).

    M(t) = (1 / sum(c(t))) * sum(c(t) * |y'(t) - x(t)|)

    Args:
        orig      (ndarray): original unclipped waveform y'(t)
        recon     (ndarray): reconstructed waveform x(t)
        clip_flag (ndarray): boolean clip flags c(t)

    Returns:
        float: time-domain misfit
    """
    clip_flag = np.asarray(clip_flag, dtype=bool)
    n_clip    = np.sum(clip_flag)
    if n_clip == 0:
        return 0.0
    return np.sum(clip_flag * np.abs(orig - recon)) / n_clip


def spectral_misfit(orig, recon, dt, periods=None):
    """compute spectral misfit per period (eq. 10).

    M_PSA(T) = log10(Y'(T)) - log10(Y(T))

    Args:
        orig    (ndarray): original unclipped waveform
        recon   (ndarray): reconstructed waveform
        dt      (float):   time step [s]
        periods (ndarray): oscillator periods [s]

    Returns:
        periods    (ndarray): periods used
        m_psa_t    (ndarray): per-period misfit
        m_psa_mean (float):   mean misfit over periods
        m_psa_abs  (float):   mean of absolute misfit over periods
    """
    if periods is None:
        periods = PSA_PERIODS.copy()

    ## compute PSA for original (differentiated to acceleration)
    acc_orig  = np.diff(orig) / dt
    acc_recon = np.diff(recon) / dt

    _, psa_orig  = compute_psa(acc_orig, dt, periods)
    _, psa_recon = compute_psa(acc_recon, dt, periods)

    ## avoid log of zero
    psa_orig  = np.maximum(psa_orig, 1e-30)
    psa_recon = np.maximum(psa_recon, 1e-30)

    m_psa_t    = np.log10(psa_recon) - np.log10(psa_orig)
    m_psa_mean = np.mean(m_psa_t)
    m_psa_abs  = np.mean(np.abs(m_psa_t))

    return periods, m_psa_t, m_psa_mean, m_psa_abs


def compute_misfit(orig, recon, clip_flag, dt, periods=None):
    """compute all misfit metrics.

    Args:
        orig      (ndarray): original unclipped waveform
        recon     (ndarray): reconstructed waveform
        clip_flag (ndarray): boolean clip flags
        dt        (float):   time step [s]
        periods   (ndarray): oscillator periods [s]

    Returns:
        dict with keys:
            m_time      : time-domain misfit
            periods     : periods array
            m_psa_t     : per-period spectral misfit
            m_psa_mean  : mean spectral misfit
            m_psa_abs   : mean absolute spectral misfit
    """
    m_t = time_domain_misfit(orig, recon, clip_flag)
    p, m_psa_t, m_mean, m_abs = spectral_misfit(orig, recon, dt, periods)

    return {
        "m_time"     : m_t,
        "periods"    : p,
        "m_psa_t"    : m_psa_t,
        "m_psa_mean" : m_mean,
        "m_psa_abs"  : m_abs,
    }


def run_benchmark(orig, dt, fc=None, mw=None, cl_values=None,
                  clip_levels=None, os=0.025, n_alpha=100, fs=None):
    """run full benchmark: clip at multiple levels, reconstruct, measure misfit.

    Args:
        orig       (ndarray): original velocity waveform
        dt         (float):   time step [s]
        fc         (float):   corner frequency [Hz]
        mw         (float):   moment magnitude
        cl_values  (list):    cl parameter values to test
        clip_levels(list):    clip levels as fractions of PGV
        os         (float):   secondary offset
        n_alpha    (int):     number of alpha iterations
        fs         (float):   sampling rate [Hz]

    Returns:
        dict: results keyed by (cl, clip_level)
    """
    from waveclipy.reconstruction import cwrdt

    if cl_values is None:
        cl_values = [5, 7, 10, 20, 1000]
    if clip_levels is None:
        clip_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    if fs is None:
        fs = 1.0 / dt

    results = {}
    for cl in cl_values:
        for lvl in clip_levels:
            clpd, clvl = artificial_clip(orig, lvl)
            cflag      = (np.abs(clpd) >= clvl * 0.999)

            x_mean, x_std = cwrdt(
                clpd, cflag, fc=fc, mw=mw, cl=cl,
                os=os, n_alpha=n_alpha, fs=fs,
            )
            mfit = compute_misfit(orig, x_mean, cflag, dt)
            results[(cl, lvl)] = {
                "misfit"  : mfit,
                "x_mean"  : x_mean,
                "x_std"   : x_std,
                "clipped" : clpd,
            }

    return results
