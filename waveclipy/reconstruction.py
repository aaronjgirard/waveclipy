"""core CWRdt nonstationary clipped waveform restoration.

implements the POCS-based reconstruction from Edwards et al. (2025),
equations 1-8.
"""

import numpy as np
from waveclipy.utils import (
    corner_frequency,
    segment_duration,
    butterworth_lowpass_freq,
    fix_wraparound_clipping,
)


def _reconstruct_segment(seg, clip_seg, n_alpha=100, fs=None,
                         freq_constraint=True, tol=0.05):
    """reconstruct a single clipped segment via POCS iteration.

    equations 5 and 6 of Edwards et al. (2025).

    for each iteration n (alpha decreasing from 1 to 0):
        - compute FFT of current segment
        - zero spectral components below alpha * max|X(f)|
        - inverse FFT to get trial reconstruction
        - retain maxima: keep trial where |trial| > |current|

    Args:
        seg              (ndarray): clipped segment waveform
        clip_seg         (ndarray): boolean clip flags for segment
        n_alpha          (int):     number of alpha iterations
        fs               (float):   sampling rate [Hz]
        freq_constraint  (bool):    apply derivative constraint
        tol              (float):   tolerance for derivative constraint (0.05 = 5%)

    Returns:
        ndarray: reconstructed segment
    """
    m_tau = len(seg)
    if m_tau == 0:
        return seg.copy()

    xps = seg.copy().astype(float)

    ## peak of first time-derivative of clipped waveform (for constraint)
    if freq_constraint and fs is not None:
        drv_clip = np.max(np.abs(np.diff(xps))) * fs
    else:
        drv_clip = None

    nyq    = fs / 2.0 if fs is not None else None
    alphas = np.linspace(1.0, 0.0, n_alpha + 1)[:-1]  # exclude alpha=0

    for n, alpha in enumerate(alphas):
        spec    = np.fft.fft(xps)
        pk_spec = np.max(np.abs(spec))
        if pk_spec == 0:
            break

        ## eq 5: zero components below threshold
        thresh = alpha * pk_spec
        mask   = np.abs(spec) > thresh
        spec_n = np.where(mask, spec, 0.0 + 0j)

        ## optional frequency constraint (eq 6 note)
        if freq_constraint and drv_clip is not None and nyq is not None:
            trial_t = np.real(np.fft.ifft(spec_n))
            drv_new = np.max(np.abs(np.diff(trial_t))) * fs
            if drv_new > drv_clip * (1.0 + tol):
                ## iteratively reduce low-pass cutoff
                freqs  = np.fft.fftfreq(m_tau, d=1.0 / fs)
                f_lp   = nyq * 0.95
                f_min  = np.sqrt(nyq)
                while f_lp > f_min:
                    spec_f  = butterworth_lowpass_freq(spec_n, freqs, f_lp)
                    trial_t = np.real(np.fft.ifft(spec_f))
                    drv_new = np.max(np.abs(np.diff(trial_t))) * fs
                    if drv_new <= drv_clip * (1.0 + tol):
                        spec_n = spec_f
                        break
                    f_lp *= 0.9
                else:
                    spec_n = butterworth_lowpass_freq(spec_n, freqs, f_min)

        ## inverse FFT
        xpsn = np.real(np.fft.ifft(spec_n))

        ## eq 6: iteratively retain maxima
        update     = np.abs(xpsn) > np.abs(xps)
        update     = update & clip_seg       # only update clipped samples
        xps[update] = xpsn[update]

    return xps


def cwrdt(data, clip_flag, fc=None, mw=None, cl=10.0, op=0.5, os=0.025,
          n_alpha=100, fs=100.0, freq_constraint=True, tol=0.05):
    """nonstationary clipped waveform restoration (CWRdt).

    equations 1-8 of Edwards et al. (2025).

    Args:
        data             (ndarray): clipped waveform (integer counts)
        clip_flag        (ndarray): boolean, True = clipped sample
        fc               (float):  corner frequency [Hz] (or None to estimate)
        mw               (float):  moment magnitude (used if fc is None)
        cl               (float):  segment length constant (default 10)
        op               (float):  primary overlap fraction (default 0.5)
        os               (float):  secondary offset fraction (default 0.025)
        n_alpha          (int):    number of alpha iterations
        fs               (float):  sampling rate [Hz]
        freq_constraint  (bool):   apply derivative constraint
        tol              (float):  derivative constraint tolerance

    Returns:
        x_mean (ndarray): reconstructed waveform (best estimate)
        x_std  (ndarray): temporal uncertainty (standard deviation)
    """
    data      = np.asarray(data, dtype=float)
    clip_flag = np.asarray(clip_flag, dtype=bool)
    n_samp    = len(data)

    ## estimate corner frequency if not provided (eq 2)
    if fc is None:
        if mw is None:
            mw = 5.0
        fc = corner_frequency(mw)

    ## segment duration in samples (eq 1)
    dt_p     = segment_duration(fc, cl)
    seg_len  = max(int(dt_p * fs), 64)
    seg_len  = min(seg_len, n_samp)

    ## number of secondary subsegments (eq 4 constraint: (s-1)*os <= op)
    n_s = int(np.floor(op / os)) + 1
    n_s = max(n_s, 1)

    ## handle wrap-around clipping
    data = fix_wraparound_clipping(data, clip_flag)

    ## collect all S full-length reconstructions
    all_recon = np.zeros((n_s, n_samp))

    for s in range(n_s):
        recon = data.copy()

        ## compute start times for primary segments (eq 4)
        s_offset = s * os
        stride   = (op + s_offset) * seg_len
        stride   = max(int(stride), 1)

        ## determine number of primary segments
        if stride >= seg_len:
            stride = max(int(op * seg_len), 1)

        starts = []
        pos    = 0
        while pos < n_samp:
            starts.append(pos)
            pos += stride

        ## reconstruct each primary segment
        for p, st in enumerate(starts):
            en      = min(st + seg_len, n_samp)
            seg     = recon[st:en].copy()
            c_seg   = clip_flag[st:en]

            if not np.any(c_seg):
                continue

            rec_seg     = _reconstruct_segment(
                seg, c_seg, n_alpha=n_alpha, fs=fs,
                freq_constraint=freq_constraint, tol=tol
            )
            recon[st:en] = rec_seg

        all_recon[s] = recon

    ## eq 7: mean over S realizations
    x_mean = np.mean(all_recon, axis=0)

    ## eq 8: sample standard deviation
    if n_s > 1:
        x_var = np.sum((all_recon - x_mean[np.newaxis, :]) ** 2, axis=0)
        x_var /= (n_s - 1)
        x_std = np.sqrt(x_var)
    else:
        x_std = np.zeros(n_samp)

    ## preserve unclipped samples exactly
    x_mean[~clip_flag] = data[~clip_flag]
    x_std[~clip_flag]  = 0.0

    return x_mean, x_std
