"""clipping detection: amplitude-based and gradient-based methods.

implements the detection algorithm described in the 'clipping detection'
section of Edwards et al. (2025).
"""

import numpy as np
from scipy.signal import find_peaks


def _amplitude_histogram(data):
    """compute amplitude histogram (derivative of CDF of sorted amplitudes).

    normalizes amplitudes to [-1, 1] and computes a smoothed derivative
    of the CDF. uses high bin resolution to resolve narrow clip spikes.

    Args:
        data (ndarray): raw waveform in integer counts

    Returns:
        x_bins (ndarray): normalized amplitude bin centers
        hist   (ndarray): amplitude histogram (scaled PDF)
    """
    peak = np.max(np.abs(data))
    if peak == 0:
        peak = 1.0
    norm  = data / peak                          # normalize to [-1, 1]
    n_bin = min(2000, max(500, int(2 * peak)))   # high resolution
    hist, edges = np.histogram(norm, bins=n_bin, range=(-1.05, 1.05))
    x_bin = 0.5 * (edges[:-1] + edges[1:])
    hist  = hist.astype(float)
    return x_bin, hist


def _detect_amplitude(data, min_iter=5):
    """amplitude-based clip detection.

    detects 2-3 peaks in the amplitude histogram; central peak = signal,
    edge peaks = clip levels. uses half-width-at-half-height to set
    conservative clip boundaries.

    Args:
        data     (ndarray): raw waveform in integer counts
        min_iter (int):     minimum number of prominence iterations

    Returns:
        clip_flag (ndarray): boolean, True where amplitude-clipped
        clip_lo   (float):  lower clip level (in original units)
        clip_hi   (float):  upper clip level (in original units)
    """
    peak_amp = np.max(np.abs(data))
    if peak_amp == 0:
        return np.zeros(len(data), dtype=bool), None, None

    x_bin, hist = _amplitude_histogram(data)
    hist_max    = np.max(hist)
    min_sep     = 0.6
    dx          = x_bin[1] - x_bin[0]
    min_dist    = int(min_sep / dx)

    ## iterate prominence threshold to find exactly 3 peaks
    prom    = 0.5 * hist_max
    found   = False
    n_iter  = 0
    peaks   = []

    ## phase 1: decrease prominence until 3 peaks found
    while n_iter < max(min_iter, 20):
        peaks, props = find_peaks(hist, prominence=prom, distance=min_dist)
        n_iter += 1
        if len(peaks) == 3:
            found = True
            break
        elif len(peaks) > 3:
            break
        prom *= 0.5

    ## phase 2: if too many peaks, increase prominence with 25% steps
    if not found and len(peaks) > 3:
        prom_lo = prom
        prom_hi = 0.5 * hist_max
        while not found and n_iter < 50:
            prom = prom_lo + 0.25 * (prom_hi - prom_lo)
            peaks, props = find_peaks(
                hist, prominence=prom, distance=min_dist
            )
            n_iter += 1
            if len(peaks) == 3:
                found = True
            elif len(peaks) > 3:
                prom_lo = prom
            else:
                prom_hi = prom

    if not found or len(peaks) != 3:
        return np.zeros(len(data), dtype=bool), None, None

    ## sort peaks by x-position; edges are clip levels
    pk_x   = x_bin[peaks]
    pk_idx = np.argsort(pk_x)
    lo_pk  = peaks[pk_idx[0]]
    mid_pk = peaks[pk_idx[1]]
    hi_pk  = peaks[pk_idx[2]]

    ## validate: at least one edge peak must be taller than central peak
    ## (clipping concentrates samples at clip level, creating tall spikes)
    lo_h = hist[lo_pk]
    hi_h = hist[hi_pk]
    md_h = hist[mid_pk]
    if lo_h <= md_h and hi_h <= md_h:
        return np.zeros(len(data), dtype=bool), None, None

    ## half-width-at-half-height for conservative boundaries
    lo_lvl = _hwhh_boundary(x_bin, hist, lo_pk, side="upper")
    hi_lvl = _hwhh_boundary(x_bin, hist, hi_pk, side="lower")

    clip_lo = lo_lvl * peak_amp
    clip_hi = hi_lvl * peak_amp

    clip_flag = (data <= clip_lo) | (data >= clip_hi)
    return clip_flag, clip_lo, clip_hi


def _hwhh_boundary(x_bin, hist, pk_idx, side="lower"):
    """compute half-width-at-half-height boundary.

    Args:
        x_bin  (ndarray): x axis
        hist   (ndarray): histogram values
        pk_idx (int):     peak index
        side   (str):     'lower' or 'upper' - which side toward zero

    Returns:
        float: x-value at the half-height boundary
    """
    half_h = hist[pk_idx] / 2.0
    if side == "lower":
        ## search leftward from peak
        idx = pk_idx
        while idx > 0 and hist[idx] > half_h:
            idx -= 1
        return x_bin[idx]
    else:
        ## search rightward from peak
        idx = pk_idx
        while idx < len(hist) - 1 and hist[idx] > half_h:
            idx += 1
        return x_bin[idx]


def _peak_width(hist, pk_idx):
    """measure width of a peak at half its height.

    Args:
        hist   (ndarray): histogram values
        pk_idx (int):     peak index

    Returns:
        int: width in bins
    """
    half_h = hist[pk_idx] / 2.0
    lo     = pk_idx
    while lo > 0 and hist[lo] > half_h:
        lo -= 1
    hi = pk_idx
    while hi < len(hist) - 1 and hist[hi] > half_h:
        hi += 1
    return hi - lo


def _detect_gradient(data):
    """gradient-based clip detection.

    flags samples where gradient < 1% of quantization resolution.
    requires at least 2 consecutive near-zero-gradient samples to
    distinguish clipping from natural turning points. removes flags
    where amplitude < 5% of peak AND range across +/-10 samples
    < 50 counts (low-amplitude quantization, not clipping).

    Args:
        data (ndarray): raw waveform in integer counts

    Returns:
        clip_flag (ndarray): boolean, True where gradient-clipped
    """
    dflt = data.astype(float)
    grad = np.diff(dflt)
    grad = np.append(grad, 0)

    ## quantization resolution = 1 count for integer data
    quant     = 1.0
    threshold = 0.01 * quant
    flat_flag = np.abs(grad) < threshold

    ## require at least 2 consecutive flat samples (isolated zeros are turning points)
    n         = len(data)
    clip_flag = np.zeros(n, dtype=bool)
    for i in range(n):
        if not flat_flag[i]:
            continue
        has_nbr = False
        if i > 0 and flat_flag[i - 1]:
            has_nbr = True
        if i < n - 1 and flat_flag[i + 1]:
            has_nbr = True
        if has_nbr:
            clip_flag[i] = True

    ## remove flags at low amplitude / low range (quantization, not clipping)
    peak_amp = np.max(np.abs(data))
    for i in range(n):
        if clip_flag[i]:
            lo      = max(0, i - 10)
            hi      = min(n, i + 11)
            amp_low = np.abs(data[i]) < 0.05 * peak_amp
            rng_low = (np.max(dflt[lo:hi]) - np.min(dflt[lo:hi])) < 50
            if amp_low and rng_low:
                clip_flag[i] = False

    return clip_flag


def detect_clipping(data, conservative_pct=0.15):
    """detect clipped samples using amplitude + gradient methods.

    the final clip flag is the union of both methods. a conservative
    threshold also flags samples within conservative_pct of the
    detected clip level.

    should be applied to raw integer digitizer counts only.

    Args:
        data             (ndarray): raw waveform in integer counts
        conservative_pct (float):   fraction of clip level for
                                    conservative threshold (default 0.15)

    Returns:
        clip_flag (ndarray): boolean array, True = clipped/potentially clipped
        clip_lo   (float):  detected lower clip level (or None)
        clip_hi   (float):  detected upper clip level (or None)
    """
    data = np.asarray(data, dtype=float)

    amp_flag, clip_lo, clip_hi = _detect_amplitude(data)
    grad_flag                  = _detect_gradient(data)

    ## union of both methods
    clip_flag = amp_flag | grad_flag

    ## conservative threshold: flag within conservative_pct of clip level
    if clip_lo is not None and clip_hi is not None:
        margin    = conservative_pct * max(abs(clip_lo), abs(clip_hi))
        cons_flag = (data <= clip_lo + margin) & (data < 0)
        cons_flag |= (data >= clip_hi - margin) & (data > 0)
        clip_flag |= cons_flag

    return clip_flag, clip_lo, clip_hi
