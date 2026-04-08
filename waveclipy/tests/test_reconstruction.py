"""tests for CWRdt reconstruction module."""

import numpy as np
import pytest
from waveclipy.reconstruction import cwrdt


def _make_clipped_signal(n=2000, fs=100.0, clip_pct=0.5):
    """create a synthetic clipped signal."""
    t    = np.arange(n) / fs
    sig  = np.sin(2 * np.pi * 2.0 * t) + 0.3 * np.sin(2 * np.pi * 5.0 * t)
    sig  = sig / np.max(np.abs(sig)) * 1000
    peak = np.max(np.abs(sig))
    clvl = clip_pct * peak

    clip = sig.copy()
    clip[clip > clvl]  = clvl
    clip[clip < -clvl] = -clvl
    cflag = np.abs(clip) >= clvl * 0.999

    return sig, clip, cflag, fs


def test_reconstruction_reduces_error():
    """reconstructed signal should be closer to original than clipped."""
    orig, clip, cflag, fs = _make_clipped_signal(clip_pct=0.5)

    x_mean, x_std = cwrdt(
        clip, cflag, fc=2.0, cl=10.0,
        n_alpha=50, fs=fs, freq_constraint=False,
    )

    err_clip  = np.mean(np.abs(orig[cflag] - clip[cflag]))
    err_recon = np.mean(np.abs(orig[cflag] - x_mean[cflag]))
    assert err_recon < err_clip


def test_unclipped_samples_preserved():
    """unclipped samples should remain unchanged."""
    orig, clip, cflag, fs = _make_clipped_signal(clip_pct=0.5)

    x_mean, x_std = cwrdt(
        clip, cflag, fc=2.0, cl=10.0,
        n_alpha=50, fs=fs, freq_constraint=False,
    )

    np.testing.assert_array_almost_equal(
        x_mean[~cflag], clip[~cflag], decimal=10
    )


def test_output_shapes():
    """output should match input length."""
    _, clip, cflag, fs = _make_clipped_signal()
    x_mean, x_std = cwrdt(
        clip, cflag, fc=2.0, cl=10.0,
        n_alpha=20, fs=fs, freq_constraint=False,
    )
    assert len(x_mean) == len(clip)
    assert len(x_std) == len(clip)


def test_uncertainty_nonzero_at_clipped():
    """uncertainty should be nonzero at clipped samples (with multiple subsegments)."""
    _, clip, cflag, fs = _make_clipped_signal(clip_pct=0.4)
    x_mean, x_std = cwrdt(
        clip, cflag, fc=2.0, cl=10.0,
        os=0.1, n_alpha=30, fs=fs, freq_constraint=False,
    )
    ## some clipped samples should have nonzero uncertainty
    assert np.any(x_std[cflag] > 0)
