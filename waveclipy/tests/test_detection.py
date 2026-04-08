"""tests for clipping detection module."""

import numpy as np
import pytest
from waveclipy.detection import detect_clipping, _detect_gradient


def _make_signal(n=5000, fs=100.0, clip_pct=0.3):
    """create a synthetic clipped signal in integer counts."""
    t    = np.arange(n) / fs
    sig  = np.sin(2 * np.pi * 1.5 * t) + 0.5 * np.sin(2 * np.pi * 3.7 * t)
    sig  = sig / np.max(np.abs(sig))
    cnts = np.round(sig * 2000).astype(int)   # simulate 12-bit range
    peak = np.max(np.abs(cnts))
    clvl = int(clip_pct * peak)
    clip = cnts.copy()
    clip[clip > clvl]  = clvl
    clip[clip < -clvl] = -clvl
    return cnts, clip, clvl


def test_detect_clipping_flags_clipped_samples():
    """most clipped samples should be flagged."""
    orig, clip, clvl = _make_signal(clip_pct=0.3)
    flag, lo, hi = detect_clipping(clip)
    true_clip = (np.abs(clip) >= clvl * 0.999)
    ## at least 90% of truly clipped samples should be flagged
    recall = np.sum(flag[true_clip]) / np.sum(true_clip)
    assert recall > 0.9


def test_unclipped_signal_not_flagged():
    """an unclipped signal should have no (or very few) flags."""
    t   = np.arange(5000) / 100.0
    sig = np.round(np.sin(2 * np.pi * 2 * t) * 500).astype(int)
    flag, lo, hi = detect_clipping(sig)
    ## allow some false positives from gradient method at quantization steps
    assert np.sum(flag) < 0.05 * len(sig)


def test_gradient_detection_basic():
    """gradient detection should flag flat (zero-gradient) regions."""
    data = np.array([0, 1, 2, 3, 100, 100, 100, 100, 3, 2, 1, 0])
    flag = _detect_gradient(data)
    ## the flat region at 100 should be flagged
    assert flag[5] or flag[6]


def test_detect_returns_correct_shape():
    """output flag should match input length."""
    _, clip, _ = _make_signal()
    flag, _, _ = detect_clipping(clip)
    assert len(flag) == len(clip)
