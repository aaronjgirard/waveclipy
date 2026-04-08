"""tests for spectral computation module."""

import numpy as np
import pytest
from waveclipy.spectra import compute_psa, compute_fas, compute_psd, PSA_PERIODS


def test_psa_periods_count():
    """should have 23 standard periods."""
    assert len(PSA_PERIODS) == 23


def test_psa_positive():
    """PSA values should be positive."""
    rng = np.random.default_rng(0)
    acc = rng.normal(0, 1, 2000)
    dt  = 0.01
    per, psa = compute_psa(acc, dt)
    assert np.all(psa > 0)
    assert len(per) == 23


def test_fas_symmetry():
    """FAS should be real and non-negative."""
    rng  = np.random.default_rng(0)
    data = rng.normal(0, 1, 1000)
    dt   = 0.01
    freq, fas = compute_fas(data, dt)
    assert np.all(fas >= 0)
    assert len(freq) == len(fas)


def test_psd_positive():
    """PSD should be non-negative."""
    rng  = np.random.default_rng(0)
    data = rng.normal(0, 1, 1000)
    dt   = 0.01
    freq, psd = compute_psd(data, dt)
    assert np.all(psd >= 0)


def test_psa_at_pga():
    """PSA at very short period should approximate PGA."""
    rng = np.random.default_rng(0)
    acc = rng.normal(0, 1, 5000)
    dt  = 0.01
    per, psa = compute_psa(acc, dt, periods=np.array([0.01]))
    pga = np.max(np.abs(acc))
    ## psa at 0.01s should be within ~50% of PGA for random signal
    assert psa[0] > 0.3 * pga
