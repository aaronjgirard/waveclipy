"""synthetic clip-and-restore demo for waveclipy.

generates a test waveform (no external data required), artificially
clips it, reconstructs via CWRdt, and prints summary metrics.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from waveclipy.detection import detect_clipping
from waveclipy.reconstruction import cwrdt
from waveclipy.benchmark import artificial_clip, compute_misfit
from waveclipy.spectra import (
    compute_psa, compute_psa_with_uncertainty,
    compute_fas, compute_fas_with_uncertainty,
)
from waveclipy.utils import corner_frequency, husid_plot
from waveclipy.visualization import plot_waveforms, plot_psa, plot_husid


def make_synthetic_waveform(n=4000, fs=100.0, seed=42):
    """generate a synthetic earthquake-like velocity waveform.

    uses a sum of modulated sinusoids with an envelope that mimics
    a seismic signal (ramp up, sustain, decay).

    Args:
        n    (int):   number of samples
        fs   (float): sampling rate [Hz]
        seed (int):   random seed

    Returns:
        t    (ndarray): time axis [s]
        vel  (ndarray): velocity waveform (in 'counts')
    """
    rng = np.random.default_rng(seed)
    t   = np.arange(n) / fs
    dur = n / fs

    ## envelope: gaussian-like with asymmetric ramp
    t_pk = dur * 0.3
    env  = np.exp(-((t - t_pk) / (dur * 0.15)) ** 2)
    env += 0.3 * np.exp(-((t - dur * 0.5) / (dur * 0.2)) ** 2)

    ## multi-frequency content
    sig  = np.zeros(n)
    for f0 in [1.2, 2.5, 3.8, 5.1, 7.3]:
        phi  = rng.uniform(0, 2 * np.pi)
        amp  = rng.uniform(0.3, 1.0)
        sig += amp * np.sin(2 * np.pi * f0 * t + phi)

    ## add some band-limited noise
    sig += 0.2 * rng.normal(0, 1, n)

    vel = sig * env
    vel = vel / np.max(np.abs(vel)) * 2000  # scale to ~12-bit range
    vel = np.round(vel).astype(float)

    return t, vel


def main():
    fs  = 100.0
    dt  = 1.0 / fs
    mw  = 5.5
    fc  = corner_frequency(mw)

    print(f"corner frequency for Mw {mw}: {fc:.3f} Hz")
    print(f"segment duration (cl=10): {10.0/fc:.2f} s")

    ## generate synthetic waveform
    t, vel = make_synthetic_waveform(n=4000, fs=fs)

    ## artificially clip at 30% of peak
    clip_pct = 0.3
    clpd, clvl = artificial_clip(vel, clip_pct)
    cflag = np.abs(clpd) >= clvl * 0.999
    n_clip = np.sum(cflag)
    print(f"\nclip level: {clip_pct*100:.0f}% of peak ({clvl:.0f} counts)")
    print(f"clipped samples: {n_clip} / {len(vel)} ({100*n_clip/len(vel):.1f}%)")

    ## also test automatic detection on the clipped signal
    det_flag, det_lo, det_hi = detect_clipping(clpd.astype(int))
    print(f"auto-detected clip flags: {np.sum(det_flag)}")

    ## reconstruct
    print("\nrunning CWRdt reconstruction...")
    x_mean, x_std = cwrdt(
        clpd, cflag, fc=fc, cl=10.0,
        op=0.5, os=0.05, n_alpha=80, fs=fs,
        freq_constraint=False,
    )

    ## compute misfit
    mfit = compute_misfit(vel, x_mean, cflag, dt)
    print(f"\ntime-domain misfit (clipped samples): {mfit['m_time']:.2f} counts")
    print(f"mean spectral misfit M_PSA: {mfit['m_psa_mean']:.4f}")
    print(f"mean |M_PSA|: {mfit['m_psa_abs']:.4f}")

    ## error reduction
    err_clip  = np.mean(np.abs(vel[cflag] - clpd[cflag]))
    err_recon = np.mean(np.abs(vel[cflag] - x_mean[cflag]))
    print(f"\nmean error on clipped samples:")
    print(f"  clipped:       {err_clip:.1f} counts")
    print(f"  reconstructed: {err_recon:.1f} counts")
    print(f"  reduction:     {100*(1-err_recon/err_clip):.1f}%")

    ## PSA comparison
    acc_orig = np.diff(vel) / dt
    acc_clip = np.diff(clpd) / dt
    acc_rec  = np.diff(x_mean) / dt
    acc_std  = np.diff(x_std) / dt

    per, psa_orig = compute_psa(acc_orig, dt)
    _,   psa_clip = compute_psa(acc_clip, dt)
    per, psa_rec, psa_lo, psa_hi = compute_psa_with_uncertainty(
        acc_rec, acc_std, dt, n_real=30,
    )

    ## husid plot
    t_hc, hc = husid_plot(acc_clip, dt)
    t_hr, hr = husid_plot(acc_rec, dt)

    ## FAS
    freq_o, fas_o = compute_fas(vel, dt)
    freq_r, fas_r = compute_fas(x_mean, dt)

    ## plots
    fig1, _ = plot_waveforms(
        t, vel, clpd, x_mean, std=x_std,
        title=f"CWRdt reconstruction ({clip_pct*100:.0f}% clip)",
        save="waveclipy_waveform.png",
    )
    fig2, _ = plot_psa(
        per, psa_orig, psa_clip, psa_rec, psa_lo, psa_hi,
        title="PSA comparison", save="waveclipy_psa.png",
    )
    fig3, _ = plot_husid(
        t_hc, hc, t_hr, hr,
        title="husid plot", save="waveclipy_husid.png",
    )

    print("\nplots saved: waveclipy_waveform.png, waveclipy_psa.png, waveclipy_husid.png")
    print("done.")


if __name__ == "__main__":
    main()
