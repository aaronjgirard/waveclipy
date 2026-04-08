"""real-data demo: clip and restore an actual seismic waveform.

uses a 10-minute BHZ record from station IU.ANTO (Ankara, Turkey)
during the 2023-02-06 Mw 7.8 Turkey earthquake, fetched from IRIS.
the record is artificially clipped at 30% of peak amplitude, then
reconstructed with CWRdt and compared.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")

from waveclipy.benchmark import artificial_clip, compute_misfit
from waveclipy.reconstruction import cwrdt
from waveclipy.spectra import compute_psa, compute_psa_with_uncertainty
from waveclipy.utils import corner_frequency, husid_plot
from waveclipy.visualization import (
    plot_waveforms, plot_psa, plot_husid, plot_fas,
)
from waveclipy.spectra import compute_fas


def load_iris_ascii(path):
    """load IRIS TSPAIR ascii waveform file.

    Args:
        path (str): path to ascii file

    Returns:
        data (ndarray): waveform in integer counts
        fs   (float):   sampling rate [Hz]
        meta (str):     header line
    """
    with open(path, "r") as f:
        hdr  = f.readline().strip()
        vals = []
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                vals.append(int(parts[1]))
    data = np.array(vals, dtype=float)

    ## parse sample rate from header
    fs = 40.0
    for tok in hdr.split(","):
        tok = tok.strip()
        if "sps" in tok:
            fs = float(tok.replace("sps", "").strip())
            break

    return data, fs, hdr


def main():
    ## load real waveform
    data, fs, hdr = load_iris_ascii("waveform_turkey_anto.ascii")
    dt   = 1.0 / fs
    n    = len(data)
    time = np.arange(n) * dt

    print(f"station: IU.ANTO.00.BHZ")
    print(f"event:   2023-02-06 Mw 7.8 Turkey earthquake")
    print(f"samples: {n}, fs: {fs} Hz, duration: {n*dt:.1f} s")
    print(f"peak amplitude: {np.max(np.abs(data)):.0f} counts")

    ## earthquake parameters
    mw = 7.8
    fc = corner_frequency(mw)
    print(f"\ncorner frequency (Mw {mw}): {fc:.4f} Hz")
    print(f"segment duration (cl=10): {10.0/fc:.1f} s")

    ## artificially clip at 30% of peak
    clip_pct    = 0.30
    clpd, clvl  = artificial_clip(data, clip_pct)
    cflag       = np.abs(clpd) >= clvl * 0.999
    n_clip      = np.sum(cflag)
    print(f"\nclip level: {clip_pct*100:.0f}% of peak ({clvl:.0f} counts)")
    print(f"clipped samples: {n_clip} / {n} ({100*n_clip/n:.1f}%)")

    ## reconstruct
    print("\nrunning CWRdt reconstruction...")
    x_mean, x_std = cwrdt(
        clpd, cflag, fc=fc, cl=10.0,
        op=0.5, os=0.05, n_alpha=80, fs=fs,
        freq_constraint=False,
    )

    ## misfit
    mfit = compute_misfit(data, x_mean, cflag, dt)
    print(f"\ntime-domain misfit (clipped samples): {mfit['m_time']:.1f} counts")
    print(f"mean spectral misfit M_PSA: {mfit['m_psa_mean']:.4f}")
    print(f"mean |M_PSA|: {mfit['m_psa_abs']:.4f}")

    err_clip  = np.mean(np.abs(data[cflag] - clpd[cflag]))
    err_recon = np.mean(np.abs(data[cflag] - x_mean[cflag]))
    print(f"\nmean error on clipped samples:")
    print(f"  clipped:       {err_clip:.1f} counts")
    print(f"  reconstructed: {err_recon:.1f} counts")
    print(f"  reduction:     {100*(1 - err_recon/err_clip):.1f}%")

    ## PSA
    acc_orig = np.diff(data) / dt
    acc_clip = np.diff(clpd) / dt
    acc_rec  = np.diff(x_mean) / dt
    acc_std  = np.diff(x_std) / dt

    per, psa_orig = compute_psa(acc_orig, dt)
    _,   psa_clip = compute_psa(acc_clip, dt)
    per, psa_rec, psa_lo, psa_hi = compute_psa_with_uncertainty(
        acc_rec, acc_std, dt, n_real=30,
    )

    ## husid
    t_hc, hc = husid_plot(acc_clip, dt)
    t_hr, hr = husid_plot(acc_rec, dt)

    ## FAS
    freq_o, fas_o = compute_fas(data, dt)
    freq_r, fas_r = compute_fas(x_mean, dt)

    ## plots
    plot_waveforms(
        time, data, clpd, x_mean, std=x_std,
        title="CWRdt: IU.ANTO Mw 7.8 Turkey (30% clip)",
        save="real_data_waveform.png",
    )
    plot_psa(
        per, psa_orig, psa_clip, psa_rec, psa_lo, psa_hi,
        title="PSA: IU.ANTO Mw 7.8 Turkey",
        save="real_data_psa.png",
    )
    plot_husid(
        t_hc, hc, t_hr, hr,
        title="husid: IU.ANTO Mw 7.8 Turkey",
        save="real_data_husid.png",
    )
    plot_fas(
        freq_o, fas_o, fas_r,
        title="FAS: IU.ANTO Mw 7.8 Turkey",
        save="real_data_fas.png",
    )

    print("\nplots saved: real_data_waveform.png, real_data_psa.png,")
    print("             real_data_husid.png, real_data_fas.png")
    print("done.")


if __name__ == "__main__":
    main()
