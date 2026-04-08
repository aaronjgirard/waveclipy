"""plotting: waveforms, PSA, husid, uncertainty bands."""

import numpy as np
import matplotlib.pyplot as plt


def plot_waveforms(time, original, clipped, reconstructed, std=None,
                   title="cwrdt reconstruction", save=None):
    """plot original, clipped, and reconstructed waveforms.

    Args:
        time          (ndarray): time axis [s]
        original      (ndarray): original waveform (or None)
        clipped       (ndarray): clipped waveform
        reconstructed (ndarray): reconstructed waveform
        std           (ndarray): uncertainty (or None)
        title         (str):     plot title
        save          (str):     path to save figure (or None)
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    if original is not None:
        ax.plot(time, original, "0.7", lw=0.8, label="original")
    ax.plot(time, clipped, "k", lw=0.5, label="clipped")
    ax.plot(time, reconstructed, "r", lw=0.5, label="reconstructed")
    if std is not None:
        ax.fill_between(
            time,
            reconstructed - 2 * std,
            reconstructed + 2 * std,
            color="r", alpha=0.15, label="95% conf.",
        )
    ax.set_xlabel("time [s]")
    ax.set_ylabel("amplitude")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150)
    return fig, ax


def plot_psa(periods, psa_orig=None, psa_clip=None, psa_recon=None,
             psa_lo=None, psa_hi=None, title="psa comparison", save=None):
    """plot pseudo-spectral acceleration comparison.

    Args:
        periods   (ndarray): oscillator periods [s]
        psa_orig  (ndarray): PSA of original (or None)
        psa_clip  (ndarray): PSA of clipped (or None)
        psa_recon (ndarray): PSA of reconstructed (or None)
        psa_lo    (ndarray): lower uncertainty bound (or None)
        psa_hi    (ndarray): upper uncertainty bound (or None)
        title     (str):     plot title
        save      (str):     path to save figure (or None)
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    if psa_orig is not None:
        ax.loglog(periods, psa_orig, "0.5", lw=1.5, label="original")
    if psa_clip is not None:
        ax.loglog(periods, psa_clip, "k--", lw=1, label="clipped")
    if psa_recon is not None:
        ax.loglog(periods, psa_recon, "r", lw=1, label="reconstructed")
    if psa_lo is not None and psa_hi is not None:
        ax.fill_between(
            periods,
            np.maximum(psa_lo, 1e-30),
            psa_hi,
            color="r", alpha=0.15, label="95% conf.",
        )
    ax.set_xlabel("period [s]")
    ax.set_ylabel("PSA")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150)
    return fig, ax


def plot_husid(time_clip, hsd_clip, time_recon, hsd_recon,
               title="husid plot", save=None):
    """plot husid (cumulative arias intensity) comparison.

    Args:
        time_clip  (ndarray): time axis for clipped
        hsd_clip   (ndarray): husid of clipped
        time_recon (ndarray): time axis for reconstructed
        hsd_recon  (ndarray): husid of reconstructed
        title      (str):     plot title
        save       (str):     path to save figure (or None)
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(time_clip, hsd_clip, "k", lw=1, label="clipped")
    ax.plot(time_recon, hsd_recon, "r", lw=1, label="reconstructed")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("normalized arias intensity")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150)
    return fig, ax


def plot_fas(freq, fas_orig=None, fas_recon=None, fas_lo=None, fas_hi=None,
             title="fourier amplitude spectrum", save=None):
    """plot FAS comparison with optional uncertainty.

    Args:
        freq      (ndarray): frequency axis [Hz]
        fas_orig  (ndarray): FAS of original (or None)
        fas_recon (ndarray): FAS of reconstructed (or None)
        fas_lo    (ndarray): lower bound (or None)
        fas_hi    (ndarray): upper bound (or None)
        title     (str):     plot title
        save      (str):     path to save figure (or None)
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    mask = freq > 0
    if fas_orig is not None:
        ax.loglog(freq[mask], fas_orig[mask], "0.5", lw=1, label="original")
    if fas_recon is not None:
        ax.loglog(freq[mask], fas_recon[mask], "r", lw=1, label="reconstructed")
    if fas_lo is not None and fas_hi is not None:
        ax.fill_between(
            freq[mask],
            np.maximum(fas_lo[mask], 1e-30),
            fas_hi[mask],
            color="r", alpha=0.15, label="95% conf.",
        )
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("FAS")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150)
    return fig, ax
