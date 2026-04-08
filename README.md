# waveclipy

Nonstationary clipped waveform restoration ($\text{CWR}_{\Delta t}$) for seismic time-history records.

Implements the algorithm from:

> Edwards, B., Ntinalexis, M., Vanneste, K., & van Elk, J. (2025).
> Reconstruction of Clipped Time-History Records, with Application to Ground Motions
> from the 1992 $M_L$ 5.8 Roermond, Netherlands, Earthquake.
> *Seismological Research Letters*, doi: 10.1785/0220250139.

Development assisted by ClaudeCode

## Installation

```bash
pip install -e .
```

Dependencies: `numpy`, `scipy`, `matplotlib`.

## Usage

```python
import numpy as np
from waveclipy import detect_clipping, cwrdt
from waveclipy.spectra import compute_psa_with_uncertainty

## load your raw integer-count waveform
data = np.loadtxt("my_waveform.txt")
fs   = 100.0  # sampling rate [Hz]

## detect clipping
clip_flag, clip_lo, clip_hi = detect_clipping(data)

## reconstruct
x_mean, x_std = cwrdt(data, clip_flag, mw=5.5, cl=10.0, fs=fs)

## compute PSA with uncertainty
dt = 1.0 / fs
acc_mean = np.diff(x_mean) / dt
acc_std  = np.diff(x_std) / dt
periods, psa, psa_lo, psa_hi = compute_psa_with_uncertainty(acc_mean, acc_std, dt)
```

## Algorithm

The $\text{CWR}_{\Delta t}$ method extends the POCS-based CWR approach (Zhang et al., 2016) through:

1. **Time segmentation**: the waveform is split into overlapping segments of duration $\Delta t_p = c_l / f_c$, where $f_c$ is the corner frequency. This handles nonstationarity.
2. **Multiple realizations**: secondary subsegments with varying offsets produce $S$ independent reconstructions, yielding a statistical estimate of uncertainty.
3. **POCS iteration**: within each segment, Fourier amplitudes are iteratively thresholded (from $\alpha = 1 \to 0$), retaining maxima at each step.
4. **Optional frequency constraint**: ensures the peak derivative of the reconstruction does not exceed that of the clipped waveform.

## Modules

| Module              | Description                                          |
|---------------------|------------------------------------------------------|
| `detection.py`      | Amplitude + gradient clipping detection              |
| `reconstruction.py` | Core $\text{CWR}_{\Delta t}$ algorithm (Eqs. 1--8)  |
| `spectra.py`        | PSA, FAS, PSD computation with uncertainty           |
| `benchmark.py`      | Artificial clipping + misfit metrics (Eqs. 9--10)    |
| `utils.py`          | Corner frequency, Butterworth filter, Husid plot     |
| `visualization.py`  | Plotting functions                                   |

## Example

```bash
python example_usage.py
```

Generates a synthetic waveform, clips it, reconstructs via $\text{CWR}_{\Delta t}$, and produces comparison plots.
