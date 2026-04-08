"""waveclipy - seismic waveform clipping restoration (CWRdt).

implements the nonstationary clipped waveform restoration method
from Edwards et al. (2025), Seismological Research Letters.
"""

from waveclipy.detection import detect_clipping
from waveclipy.reconstruction import cwrdt
from waveclipy.spectra import compute_psa, compute_fas, compute_psd
from waveclipy.benchmark import artificial_clip, compute_misfit
from waveclipy.utils import corner_frequency, husid_plot

__version__ = "0.1.0"
__all__ = [
    "detect_clipping",
    "cwrdt",
    "compute_psa",
    "compute_fas",
    "compute_psd",
    "artificial_clip",
    "compute_misfit",
    "corner_frequency",
    "husid_plot",
]
