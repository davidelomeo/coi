"""Cyanobacteria Occurrence Index calculation package."""

from importlib.metadata import version, PackageNotFoundError
from .COI import COI
from .compute_spectral_angle import compute_spectral_angle
from .calculate_wsum_and_coi import calculate_wsum_and_coi

try:
    __version__ = version("cyanobacteria_occurrence_index")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

__all__ = [
    "COI",
    "compute_spectral_angle",
    "calculate_wsum_and_coi",
    "__version__",
]