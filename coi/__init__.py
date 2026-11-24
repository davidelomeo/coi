"""Cyanobacteria Occurrence Index (COI) calculation package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version('coi')
except PackageNotFoundError:  # pragma: no cover - package not installed
    __version__ = 'unknown'

from .COI import COI  # noqa: E402
from .calculate_wsum_and_coi import calculate_wsum_and_coi  # noqa: E402
from .compute_spectral_angle import compute_spectral_angle  # noqa: E402

__all__ = [
    'COI',
    'compute_spectral_angle',
    'calculate_wsum_and_coi',
    '__version__',
]
