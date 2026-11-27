"""Cyanobacteria Occurrence Index (COI) calculation package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version('coi')
except PackageNotFoundError:  # pragma: no cover - package not installed
    __version__ = 'unknown'

from .calculate_wsum_and_coi import calculate_wsum_and_coi
from .COI import COI, load_reference_owt
from .compute import CoiArrays, compute_coi
from .compute_spectral_angle import compute_spectral_angle

__all__ = [
    'COI',
    'CoiArrays',
    '__version__',
    'calculate_wsum_and_coi',
    'compute_coi',
    'compute_spectral_angle',
    'load_reference_owt',
]
