import importlib
import json
from importlib.resources import files

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import coi
from coi import COI, calculate_wsum_and_coi, compute_spectral_angle
from coi.cli import main
from coi.compute_spectral_angle import _trapezoid

# The module, not the COI class that the package exports under the same name
coi_module = importlib.import_module('coi.COI')

CORE = files('coi') / 'core_files'
REFERENCE = pd.read_csv(CORE / 'OWT_inland_mean_standardised_extended.csv', index_col='wl')
WEIGHTS = json.loads((CORE / 'w_owt.json').read_text())
OLCI = COI.SENSOR_BANDS['OLCI']
MODIS_RENAME = {k: v for k, v in COI.MODIS_BAND_MAP.items() if k != v}


@pytest.fixture
def ds():
    """Small scene: mixtures of reference OWTs plus noise, with a few missing pixels."""
    rng = np.random.default_rng(0)
    ny, nx = 12, 15
    a = rng.integers(0, REFERENCE.shape[1], (ny, nx))
    b = rng.integers(0, REFERENCE.shape[1], (ny, nx))
    mix = rng.uniform(0, 0.4, (ny, nx))
    data = {}
    for band in REFERENCE.index:
        ref = REFERENCE.loc[band].values
        values = (ref[a] * (1 - mix) + ref[b] * mix) * 0.05 * (1 + 0.05 * rng.standard_normal((ny, nx)))
        data[band] = (('y', 'x'), values)
    scene = xr.Dataset(data, coords={'y': np.arange(ny), 'x': np.arange(nx)})
    scene['Rw412'][0, 0] = np.nan
    for band in REFERENCE.index:
        scene[band][0, 1] = np.nan
    return scene


def slow_reference(spectrum, bands, reference, weights, min_wsum=223.0, max_wsum=321.0, exp=1.0):
    """Plain, one-pixel implementation of the method, used to check the fast version."""
    wl = np.array([float(b[2:]) for b in bands])
    order = np.argsort(wl)
    x = np.asarray(spectrum, dtype=float)
    x = x / _trapezoid(x[order], wl[order])
    scores = []
    for col in reference.columns:
        r = reference.loc[bands, col].values
        r = r / _trapezoid(r[order], wl[order])
        cos = np.dot(x, r) / (np.linalg.norm(x) * np.linalg.norm(r))
        scores.append(1 - np.arccos(np.clip(cos, -1, 1)) / np.pi)
    scores = np.array(scores)
    w = np.array([weights['sowt_' + c.split('OWT_')[1]] for c in reference.columns])
    fractions = scores ** exp / np.sum(scores ** exp)
    wsum = np.sum(scores * w * fractions)
    return wsum, float(np.clip((wsum - min_wsum) / (max_wsum - min_wsum), 0, 1))


def spectrum_at(scene, bands, y, x):
    return [float(scene[band][y, x]) for band in bands]


@pytest.mark.parametrize('kwargs', [{}, {'exp': 2.0, 'min_wsum': 200.0, 'max_wsum': 300.0}])
def test_matches_plain_implementation(ds, kwargs):
    out = COI(ds, sensor='OLCI', **kwargs).process_spectral_data()
    for y, x in [(1, 1), (5, 7), (11, 14)]:
        wsum, coi_value = slow_reference(spectrum_at(ds, OLCI, y, x), OLCI, REFERENCE, WEIGHTS, **kwargs)
        assert float(out['wsum'][y, x]) == pytest.approx(wsum, rel=1e-6)
        assert float(out['coi'][y, x]) == pytest.approx(coi_value, rel=1e-6, abs=1e-9)


def test_output_structure(ds):
    out = COI(ds, sensor='OLCI').process_spectral_data()
    assert list(out.data_vars) == ['wsum', 'coi']
    assert out['coi'].dims == ('y', 'x')
    np.testing.assert_array_equal(out['y'], ds['y'])
    valid = out['coi'].values[~np.isnan(out['coi'].values)]
    assert valid.min() >= 0 and valid.max() <= 1


def test_missing_band_values_give_nan(ds):
    out = COI(ds, sensor='OLCI').process_spectral_data()
    assert np.isnan(out['coi'][0, 0]) and np.isnan(out['coi'][0, 1])
    assert np.isfinite(out['coi'][1:, :]).all()


def test_reference_spectrum_scores_one(ds):
    for band in OLCI:
        ds[band][3, 3] = REFERENCE.loc[band, 'OWT_5'] * 0.02
    out = COI(ds, sensor='OLCI', return_all=True).process_spectral_data()
    assert float(out['sowt_5'][3, 3]) == pytest.approx(1.0)


def test_return_all_contents(ds):
    out = COI(ds, sensor='OLCI', return_all=True).process_spectral_data()
    sowt = [v for v in out.data_vars if v.startswith('sowt_')]
    assert len(sowt) == REFERENCE.shape[1]
    assert set(OLCI) <= set(out.data_vars)
    assert list(out.data_vars)[-2:] == ['wsum', 'coi']


def test_blocks_give_same_result(ds, monkeypatch):
    expected = COI(ds, sensor='OLCI', return_all=True).process_spectral_data()
    monkeypatch.setattr(coi_module, '_BLOCK_SIZE', 7)
    out = COI(ds, sensor='OLCI', return_all=True).process_spectral_data()
    xr.testing.assert_allclose(out, expected)


def test_valid_mask(ds):
    mask = xr.ones_like(ds['Rw443'], dtype=bool)
    mask[4, :] = False
    out = COI(ds, sensor='OLCI', valid_mask=mask).process_spectral_data()
    assert np.isnan(out['coi'][4, :]).all()
    assert np.isfinite(out['coi'][5, :]).all()


def test_custom_bands_equal_sensor_bands(ds):
    by_sensor = COI(ds, sensor='OLCI').process_spectral_data()
    by_bands = COI(ds, bands=OLCI).process_spectral_data()
    xr.testing.assert_allclose(by_sensor, by_bands)


def test_custom_band_subset(ds):
    bands = ['Rw443', 'Rw490', 'Rw560', 'Rw665', 'Rw681']
    out = COI(ds, bands=bands).process_spectral_data()
    wsum, _ = slow_reference(spectrum_at(ds, bands, 6, 6), bands, REFERENCE, WEIGHTS)
    assert float(out['wsum'][6, 6]) == pytest.approx(wsum, rel=1e-6)


def test_meris_ignores_rw674(ds):
    out = COI(ds.drop_vars('Rw674'), sensor='MERIS').process_spectral_data()
    meris = COI.SENSOR_BANDS['MERIS']
    wsum, _ = slow_reference(spectrum_at(ds, meris, 2, 2), meris, REFERENCE, WEIGHTS)
    assert float(out['wsum'][2, 2]) == pytest.approx(wsum, rel=1e-6)


@pytest.mark.parametrize('sensor', ['MODIS', 'modis'])
def test_modis_band_mapping(ds, sensor):
    modis_ds = ds.rename(MODIS_RENAME)
    out = COI(modis_ds, sensor=sensor).process_spectral_data()
    olci_names = list(COI.MODIS_BAND_MAP)
    reference = REFERENCE.loc[olci_names].rename(index=COI.MODIS_BAND_MAP)
    modis = COI.SENSOR_BANDS['MODIS']
    wsum, _ = slow_reference(spectrum_at(modis_ds, modis, 8, 3), modis, reference, WEIGHTS)
    assert float(out['wsum'][8, 3]) == pytest.approx(wsum, rel=1e-6)


def test_lowercase_sensor(ds):
    xr.testing.assert_allclose(COI(ds, sensor='olci').process_spectral_data(),
                               COI(ds, sensor='OLCI').process_spectral_data())


def test_mean_suffix(ds):
    suffixed = ds.rename({band: band + '_mean' for band in OLCI})
    xr.testing.assert_allclose(COI(suffixed, sensor='OLCI').process_spectral_data(),
                               COI(ds, sensor='OLCI').process_spectral_data())


def test_custom_weights_and_reference(ds):
    weights = {k: 1.0 for k in WEIGHTS}
    out = COI(ds, sensor='OLCI', w_owt=weights, reference_owt=REFERENCE).process_spectral_data()
    wsum, _ = slow_reference(spectrum_at(ds, OLCI, 4, 4), OLCI, REFERENCE, weights)
    assert float(out['wsum'][4, 4]) == pytest.approx(wsum, rel=1e-6)


def test_errors(ds):
    with pytest.raises(ValueError, match='not both'):
        COI(ds, sensor='OLCI', bands=['Rw443'])
    with pytest.raises(ValueError, match='Either sensor'):
        COI(ds)
    with pytest.raises(ValueError, match='Unsupported sensor'):
        COI(ds, sensor='SEAWIFS')
    with pytest.raises(KeyError, match='Rw999'):
        COI(ds, bands=['Rw443', 'Rw999'])
    with pytest.raises(KeyError, match=r"missing bands \['Rw412'\]"):
        COI(ds.drop_vars('Rw412'), sensor='OLCI').process_spectral_data()
    with pytest.raises(KeyError, match='weights are missing'):
        COI(ds, sensor='OLCI', w_owt={'sowt_5': 1}).process_spectral_data()


def test_public_helpers_match_class(ds):
    full = COI(ds, sensor='OLCI', return_all=True).process_spectral_data()
    recomputed = calculate_wsum_and_coi(full.drop_vars(['wsum', 'coi']), WEIGHTS, 223.0, 321.0)
    xr.testing.assert_allclose(recomputed['coi'], full['coi'])

    spectra = np.stack([ds[band].values.ravel() for band in OLCI], axis=1)[2:5]
    scores = compute_spectral_angle(spectra, OLCI, REFERENCE)
    assert scores.shape == (3, REFERENCE.shape[1])
    np.testing.assert_allclose(scores[0, 0], full['sowt_1.2'].values.ravel()[2])


def test_dask_is_lazy_and_matches(ds):
    pytest.importorskip('dask')
    lazy = COI(ds.chunk({'y': 5, 'x': 6}), sensor='OLCI', return_all=True).process_spectral_data()
    assert lazy['coi'].chunks is not None
    xr.testing.assert_allclose(lazy.compute(), COI(ds, sensor='OLCI', return_all=True).process_spectral_data())


def test_version():
    assert coi.__version__ != 'unknown'


def test_cli(ds, tmp_path):
    src = tmp_path / 'in.nc'
    ds.to_netcdf(src)
    expected = COI(ds, sensor='OLCI', exp=2.0, min_wsum=200.0, max_wsum=300.0).process_spectral_data()

    main(['-i', str(src), '-o', str(tmp_path / 'a.nc'), '-s', 'olci', '--exp', '2', '--min-wsum', '200', '--max-wsum', '300'])
    with xr.open_dataset(tmp_path / 'a.nc') as out:
        assert list(out.data_vars) == ['wsum', 'coi']
        np.testing.assert_allclose(out['coi'], expected['coi'])

    main(['-i', str(src), '-o', str(tmp_path / 'b.nc'), '-s', 'OLCI', '--return-all'])
    with xr.open_dataset(tmp_path / 'b.nc') as out:
        assert 'sowt_5' in out and 'Rw443' in out

    main(['-i', str(src), '-o', str(tmp_path / 'c.nc'), '-b', 'Rw443, Rw490,Rw560,Rw665,Rw681'])
    with xr.open_dataset(tmp_path / 'c.nc') as out:
        assert np.isfinite(out['coi'][5, 5])


def test_cli_failure_exits_nonzero(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(['-i', str(tmp_path / 'missing.nc'), '-o', str(tmp_path / 'x.nc'), '-s', 'OLCI'])
    assert exc.value.code == 1
