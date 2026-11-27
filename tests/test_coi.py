import importlib
import json
import subprocess
import sys
import warnings
from importlib.resources import files
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import coi
from coi import (
    COI,
    CoiArrays,
    calculate_wsum_and_coi,
    compute_coi,
    compute_spectral_angle,
    load_reference_owt,
)
from coi.cli import main

# The module, not the COI class that the package exports under the same name
coi_module = importlib.import_module('coi.COI')

CORE = files('coi') / 'core_files'
REFERENCE = pd.read_csv(CORE / 'OWT_inland_mean_standardised_extended.csv', index_col='wl')
WEIGHTS = json.loads((CORE / 'w_owt.json').read_text())
OLCI = COI.SENSOR_BANDS['OLCI']
DOMINANT = ['dominant_owt_1', 'dominant_owt_2', 'dominant_owt_3']
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


def slow_scores(spectrum, bands, reference):
    """Plain, one-pixel implementation of the OWT membership scores (Sowt)."""
    wl = np.array([float(b[2:]) for b in bands])
    order = np.argsort(wl)
    x = np.asarray(spectrum, dtype=float)
    x = x / np.trapezoid(x[order], wl[order])
    scores = []
    for col in reference.columns:
        r = reference.loc[bands, col].values
        r = r / np.trapezoid(r[order], wl[order])
        cos = np.dot(x, r) / (np.linalg.norm(x) * np.linalg.norm(r))
        scores.append(1 - np.arccos(np.clip(cos, -1, 1)) / np.pi)
    return np.array(scores)


def slow_reference(spectrum, bands, reference, weights, min_wsum=223.0, max_wsum=321.0, exp=1.0):
    """Plain, one-pixel implementation of Wsum and COI, used to check the fast version."""
    scores = slow_scores(spectrum, bands, reference)
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
    assert list(out.data_vars) == ['wsum', 'coi', *DOMINANT]
    assert all(out[v].dims == ('y', 'x') for v in out.data_vars)
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
    assert list(out.data_vars)[-5:] == ['wsum', 'coi', *DOMINANT]


def test_dominant_owts_are_the_highest_scores(ds):
    out = COI(ds, sensor='OLCI').process_spectral_data()
    names = [c.split('OWT_')[1] for c in REFERENCE.columns]
    for y, x in [(1, 1), (5, 7), (11, 14)]:
        scores = slow_scores(spectrum_at(ds, OLCI, y, x), OLCI, REFERENCE)
        expected = [names[i] for i in np.argsort(-scores)[:3]]
        assert [str(out[v][y, x].values) for v in DOMINANT] == expected


def test_dominant_owts_reference_and_missing(ds):
    for band in OLCI:
        ds[band][3, 3] = REFERENCE.loc[band, 'OWT_5'] * 0.02
    out = COI(ds, sensor='OLCI').process_spectral_data()
    assert str(out['dominant_owt_1'][3, 3].values) == '5'
    for var in DOMINANT:
        assert str(out[var][0, 0].values) == '' and str(out[var][0, 1].values) == ''
        assert (out[var].values[1:, :] != '').all()


def test_fewer_owts_than_three(ds):
    reference = REFERENCE[['OWT_5', 'OWT_13']]
    weights = {'sowt_5': 100, 'sowt_13': 0}
    out = COI(ds, sensor='OLCI', w_owt=weights, reference_owt=reference).process_spectral_data()
    assert list(out.data_vars) == ['wsum', 'coi', 'dominant_owt_1', 'dominant_owt_2']


def test_outputs_do_not_inherit_input_attributes(ds):
    ds['Rw412'].attrs['units'] = 'sr-1'
    out = COI(ds, sensor='OLCI', return_all=True).process_spectral_data()
    assert out['Rw412'].attrs == {'units': 'sr-1'}
    for var in ['wsum', 'coi', 'sowt_5', *DOMINANT]:
        assert 'units' not in out[var].attrs


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
    expected = COI(ds, sensor='OLCI', return_all=True).process_spectral_data()
    xr.testing.assert_allclose(lazy.compute(), expected)


def test_version():
    assert coi.__version__ != 'unknown'


def test_cli(ds, tmp_path):
    src = tmp_path / 'in.nc'
    ds.to_netcdf(src)
    expected = COI(ds, sensor='OLCI', exp=2.0, min_wsum=200.0, max_wsum=300.0).process_spectral_data()

    main(['-i', str(src), '-o', str(tmp_path / 'a.nc'), '-s', 'olci',
          '--exp', '2', '--min-wsum', '200', '--max-wsum', '300'])
    with xr.open_dataset(tmp_path / 'a.nc') as out:
        assert list(out.data_vars) == ['wsum', 'coi', *DOMINANT]
        np.testing.assert_allclose(out['coi'], expected['coi'])
        for var in DOMINANT:
            np.testing.assert_array_equal(out[var].values.astype(str), expected[var].values)

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


def test_cli_text_conversion_round_trips(tmp_path):
    from coi.cli import _ascii_text_as_bytes
    ds = xr.Dataset({'ascii': (('y', 'x'), np.array([['6.2', ''], ['1.1_7.1', '13']])),
                     'other': (('y', 'x'), np.array([['é', 'x'], ['', 'ü1']]))})
    for var in ds.data_vars:
        ds[var].encoding = {'dtype': 'S1'}
    converted = _ascii_text_as_bytes(ds)
    assert converted['ascii'].dtype.kind == 'S' and converted['other'].dtype.kind == 'U'
    converted.to_netcdf(tmp_path / 't.nc', encoding={v: converted[v].encoding for v in converted.data_vars})
    with xr.open_dataset(tmp_path / 't.nc') as back:
        for var in ds.data_vars:
            np.testing.assert_array_equal(back[var].values.astype(str), ds[var].values)


OLCI_WL = np.array([float(band[2:]) for band in OLCI])


def cube_of(scene, bands=OLCI):
    """The scene as one array with wavelength as the last axis, shape (y, x, band)."""
    return np.stack([scene[band].values for band in bands], axis=-1)


def spectral_dataarray(scene, dim='wavelength'):
    return xr.DataArray(cube_of(scene), dims=('y', 'x', dim),
                        coords={'y': scene['y'], 'x': scene['x'], dim: OLCI_WL}, name='rrs')


def assert_matches_class(result, expected):
    np.testing.assert_allclose(result.wsum, expected['wsum'])
    np.testing.assert_allclose(result.coi, expected['coi'])
    for rank, name in enumerate(DOMINANT):
        np.testing.assert_array_equal(result.dominant_owt[..., rank], expected[name].values)


@pytest.mark.parametrize('kwargs', [{}, {'exp': 2.0, 'min_wsum': 200.0, 'max_wsum': 300.0}])
def test_numpy_input_matches_class(ds, kwargs):
    result = compute_coi(cube_of(ds), OLCI_WL, sensor='OLCI', **kwargs)
    assert isinstance(result, CoiArrays) and result.scores is None
    assert result.wsum.shape == result.coi.shape == (12, 15)
    assert result.dominant_owt.shape == (12, 15, 3)
    assert_matches_class(result, COI(ds, sensor='OLCI', **kwargs).process_spectral_data())


def test_numpy_single_spectrum_and_empty_input(ds):
    wsum, coi_value, dominant, *_ = compute_coi(cube_of(ds)[5, 7], OLCI_WL, sensor='olci')
    expected, _ = slow_reference(spectrum_at(ds, OLCI, 5, 7), OLCI, REFERENCE, WEIGHTS)
    assert wsum.shape == coi_value.shape == () and dominant.shape == (3,)
    assert float(wsum) == pytest.approx(expected, rel=1e-6)
    empty = compute_coi(np.empty((0, len(OLCI))), OLCI_WL, sensor='OLCI')
    assert empty.coi.shape == (0,) and empty.dominant_owt.shape == (0, 3)


def test_numpy_bands_are_found_by_wavelength(ds):
    # Any order, extra bands, and a centre such as 682 nm for the Rw681 band
    wavelengths = np.concatenate([[400.0], np.where(OLCI_WL == 681, 682, OLCI_WL), [865.0]])
    cube = cube_of(ds)
    padded = np.concatenate([cube[..., :1] * 3, cube, cube[..., -1:] * 3], axis=-1)
    order = np.random.default_rng(1).permutation(wavelengths.size)
    result = compute_coi(padded[..., order], wavelengths[order], sensor='OLCI')
    assert_matches_class(result, COI(ds, sensor='OLCI').process_spectral_data())


def test_numpy_custom_bands_and_modis(ds):
    bands = ['Rw443', 'Rw490', 'Rw560', 'Rw665', 'Rw681']
    result = compute_coi(cube_of(ds), OLCI_WL, bands=bands)
    assert_matches_class(result, COI(ds, bands=bands).process_spectral_data())
    modis_ds = ds.rename(MODIS_RENAME)
    modis = COI.SENSOR_BANDS['MODIS']
    result = compute_coi(cube_of(modis_ds, modis), [float(b[2:]) for b in modis], sensor='MODIS')
    assert_matches_class(result, COI(modis_ds, sensor='MODIS').process_spectral_data())


def test_numpy_scores_mask_and_blocks(ds, monkeypatch):
    mask = np.arange(15) != 4  # broadcast over y
    expected = COI(ds, sensor='OLCI', return_all=True,
                   valid_mask=xr.DataArray(mask, dims='x')).process_spectral_data()
    monkeypatch.setattr(coi_module, '_BLOCK_SIZE', 7)
    result = compute_coi(cube_of(ds), OLCI_WL, sensor='OLCI', valid_mask=mask, return_scores=True)
    assert result.scores.shape == (12, 15, REFERENCE.shape[1])
    assert_matches_class(result, expected)
    np.testing.assert_allclose(result.scores[..., 0], expected['sowt_1.2'])
    assert np.isnan(result.coi[:, 4]).all()


def test_numpy_masked_values_are_missing(ds):
    cube = np.ma.masked_array(cube_of(ds))
    cube[5, 5, 3] = np.ma.masked
    result = compute_coi(cube, OLCI_WL, sensor='OLCI')
    assert np.isnan(result.coi[5, 5]) and (result.dominant_owt[5, 5] == '').all()
    assert np.isfinite(result.coi[5, 6])


def test_dataarray_input(ds):
    expected = COI(ds, sensor='OLCI').process_spectral_data()
    xr.testing.assert_allclose(compute_coi(spectral_dataarray(ds), sensor='OLCI'), expected)
    # Wavelength first, and a dimension name that has to be given
    named = spectral_dataarray(ds, 'z').transpose('z', 'y', 'x')
    xr.testing.assert_allclose(compute_coi(named, sensor='OLCI', dim='z'), expected)
    with pytest.raises(ValueError, match='pass dim'):
        compute_coi(named, sensor='OLCI')
    # Wavelengths in a separate coordinate of a band dimension
    banded = spectral_dataarray(ds, 'band').assign_coords(
        band=np.arange(len(OLCI)), wavelength=('band', OLCI_WL))
    xr.testing.assert_allclose(compute_coi(banded, sensor='OLCI'), expected)
    unlabelled = named.drop_vars('z')
    with pytest.raises(ValueError, match='no wavelength coordinate'):
        compute_coi(unlabelled, sensor='OLCI', dim='z')
    xr.testing.assert_allclose(
        compute_coi(unlabelled, OLCI_WL, sensor='OLCI', dim='z'), expected)


def test_dataarray_scores(ds):
    result = compute_coi(spectral_dataarray(ds), sensor='OLCI', return_scores=True)
    assert 'sowt_5' in result and not set(OLCI) & set(result.data_vars)


def test_dataarray_with_dask_is_lazy(ds):
    pytest.importorskip('dask')
    data = spectral_dataarray(ds)
    lazy = compute_coi(data.chunk({'y': 5, 'wavelength': 4}), sensor='OLCI')
    assert lazy['coi'].chunks is not None
    xr.testing.assert_allclose(lazy.compute(), compute_coi(data, sensor='OLCI'))


@pytest.mark.parametrize('columns', [OLCI_WL, OLCI_WL.astype(int).astype(str), OLCI])
def test_dataframe_input(ds, columns):
    cube = cube_of(ds)[1:3].reshape(-1, len(OLCI))
    frame = pd.DataFrame(cube, columns=columns, index=[f'obs{i}' for i in range(len(cube))])
    out = compute_coi(frame, sensor='OLCI')
    assert list(out.columns) == ['wsum', 'coi', *DOMINANT]
    pd.testing.assert_index_equal(out.index, frame.index)
    expected = COI(ds, sensor='OLCI').process_spectral_data().isel(y=slice(1, 3))
    np.testing.assert_allclose(out['coi'], expected['coi'].values.ravel())
    np.testing.assert_array_equal(out['dominant_owt_1'], expected['dominant_owt_1'].values.ravel())
    scored = compute_coi(frame, sensor='OLCI', return_scores=True)
    assert scored.shape[1] == 5 + REFERENCE.shape[1] and 'sowt_5' in scored


def test_compute_coi_errors(ds):
    cube = cube_of(ds)
    with pytest.raises(ValueError, match='Pass wavelengths'):
        compute_coi(cube, sensor='OLCI')
    with pytest.raises(ValueError, match='last axis'):
        compute_coi(cube.swapaxes(1, 2), OLCI_WL, sensor='OLCI')
    with pytest.raises(ValueError, match='only applies to xarray'):
        compute_coi(cube, OLCI_WL, sensor='OLCI', dim='wavelength')
    with pytest.raises(TypeError, match='process_spectral_data'):
        compute_coi(ds, sensor='OLCI')
    with pytest.raises(KeyError, match=r"missing bands \['Rw412'\]"):
        compute_coi(cube[..., 1:], OLCI_WL[1:], sensor='OLCI')
    with pytest.raises(ValueError, match='same input wavelength'):
        compute_coi(cube[..., :2], [412.0, 677.0], bands=['Rw674', 'Rw681'], tolerance=5)
    with pytest.raises(ValueError, match='finite'):
        compute_coi(cube, np.where(OLCI_WL == 443, np.nan, OLCI_WL), sensor='OLCI')
    with pytest.raises(ValueError, match='Either sensor'):
        compute_coi(cube, OLCI_WL)


def test_hyperspectral_input_warns(ds):
    wavelengths = np.arange(400.0, 801.0)
    spectrum = np.interp(wavelengths, OLCI_WL, cube_of(ds)[5, 5])
    with pytest.warns(UserWarning, match='looks hyperspectral'):
        compute_coi(spectrum, wavelengths, sensor='OLCI')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        compute_coi(cube_of(ds), OLCI_WL, sensor='OLCI')


def test_load_reference_owt(ds):
    table = load_reference_owt()
    pd.testing.assert_frame_equal(table, REFERENCE)
    assert list(table.index) == OLCI and table.index.name == 'wl' and table.shape == (12, 25)
    # Changing one table does not affect the next, or the library COI uses
    table.iloc[:, :] = 0.0
    pd.testing.assert_frame_equal(load_reference_owt(), REFERENCE)
    # It can be passed back, also as a subset of the water types
    expected = COI(ds, sensor='OLCI').process_spectral_data()
    given = COI(ds, sensor='OLCI', reference_owt=load_reference_owt()).process_spectral_data()
    xr.testing.assert_identical(given, expected)
    subset = load_reference_owt()[['OWT_2', 'OWT_3']]
    result = COI(ds, sensor='OLCI', reference_owt=subset).process_spectral_data()
    assert set(np.unique(result['dominant_owt_1'])) <= {'2', '3', ''}


def own_range(wsum):
    return (wsum - np.nanmin(wsum)) / (np.nanmax(wsum) - np.nanmin(wsum))


def test_local_coi_dataset(ds):
    plain = COI(ds, sensor='OLCI').process_spectral_data()
    out = COI(ds, sensor='OLCI', local_coi=True).process_spectral_data()
    assert list(plain.data_vars) == ['wsum', 'coi', *DOMINANT]
    added = ['wsum_local', 'coi_local', 'wsum_local_min', 'wsum_local_max']
    assert list(out.data_vars) == ['wsum', 'coi', *added, *DOMINANT]
    xr.testing.assert_identical(out[list(plain.data_vars)], plain)  # the global COI is unchanged
    np.testing.assert_array_equal(out['wsum_local'], plain['wsum'])
    assert not np.shares_memory(out['wsum_local'].values, out['wsum'].values)
    wsum = plain['wsum'].values
    assert out['wsum_local_min'].ndim == 0 and out['wsum_local_min'] == np.nanmin(wsum)
    assert out['wsum_local_max'] == np.nanmax(wsum)
    np.testing.assert_allclose(out['coi_local'], own_range(wsum))
    assert np.nanmin(out['coi_local']) == 0 and np.nanmax(out['coi_local']) == 1
    assert np.isnan(out['coi_local'][0, 0])


def test_local_coi_ignores_the_global_boundaries(ds):
    default = COI(ds, sensor='OLCI', local_coi=True).process_spectral_data()
    moved = COI(ds, sensor='OLCI', local_coi=True,
                min_wsum=0.0, max_wsum=1000.0).process_spectral_data()
    xr.testing.assert_identical(moved['coi_local'], default['coi_local'])
    assert not np.allclose(moved['coi'], default['coi'], equal_nan=True)


def test_local_coi_boundaries_skip_masked_pixels(ds):
    wsum = COI(ds, sensor='OLCI').process_spectral_data()['wsum']
    # Leave out the pixels holding the smallest and the largest Wsum
    mask = (wsum > wsum.min()) & (wsum < wsum.max())
    result = compute_coi(cube_of(ds), OLCI_WL, sensor='OLCI', valid_mask=mask.values, local_coi=True)
    kept = np.where(mask, wsum, np.nan)
    np.testing.assert_allclose(result.coi_local, own_range(kept))
    np.testing.assert_allclose(result.wsum_local, kept)  # masked pixels are NaN here too
    assert np.nanmin(result.coi_local) == 0 and np.nanmax(result.coi_local) == 1


@pytest.mark.parametrize('cube_slice', [
    np.s_[5, 7],           # one spectrum
    np.s_[0, :2],          # no spectrum with a result
    np.s_[0:0, :],         # no spectrum at all
    np.s_[[5, 5], [7, 7]],  # twice the same value
])
def test_local_coi_without_a_range_is_nan(ds, cube_slice):
    cube = cube_of(ds)[cube_slice]
    data = xr.DataArray(cube, dims=[*'ab'[:cube.ndim - 1], 'wavelength'], coords={'wavelength': OLCI_WL})
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        result = compute_coi(cube, OLCI_WL, sensor='OLCI', local_coi=True)
        labelled = compute_coi(data, sensor='OLCI', local_coi=True)
    assert result.coi_local.shape == cube.shape[:-1] and np.isnan(result.coi_local).all()
    assert np.isnan(labelled['coi_local']).all()


def test_local_coi_array_and_table_inputs(ds):
    expected = COI(ds, sensor='OLCI', local_coi=True).process_spectral_data()
    plain = compute_coi(cube_of(ds), OLCI_WL, sensor='OLCI')
    assert plain.coi_local is None and plain.wsum_local is None
    result = compute_coi(cube_of(ds), OLCI_WL, sensor='OLCI', local_coi=True)
    np.testing.assert_allclose(result.coi_local, expected['coi_local'])
    np.testing.assert_allclose(result.wsum_local, expected['wsum_local'])
    assert not np.shares_memory(result.wsum_local, result.wsum)
    xr.testing.assert_allclose(
        compute_coi(spectral_dataarray(ds), sensor='OLCI', local_coi=True), expected)

    frame = pd.DataFrame(cube_of(ds).reshape(-1, len(OLCI)), columns=OLCI_WL)
    table = compute_coi(frame, sensor='OLCI', local_coi=True)
    assert list(table.columns) == ['wsum', 'coi', 'wsum_local', 'coi_local', *DOMINANT]
    np.testing.assert_array_equal(table['wsum_local'], table['wsum'])
    np.testing.assert_allclose(table['coi_local'], expected['coi_local'].values.ravel())


def test_local_coi_with_dask_uses_all_chunks_and_is_lazy(ds):
    pytest.importorskip('dask')
    data = spectral_dataarray(ds)
    lazy = compute_coi(data.chunk({'y': 5, 'x': 6}), sensor='OLCI', local_coi=True)
    for name in ('wsum_local', 'coi_local', 'wsum_local_min', 'wsum_local_max'):
        assert lazy[name].chunks is not None
    xr.testing.assert_allclose(lazy.compute(), compute_coi(data, sensor='OLCI', local_coi=True))


@pytest.fixture
def spectral_file(ds, tmp_path):
    """A file laid out like satconv output: one spectral variable plus per-observation data."""
    scene = xr.Dataset(
        {'rrs': spectral_dataarray(ds), 'quality': (('y', 'x'), np.ones((12, 15), dtype=np.int8)),
         'station': ('y', [f'station {i}' for i in range(12)])},
        attrs={'history': 'created'})
    path = tmp_path / 'spectra.nc'
    scene.to_netcdf(path)
    return path


@pytest.mark.parametrize('extra', [[], ['--variable', 'rrs'], ['--dim', 'wavelength']])
def test_cli_wavelength_variable(ds, spectral_file, tmp_path, extra):
    out = tmp_path / 'coi.nc'
    main(['-i', str(spectral_file), '-o', str(out), '-s', 'OLCI',
          '--keep', 'quality', 'station', 'y', *extra])
    with xr.open_dataset(out) as result:
        expected = COI(ds, sensor='OLCI').process_spectral_data()
        assert list(result.data_vars) == ['wsum', 'coi', *DOMINANT, 'quality', 'station']
        np.testing.assert_allclose(result['coi'], expected['coi'])
        assert result['quality'].dtype == np.int8
        assert str(result['station'].values[3]) == 'station 3'
        assert result.attrs['history'].startswith('created\n')
        assert result.attrs['processing_date'].endswith('+00:00')


def test_cli_return_all_from_wavelength_variable(spectral_file, tmp_path):
    out = tmp_path / 'all.nc'
    main(['-i', str(spectral_file), '-o', str(out), '-s', 'OLCI', '--return-all'])
    with xr.open_dataset(out) as result:
        assert set(OLCI) <= set(result.data_vars) and 'sowt_5' in result


def test_cli_local_coi(ds, spectral_file, tmp_path):
    out = tmp_path / 'local.nc'
    main(['-i', str(spectral_file), '-o', str(out), '-s', 'OLCI', '--local-coi'])
    expected = COI(ds, sensor='OLCI', local_coi=True).process_spectral_data()
    with xr.open_dataset(out) as result:
        assert list(result.data_vars) == list(expected.data_vars)
        for name in ('coi', 'wsum_local', 'coi_local', 'wsum_local_min', 'wsum_local_max'):
            np.testing.assert_allclose(result[name], expected[name])


@pytest.fixture
def flagged_file(ds, tmp_path):
    """Spectra with a quality flag per pixel (1 to 4) and one per row."""
    flags = np.arange(12 * 15, dtype=np.int16).reshape(12, 15) % 4 + 1
    scene = xr.Dataset(
        {'rrs': spectral_dataarray(ds), 'flag': (('y', 'x'), flags),
         'row_flag': ('y', np.arange(12) % 2),
         'band_flag': ('wavelength', np.ones(len(OLCI))),
         'note': ('y', ['good'] * 12)})
    path = tmp_path / 'flagged.nc'
    scene.to_netcdf(path)
    return path


@pytest.mark.parametrize('options, keep', [
    (['--valid-where', 'flag=1'], lambda flag, row: flag == 1),
    (['--valid-where', 'flag=1,2'], lambda flag, row: flag <= 2),
    (['--valid-where', 'row_flag=1'], lambda flag, row: row == 1),
    (['--valid-where', 'flag=1,2', '--valid-where', 'row_flag=1'],
     lambda flag, row: (flag <= 2) & (row == 1)),
    (['--valid-where', 'flag=9'], lambda flag, row: flag == 9),
])
def test_cli_valid_where(ds, flagged_file, tmp_path, options, keep):
    out = tmp_path / 'valid.nc'
    main(['-i', str(flagged_file), '-o', str(out), '-s', 'OLCI', '-v', 'rrs', '--local-coi', *options])
    with xr.open_dataset(flagged_file) as source:
        mask = keep(source['flag'], source['row_flag']).broadcast_like(source['flag'])
        mask = mask.transpose('y', 'x').load()
    expected = COI(ds, sensor='OLCI', valid_mask=mask, local_coi=True).process_spectral_data()
    with xr.open_dataset(out) as result:
        assert result['coi'].dims == ('y', 'x')
        for name in ('coi', 'wsum_local', 'coi_local', 'wsum_local_min', 'wsum_local_max'):
            np.testing.assert_allclose(result[name], expected[name])
        assert np.isnan(result['coi'].values[~mask.values]).all()


def test_cli_valid_where_with_band_variables(ds, tmp_path):
    src, out = tmp_path / 'bands.nc', tmp_path / 'coi.nc'
    ds.assign(flag=('x', np.arange(15) % 2)).to_netcdf(src)
    main(['-i', str(src), '-o', str(out), '-s', 'OLCI', '--valid-where', 'flag=0'])
    with xr.open_dataset(out) as result:
        assert np.isnan(result['coi'][:, 1::2]).all()
        assert np.isfinite(result['coi'][5, 0::2]).all()


@pytest.mark.parametrize('condition, message', [
    ('nope=1', 'no variable nope'),
    ('note=1', 'note is not a number variable'),
    ('band_flag=1', 'band_flag has dimension wavelength'),
])
def test_cli_valid_where_errors(flagged_file, tmp_path, caplog, condition, message):
    with pytest.raises(SystemExit) as exc:
        main(['-i', str(flagged_file), '-o', str(tmp_path / 'x.nc'), '-s', 'OLCI', '-v', 'rrs',
              '--valid-where', condition])
    assert exc.value.code == 1 and message in caplog.text
    assert not (tmp_path / 'x.nc').exists()


@pytest.mark.parametrize('condition', ['flag', 'flag=', '=1', 'flag=good', 'flag=1,'])
def test_cli_valid_where_syntax(flagged_file, tmp_path, capsys, condition):
    with pytest.raises(SystemExit) as exc:
        main(['-i', str(flagged_file), '-o', str(tmp_path / 'x.nc'), '-s', 'OLCI',
              '--valid-where', condition])
    assert exc.value.code == 2 and 'expected NAME=VALUE' in capsys.readouterr().err


@pytest.mark.parametrize('extra, message', [
    (['--variable', 'nope'], 'No variable nope'),
    (['--variable', 'quality'], 'Cannot tell which'),
    (['--dim', 'nope'], "No variable has a dimension 'nope'"),
    (['--keep', 'nope'], '--keep: no variable nope'),
])
def test_cli_wavelength_errors(spectral_file, tmp_path, caplog, extra, message):
    with pytest.raises(SystemExit) as exc:
        main(['-i', str(spectral_file), '-o', str(tmp_path / 'x.nc'), '-s', 'OLCI', *extra])
    assert exc.value.code == 1 and message in caplog.text
    assert not (tmp_path / 'x.nc').exists()


def test_cli_asks_which_variable(ds, tmp_path, caplog):
    src = tmp_path / 'two.nc'
    xr.Dataset({'rrs': spectral_dataarray(ds), 'other': spectral_dataarray(ds)}).to_netcdf(src)
    with pytest.raises(SystemExit):
        main(['-i', str(src), '-o', str(tmp_path / 'x.nc'), '-s', 'OLCI'])
    assert 'choose with --variable: rrs, other' in caplog.text


def test_cli_does_not_overwrite_input(spectral_file, caplog):
    before = spectral_file.read_bytes()
    with pytest.raises(SystemExit):
        main(['-i', str(spectral_file), '-o', str(spectral_file), '-s', 'OLCI'])
    assert 'must not be the input' in caplog.text and spectral_file.read_bytes() == before


def test_module_can_be_run():
    done = subprocess.run([sys.executable, '-m', 'coi', '--version'],
                          capture_output=True, text=True, check=True)
    assert done.stdout == f'coi {coi.__version__}\n'


def write_variable(nc, name, kind, dims, values, **attrs):
    fill = attrs.pop('_FillValue', None)
    variable = nc.createVariable(name, kind, dims, fill_value=fill)
    variable.set_auto_maskandscale(False)
    variable.set_auto_chartostring(False)
    variable.setncatts(attrs)
    variable[...] = values


def stored_contents(path):
    """Every variable of a netCDF file as it is stored: type, dimensions, attributes, values."""
    contents = {}
    with netCDF4.Dataset(path) as nc:
        for name, variable in nc.variables.items():
            variable.set_auto_maskandscale(False)
            variable.set_auto_chartostring(False)
            data = np.asarray(variable[...])
            values = data.tolist() if data.dtype.kind in 'OSU' else data.tobytes()
            attrs = {key: repr(variable.getncattr(key)) for key in variable.ncattrs()}
            contents[name] = (str(variable.dtype), variable.dimensions, attrs, values)
    return contents


@pytest.fixture
def awkward(ds, tmp_path):
    """Spectra plus variables stored in ways that decoding and re-encoding can change."""
    path = tmp_path / 'awkward.nc'
    spectra = cube_of(ds)[1:5, 0]
    with netCDF4.Dataset(path, 'w') as nc:
        nc.createDimension('time', 4)
        nc.createDimension('wavelength', len(OLCI))
        nc.createDimension('length', 3)
        write_variable(nc, 'time', 'f8', ('time',), np.arange(4.0),
                       units='seconds since 2020-01-01', calendar='standard')
        write_variable(nc, 'wavelength', 'f8', ('wavelength',), OLCI_WL)
        write_variable(nc, 'rrs', 'f8', ('time', 'wavelength'), spectra, coordinates='lat')
        write_variable(nc, 'lat', 'f4', ('time',), np.arange(4, dtype='f4'))
        write_variable(nc, 'sst', 'i2', ('time',), np.array([40, -1, -32768, -2], 'i2'),
                       _Unsigned='true', scale_factor=np.float32(0.5), _FillValue=np.int16(-2))
        write_variable(nc, 'flags', 'i4', ('time',), np.array([0, -1, -2147483648, 7], 'i4'),
                       _Unsigned='true')
        write_variable(nc, 'two_fills', 'f4', ('time',), np.array([1, -9, 3, -8], 'f4'),
                       _FillValue=np.float32(-9), missing_value=np.float32(-8))
        write_variable(nc, 'no_fill', 'f4', ('time',), np.array([1, 2, 3, np.nan], 'f4'))
        write_variable(nc, 'boolean', 'i1', ('time',), np.array([1, 0, 1, 1], 'i1'),
                       **{'dtype': 'bool'})
        write_variable(nc, 'text', str, ('time',), np.array(['a', 'bb', '', 'é'], object))
        letters = netCDF4.stringtochar(np.array([b'x', b'yy', b'zzz', b''], 'S3'))
        write_variable(nc, 'chars', 'S1', ('time', 'length'), letters, _Encoding='utf-8')
        write_variable(nc, 'letter', 'S1', ('time',), np.array([b'a', b'b', b'', b'd'], 'S1'))
        write_variable(nc, 'number', 'i4', (), 7, comment='scalar')
        flag = nc.createEnumType('u1', 'flag_t', {'bad': 0, 'good': 1})
        write_variable(nc, 'quality', flag, ('time',), np.array([0, 1, 1, 0], 'u1'))
        pair = nc.createCompoundType(np.dtype([('a', 'f4'), ('b', 'i4')]), 'pair_t')
        nc.createVariable('pair', pair, ('time',))[:] = np.array([(1, 2)] * 4, pair.dtype)
        ragged = nc.createVariable('ragged', nc.createVLType('i4', 'ragged_t'), ('time',))
        for i in range(4):
            ragged[i] = np.arange(i + 1, dtype='i4')
    return path


def test_cli_keeps_variables_as_stored(awkward, tmp_path):
    out = tmp_path / 'coi.nc'
    names = ['sst', 'flags', 'two_fills', 'no_fill', 'boolean', 'text', 'chars', 'letter',
             'number', 'quality', 'rrs']
    main(['-i', str(awkward), '-o', str(out), '-s', 'OLCI', '--keep', *names])
    before, after = stored_contents(awkward), stored_contents(out)
    for name in [*names, 'lat', 'wavelength', 'time']:  # with the coordinates of the kept variables
        stored_type, dims, attrs, values = after[name]
        if 'coordinates' not in before[name][2]:
            attrs.pop('coordinates', None)  # added to say which output coordinates apply
        assert (stored_type, dims, attrs, values) == before[name], name
    with xr.open_dataset(out) as result:
        assert 'lat' in result.coords and np.isfinite(result['coi']).all()


@pytest.mark.parametrize('name', ['pair', 'ragged'])
def test_cli_reports_types_that_cannot_be_kept(awkward, tmp_path, caplog, name):
    out = tmp_path / 'coi.nc'
    out.write_bytes(b'earlier result')
    with pytest.raises(SystemExit):
        main(['-i', str(awkward), '-o', str(out), '-s', 'OLCI', '--keep', name])
    assert f'{name} has a compound or variable-length type' in caplog.text
    assert out.read_bytes() == b'earlier result'


def test_cli_failed_write_leaves_the_old_output(spectral_file, tmp_path, caplog, monkeypatch):
    def fail(self, path, *args, **kwargs):
        Path(path).write_bytes(b'partial')
        raise RuntimeError('NetCDF: HDF error')

    monkeypatch.setattr(xr.Dataset, 'to_netcdf', fail)
    out = tmp_path / 'coi.nc'
    out.write_bytes(b'earlier result')
    contents = sorted(tmp_path.iterdir())
    with pytest.raises(SystemExit):
        main(['-i', str(spectral_file), '-o', str(out), '-s', 'OLCI'])
    assert 'HDF error' in caplog.text
    assert out.read_bytes() == b'earlier result' and sorted(tmp_path.iterdir()) == contents


def test_cli_home_folder_paths(spectral_file, tmp_path, caplog, monkeypatch):
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.chdir(elsewhere)
    before = spectral_file.read_bytes()
    # A shell leaves '~' in --option=~/file for the program to expand
    same = [f'--input=~/{spectral_file.name}', f'--output=~/{spectral_file.name}', '-s', 'OLCI']
    with pytest.raises(SystemExit):
        main(same)
    assert 'must not be the input' in caplog.text and spectral_file.read_bytes() == before
    for _ in range(2):  # the second run replaces the output of the first
        main([f'--input=~/{spectral_file.name}', '--output=~/coi.nc', '-s', 'OLCI'])
    assert (tmp_path / 'coi.nc').exists() and not list(elsewhere.iterdir())


def test_cli_rw_prefix_alone_is_not_a_band(ds, tmp_path):
    src, out = tmp_path / 'in.nc', tmp_path / 'out.nc'
    xr.Dataset({'Rw': spectral_dataarray(ds), 'Rw_QC': ('y', np.ones(12))}).to_netcdf(src)
    main(['-i', str(src), '-o', str(out), '-s', 'OLCI'])
    with xr.open_dataset(out) as result:
        np.testing.assert_allclose(result['coi'], COI(ds, sensor='OLCI').process_spectral_data()['coi'])


def test_cli_wavelengths_in_a_data_variable(ds, tmp_path):
    src, out = tmp_path / 'in.nc', tmp_path / 'out.nc'
    xr.Dataset({'rrs': (('y', 'x', 'band'), cube_of(ds)),
                'wavelength': ('band', OLCI_WL)}).to_netcdf(src)
    main(['-i', str(src), '-o', str(out), '-s', 'OLCI'])
    with xr.open_dataset(out) as result:
        np.testing.assert_allclose(result['coi'], COI(ds, sensor='OLCI').process_spectral_data()['coi'])


@pytest.mark.parametrize('dim', ['wavelength', 'band'])
def test_dataarray_with_coordinates_along_wavelength(ds, dim):
    plain = spectral_dataarray(ds)
    data = spectral_dataarray(ds, dim)
    if dim == 'band':
        data = data.assign_coords(wavelength=('band', OLCI_WL))
    data = data.assign_coords(width=(('y', dim), np.ones((12, len(OLCI)))),
                              label=(dim, [str(w) for w in OLCI_WL]))
    result = compute_coi(data, sensor='OLCI')
    xr.testing.assert_allclose(result, compute_coi(plain, sensor='OLCI'))


@pytest.mark.parametrize('empty', [None, np.nan])
def test_empty_custom_weight_counts_as_zero(ds, empty):
    blank = {**WEIGHTS, 'sowt_5': empty}
    zero = {**WEIGHTS, 'sowt_5': 0.0}
    out = COI(ds, sensor='OLCI', w_owt=blank).process_spectral_data()
    xr.testing.assert_identical(out, COI(ds, sensor='OLCI', w_owt=zero).process_spectral_data())
    assert np.isfinite(out['coi']).any()


@pytest.mark.parametrize('step, tolerance', [(2.5, 1.5), (3.0, 1.5), (5.0, 2.5), (10.0, 5.0)])
def test_coarser_hyperspectral_input_warns(ds, step, tolerance):
    wavelengths = np.arange(400.0, 801.0, step)
    spectrum = np.interp(wavelengths, OLCI_WL, cube_of(ds)[5, 5])
    with pytest.warns(UserWarning, match='looks hyperspectral'):
        compute_coi(spectrum, wavelengths, sensor='OLCI', tolerance=tolerance)


@pytest.mark.parametrize('step', [3.3, 5.0])
def test_missing_bands_in_hyperspectral_input_give_a_hint(step):
    wavelengths = np.arange(400.0, 801.0, step)
    with pytest.raises(KeyError, match=r'looks hyperspectral.*satconv'):
        compute_coi(np.ones(wavelengths.size), wavelengths, sensor='OLCI')
