#!/usr/bin/env python3
import importlib.util
import math
import sys
from pathlib import Path
import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PACKAGE_ROOT / 'scripts' / 'generate_opacity_tables_planck_scattering.py'
spec = importlib.util.spec_from_file_location('collapse', MODULE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f'Could not import {MODULE_PATH}')
mod = importlib.util.module_from_spec(spec)
sys.modules['collapse'] = mod
spec.loader.exec_module(mod)


def regroup(collapse, edges, temperature):
    logp, logr = mod._gl_group_log_weights(edges, temperature)
    return mod._regroup_collapse_fields(collapse, logp, logr)


def grey_targets(u, bb, bf, ff, sc):
    absorption = bb + bf + ff
    total = absorption + sc
    return {
        'kplanck': mod.planck_mean(u, absorption),
        'krosseland': mod.rosseland_mean(u, None, total),
        'kplanck_scattering': mod.planck_mean(u, sc),
        'krosseland_absorption': mod.rosseland_flux_weighted_absorption(u, None, absorption, total),
    }


def assert_close(actual, expected, rel, label):
    assert math.isclose(actual, expected, rel_tol=rel, abs_tol=1e-300), (label, actual, expected)


def smooth_spectrum_reproduces_grey_means():
    temperature = 7.3
    hnu = np.logspace(-2, 4, 4001)
    u = hnu / temperature
    x = np.log(hnu)
    absorption = 2.0 + 0.3 * np.sin(1.7 * x) ** 2 + 0.01 * hnu ** 0.2
    scattering = 0.2 + 0.04 * np.cos(0.8 * x) ** 2
    absorption[(hnu > 500.0) & (hnu < 700.0)] = 0.0
    scattering[(hnu > 50.0) & (hnu < 80.0)] = 0.0
    bb, bf, ff = 0.5 * absorption, 0.3 * absorption, 0.2 * absorption
    targets = grey_targets(u, bb, bf, ff, scattering)

    for n_groups in (1024, 1):
        edges = np.logspace(np.log10(hnu[0]), np.log10(hnu[-1]), n_groups + 1)
        collapse = mod.compute_group_collapse(hnu, u, bb, bf, ff, scattering, edges, targets)
        result = regroup(collapse, edges, temperature)
        for field in ('kplanck', 'krosseland', 'kplanck_scattering'):
            assert_close(result[field], targets[field], 5e-13, (n_groups, field))
        # The flux-weighted absorption is defined by the groups; on a smooth spectrum it
        # agrees with the Newton-Cotes value to quadrature accuracy.
        assert_close(result['krosseland_absorption'], targets['krosseland_absorption'], 1e-4, (n_groups, 'kRa'))
        if n_groups == 1024:
            assert np.count_nonzero(collapse['kplanck'] == 0.0) > 0
            assert np.count_nonzero(collapse['kplanck_scattering'] == 0.0) > 0
            assert np.all(collapse['krosseland'] > 0.0)
            assert result['kplanck_scattering'] > 0.0


def constant_opacity_is_constant_in_every_group():
    # Includes the Wien tail (u = 800..1000), where exp(-u) must cancel within each group.
    for temperature, hnu in ((1.0, np.linspace(800.0, 1000.0, 101)), (3.0, np.logspace(-2, 4, 3001))):
        u = hnu / temperature
        a = np.full_like(hnu, 2.0)
        s = np.full_like(hnu, 0.5)
        zeros = np.zeros_like(hnu)
        targets = grey_targets(u, a, zeros, zeros, s)
        edges = np.logspace(np.log10(hnu[0]), np.log10(hnu[-1]), 257)
        collapse = mod.compute_group_collapse(hnu, u, a, zeros, zeros, s, edges, targets)
        np.testing.assert_allclose(collapse['kplanck'], 2.0, rtol=1e-12)
        np.testing.assert_allclose(collapse['kplanck_scattering'], 0.5, rtol=1e-12)
        np.testing.assert_allclose(collapse['krosseland'], 2.5, rtol=1e-12)
        np.testing.assert_allclose(collapse['krosseland_absorption'], 2.0, rtol=1e-12)


def spike_spectrum():
    temperature = 50.0
    hnu = np.linspace(0.01, 2000.0, 8001)
    u = hnu / temperature
    ff = 1.0e-3 * (hnu / 50.0) ** -2
    sc = np.full_like(hnu, 0.35)
    bb = np.zeros_like(hnu)
    rng = np.random.default_rng(3)
    spikes = rng.choice(np.arange(10, hnu.size - 10), 40, replace=False)
    bb[spikes] = 10.0 ** rng.uniform(1.0, 4.0, spikes.size)
    return temperature, hnu, u, bb, np.zeros_like(hnu), ff, sc


def grid_line_means(temperature, hnu, u, bb, ff, edges):
    quadrature = mod.NativeGroupQuadrature(u, edges / temperature)
    logp, _ = mod._gl_group_log_weights(edges, temperature)
    lines = mod._planck_regroup(quadrature.planck(quadrature.interpolate(bb)), logp)
    continuum = mod._planck_regroup(quadrature.planck(quadrature.interpolate(ff)), logp)
    return quadrature, lines, continuum


def isolated_spikes_are_kept():
    temperature, hnu, u, bb, bf, ff, sc = spike_spectrum()
    edges = np.logspace(np.log10(hnu[0]), np.log10(hnu[-1]), 1025)
    _quadrature, lines, continuum = grid_line_means(temperature, hnu, u, bb, ff, edges)
    weight = mod.planck_weight_u(u)
    trapezoid = np.trapz(bb * weight, u) / np.trapz(weight, u)
    assert_close(lines, trapezoid, 2e-3, 'spike Planck integral')

    targets = grey_targets(u, bb, bf, ff, sc)
    targets['kplanck'] = continuum + lines
    result = mod.compute_group_collapse_v8(hnu, u, bb, bf, ff, sc, edges, targets, kplanck_lines=lines)
    assert_close(result.diagnostics['planck_line_scale'], 1.0, 1e-12, 'line scale')
    assert not np.any(result.planck_flags & (mod.FLAG_LINES_ALIASED | mod.FLAG_LINES_UNRESOLVED))
    assert_close(regroup(result.fields, edges, temperature)['kplanck'], targets['kplanck'], 5e-13, 'kP')


def line_scale_moves_only_line_groups():
    temperature, hnu, u, bb, bf, ff, sc = spike_spectrum()
    edges = np.logspace(np.log10(hnu[0]), np.log10(hnu[-1]), 1025)
    quadrature, lines, continuum = grid_line_means(temperature, hnu, u, bb, ff, edges)
    targets = grey_targets(u, bb, bf, ff, sc)
    base = mod.compute_group_collapse_v8(
        hnu, u, bb, bf, ff, sc, edges, dict(targets, kplanck=continuum + lines), kplanck_lines=lines)
    tripled_targets = dict(targets, kplanck=continuum + 3.0 * lines)
    tripled = mod.compute_group_collapse_v8(hnu, u, bb, bf, ff, sc, edges, tripled_targets, kplanck_lines=3.0 * lines)
    assert_close(tripled.diagnostics['planck_line_scale'], 3.0, 1e-12, 'line scale 3')
    assert_close(regroup(tripled.fields, edges, temperature)['kplanck'], tripled_targets['kplanck'], 5e-13, 'kP x3')
    line_free = quadrature.planck(quadrature.interpolate(bb)) == 0.0
    np.testing.assert_allclose(tripled.fields['kplanck'][line_free], base.fields['kplanck'][line_free], rtol=1e-10)
    flagged = (tripled.planck_flags & mod.FLAG_LINES_ALIASED_X2) > 0
    assert np.any(flagged) and not np.any(flagged & line_free)

    unresolved_targets = dict(targets, kplanck=continuum + 50.0 * lines)
    unresolved = mod.compute_group_collapse_v8(
        hnu, u, bb, bf, ff, sc, edges, unresolved_targets, kplanck_lines=50.0 * lines)
    assert_close(unresolved.diagnostics['planck_lines_unresolved'], 50.0 * lines, 1e-12, 'unresolved lines')
    assert np.all(unresolved.planck_flags & mod.FLAG_LINES_UNRESOLVED)
    np.testing.assert_allclose(unresolved.fields['kplanck'], quadrature.planck(quadrature.interpolate(ff)), rtol=1e-9)


def point_line_rosseland_is_matched_by_line_scale():
    temperature, hnu, u, bb, bf, ff, sc = spike_spectrum()
    bb = np.convolve(bb, np.ones(5) / 5.0, mode='same')  # bin-averaged lines
    edges = np.logspace(np.log10(hnu[0]), np.log10(hnu[-1]), 1025)
    _quadrature, lines, continuum = grid_line_means(temperature, hnu, u, bb, ff, edges)
    targets = dict(grey_targets(u, bb, bf, ff, sc), kplanck=continuum + lines)
    without = mod.rosseland_mean(u, None, ff + sc)
    targets['krosseland'] = without + 0.3 * (targets['krosseland'] - without)
    result = mod.compute_group_collapse_v8(
        hnu, u, bb, bf, ff, sc, edges, targets, kplanck_lines=lines, rosseland_point_lines=True)
    scale = result.diagnostics['rosseland_line_scale']
    assert 0.0 < scale < 1.0, scale
    assert_close(regroup(result.fields, edges, temperature)['krosseland'], targets['krosseland'], 5e-13, 'kR')


def star_line_arrays_are_used():
    temperature, hnu, u, bb, bf, ff, sc = spike_spectrum()
    edges = np.logspace(np.log10(hnu[0]), np.log10(hnu[-1]), 1025)
    quadrature, lines, continuum = grid_line_means(temperature, hnu, u, bb, ff, edges)
    aliased = 0.02 * bb  # the grid caught 2% of each line; bb_planck holds the full strength
    targets = dict(grey_targets(u, bb, bf, ff, sc), kplanck=continuum + lines)
    result = mod.compute_group_collapse_v8(
        hnu, u, aliased, bf, ff, sc, edges, targets, kplanck_lines=lines, planck_line_array=bb)
    assert_close(result.diagnostics['planck_line_scale'], 1.0, 1e-12, 'line scale with bb_planck')
    assert not np.any(result.planck_flags & (mod.FLAG_LINES_ALIASED | mod.FLAG_LINES_UNRESOLVED))
    np.testing.assert_allclose(result.fields['kplanck'], quadrature.planck(quadrature.interpolate(bb + ff)), rtol=1e-10)

    point = 0.5 * bb  # STAR's Rosseland lines, stored as bb_ros
    with_point = mod.rosseland_mean(u, None, point + ff + sc)
    targets = dict(targets, krosseland=with_point)
    result = mod.compute_group_collapse_v8(
        hnu, u, bb, bf, ff, sc, edges, targets, kplanck_lines=lines,
        rosseland_point_lines=True, rosseland_line_array=point)
    assert result.diagnostics['rosseland_line_scale'] == 1.0
    assert not np.any(result.rosseland_flags & mod.FLAG_ROSSELAND_LINES_RESCALED)
    assert abs(math.log(result.diagnostics['rosseland_factor'])) < math.log(1.02)


def main():
    smooth_spectrum_reproduces_grey_means()
    constant_opacity_is_constant_in_every_group()
    isolated_spikes_are_kept()
    line_scale_moves_only_line_groups()
    point_line_rosseland_is_matched_by_line_scale()
    star_line_arrays_are_used()
    print('Conservative group collapse tests passed.')


if __name__ == '__main__':
    main()
