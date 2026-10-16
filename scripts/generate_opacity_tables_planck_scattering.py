#!/usr/bin/env python3
"""Build opacity tables from a STAR FAC grid runs directory.

Reads ``starout.h5`` files produced by ``run_tops_star_fac_grid.py`` and
writes long-format and pivot tables for:

  - Planck scattering opacity    (Planck arithmetic mean of ``sc``)
  - Planck opacity                (``kplnk``, absorption only)
  - Rosseland opacity             (``kros``, total)
  - Rosseland flux-weighted absorption opacity

It also writes frequency-group collapse tables: for each quantity above,
the spectrum is binned into log-spaced photon-energy groups (default 1024)
from the minimum to maximum ``hnu`` in the data, and the corresponding
group mean opacity is stored on the same (rho, temp) grid.

The total Rosseland mean uses

    1 / kappa_R = integral( kappa_nu^-1 * dB/dT dnu ) / integral( dB/dT dnu )

The absorption and scattering Planck means use

    kappa_P = integral( kappa_nu * B_nu dnu ) / integral( B_nu dnu )

The Rosseland flux-weighted absorption opacity is

    kappa_Ra = integral( (kappa_a / chi) * dB/dT dnu )
               / integral( (1 / chi) * dB/dT dnu )

with ``kappa_a = bb + bf + ff``, ``chi = kappa_a + sc``, and the Rosseland
weight function ``w = dB/dT`` matching STAR's ``RosselandWeight()`` implementation.
This is a ``w/chi``-weighted average of ``kappa_a`` and has units cm^2/g.

All group-weight ratios are evaluated after subtracting the largest logarithmic
weight in the bin.  Thus factors such as ``exp(-800)`` cancel between numerator
and denominator instead of underflowing independently.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("PYTHONNOUSERSITE", "1")

import numpy as np
import scipy.optimize

try:
    import h5py
except ImportError as exc:
    raise SystemExit("h5py is required: {}".format(exc)) from exc

PI = math.pi
ROSS_C = 15.0 / (4.0 * PI**4)
CHI_MIN = 1e-200
DEFAULT_MODE = "fac-opacity-cowan-state"
DEFAULT_FREQ_GROUPS = 1024
LABEL_RE = re.compile(r"^T([\dp]+)_rho([\dpemp]+)$")
COLLAPSE_FIELDS = (
    ("kplanck_scattering", "scattering_planck"),
    ("kplanck", "planck"),
    ("krosseland", "rosseland"),
    ("krosseland_absorption", "rosseland_absorption"),
)
MAX_VALIDATION_EXAMPLES = 5
EMPTY_BIN_QUADRATURE_ORDER = 32
EMPTY_BIN_GL_X, EMPTY_BIN_GL_W = np.polynomial.legendre.leggauss(EMPTY_BIN_QUADRATURE_ORDER)
SUBINTERVAL_GL_ORDER = 4
SUBINTERVAL_GL_X, SUBINTERVAL_GL_W = np.polynomial.legendre.leggauss(SUBINTERVAL_GL_ORDER)
# Group means may be moved onto STAR's grey means only by quadrature-sized factors: up to 2%
# silently, up to 30% with the grey_rescaled flag (STAR's Newton-Cotes rule and the exact group
# integrals differ by up to 26% where a few point-sampled bound-free edges carry the continuum), and never beyond.
RESIDUAL_LOG_TOLERANCE = math.log(1.02)
RESIDUAL_LOG_LIMIT = math.log(1.30)
# |ln s_P| thresholds for the line scale s_P = STAR line Planck mean / grid line Planck mean.
LINE_ALIASED_LOG = 0.1
LINE_ALIASED_X2_LOG = math.log(2.0)
LINE_UNRESOLVED_LOG = math.log(10.0)
GROUP_LINE_FRACTION_FLAG = 0.2
# Unresolved line content is flagged only when it exceeds this fraction of the Planck mean.
LINE_UNRESOLVED_FLAG_FRACTION = 0.01
ROSSELAND_RESCALE_FLAG_LOG = 0.1
# Per-group quality bits published with the group tables.
FLAG_LINES_ALIASED = 1
FLAG_LINES_ALIASED_X2 = 2
FLAG_LINES_UNRESOLVED = 4
FLAG_ROSSELAND_LINES_RESCALED = 8
FLAG_SUB_CELL = 16
FLAG_GREY_RESCALED = 32


class NonFiniteOutputError(Exception):
    """Raised when built opacity tables contain NaN or Inf."""


def parse_label(dirname: str) -> tuple[float, float] | None:
    match = LABEL_RE.match(dirname)
    if not match:
        return None
    temp_text = match.group(1).replace("p", ".")
    rho_text = (
        match.group(2)
        .replace("p", ".")
        .replace("em", "e-")
        .replace("ep", "e+")
    )
    try:
        return float(temp_text), float(rho_text)
    except ValueError:
        return None


def resolve_runs_root(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_dir() and (path / "runs").is_dir():
        return path / "runs"
    return path


def resolve_outdir(runs_root: Path, outdir: Path | None) -> Path:
    if outdir is not None:
        return outdir.expanduser().resolve()
    parent = runs_root.parent
    if parent.name != "runs":
        return parent / "opacity_tables"
    return parent / "opacity_tables"


def load_manifest_axes(runs_root: Path) -> tuple[list[float], list[float]] | None:
    manifest_path = runs_root.parent / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    custom = manifest.get("custom_grid")
    if isinstance(custom, dict):
        temp_count = int(custom.get("temp_count", 0) or 0)
        rho_count = int(custom.get("rho_count", 0) or 0)
        if temp_count > 1 and rho_count > 1:
            temps = np.logspace(
                math.log10(float(custom["temp_min_ev"])),
                math.log10(float(custom["temp_max_ev"])),
                temp_count,
            )
            rhos = np.logspace(
                math.log10(float(custom["rho_min"])),
                math.log10(float(custom["rho_max"])),
                rho_count,
            )
            return temps.tolist(), rhos.tolist()
    return None


def differential(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    n = p.size
    dp = np.zeros(n, dtype=np.float64)
    if n == 0:
        return dp
    if n < 5:
        dp[0] = p[0]
        if n > 1:
            dp[1:] = np.diff(p)
        return dp

    c = 1.0 / 24.0
    dp[0] = (
        -50.0 * p[0]
        + 96.0 * p[1]
        - 72.0 * p[2]
        + 32.0 * p[3]
        - 6.0 * p[4]
    ) * c
    dp[1] = (
        -6.0 * p[0]
        - 20.0 * p[1]
        + 36.0 * p[2]
        - 12.0 * p[3]
        + 2.0 * p[4]
    ) * c
    dp[-1] = (
        50.0 * p[-1]
        - 96.0 * p[-2]
        + 72.0 * p[-3]
        - 32.0 * p[-4]
        + 6.0 * p[-5]
    ) * c
    dp[-2] = (
        6.0 * p[-1]
        + 20.0 * p[-2]
        - 36.0 * p[-3]
        + 12.0 * p[-4]
        - 2.0 * p[-5]
    ) * c
    i = np.arange(2, n - 2)
    dp[i] = (
        2.0 * (p[i - 2] - p[i + 2]) + 16.0 * (p[i + 1] - p[i - 1])
    ) * c
    return dp


def newton_cotes_end(x: np.ndarray, i0: int, i1: int) -> float:
    r = np.zeros(i1 + 1, dtype=np.float64)
    r[i1] = x[i0]
    a = 0.0
    for i in range(i0 + 1, i1, 2):
        a += x[i]
    r[i1] += 4.0 * a
    a = 0.0
    k = i1 - 1
    for i in range(i0 + 2, k, 2):
        a += x[i]
    r[i1] += 2.0 * a
    if i == i1:
        r[i1] += x[i1]
        r[i1] /= 3.0
    else:
        r[i1] += x[k]
        r[i1] /= 3.0
        r[i1] += 0.5 * (x[k] + x[i1])
    r[i1] += r[i0]
    return float(r[i1])


def integrate_nc(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size == 0:
        return 0.0
    dx = differential(x)
    integrand = y * dx
    return newton_cotes_end(integrand, 0, x.size - 1)


def _log_expm1_positive(u: np.ndarray) -> np.ndarray:
    """Return log(expm1(u)) without overflow for positive u."""
    u = np.asarray(u, dtype=np.float64)
    out = np.full_like(u, np.nan)
    good = np.isfinite(u) & (u > 0.0)
    moderate = good & (u < 50.0)
    large = good & ~moderate
    out[moderate] = np.log(np.expm1(u[moderate]))
    # expm1(u) = exp(u) * (1 - exp(-u)).  This remains finite for u >> 700.
    out[large] = u[large] + np.log1p(-np.exp(-u[large]))
    return out


def planck_log_weight_u(hnux: np.ndarray) -> np.ndarray:
    """Log of the unnormalised Planck weight x^3/(exp(x)-1)."""
    u = np.asarray(hnux, dtype=np.float64)
    out = np.full_like(u, -np.inf)
    good = np.isfinite(u) & (u > 0.0)
    out[good] = 3.0 * np.log(u[good]) - _log_expm1_positive(u[good])
    return out


def rosseland_log_weight_u(hnux: np.ndarray) -> np.ndarray:
    """Log of STAR's normalised Rosseland weight.

    The expression is evaluated without forming exp(-u), so values such as
    u=800 remain representable in logarithmic form.  The normalisation constant
    cancels in every ratio, but retaining it makes this the exact log of
    rosseland_weight where the latter is representable.
    """
    u = np.asarray(hnux, dtype=np.float64)
    out = np.full_like(u, -np.inf)
    good = np.isfinite(u) & (u > 0.0)
    one_minus_exp_minus_u = -np.expm1(-u[good])
    out[good] = (
        math.log(ROSS_C)
        - u[good]
        + 4.0 * np.log(u[good])
        - 2.0 * np.log(one_minus_exp_minus_u)
    )
    return out


def _scaled_weight_from_log(log_weight: np.ndarray) -> np.ndarray:
    """Scale positive weights by their largest logarithm.

    Multiplying every weight by the same factor leaves arithmetic, harmonic,
    and flux-weighted ratios unchanged.  This is the cancellation that avoids
    exp(-800) underflow.
    """
    log_weight = np.asarray(log_weight, dtype=np.float64)
    scaled = np.zeros_like(log_weight)
    finite = np.isfinite(log_weight)
    if not np.any(finite):
        return scaled
    maximum = float(np.max(log_weight[finite]))
    scaled[finite] = np.exp(log_weight[finite] - maximum)
    return scaled


def _sorted_finite_samples(
    u: np.ndarray,
    *arrays: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Return samples sorted by u, dropping non-finite coordinates."""
    u = np.asarray(u, dtype=np.float64)
    converted = [np.asarray(array, dtype=np.float64) for array in arrays]
    if any(array.shape != u.shape for array in converted):
        raise ValueError("weighted-mean arrays must have identical shapes")
    good = np.isfinite(u)
    if not np.any(good):
        return (u[:0],) + tuple(array[:0] for array in converted)
    order = np.argsort(u[good], kind="mergesort")
    sorted_u = u[good][order]
    sorted_arrays = [array[good][order] for array in converted]
    # Duplicate coordinates contribute zero interval in trapezoidal integration.
    # Keep the first occurrence so all remaining intervals are strictly positive.
    if sorted_u.size > 1:
        unique = np.concatenate(([True], np.diff(sorted_u) > 0.0))
        sorted_u = sorted_u[unique]
        sorted_arrays = [array[unique] for array in sorted_arrays]
    return (sorted_u,) + tuple(sorted_arrays)


def _stable_trapz_ratio(
    u: np.ndarray,
    log_weight: np.ndarray,
    numerator_factor: np.ndarray,
    denominator_factor: np.ndarray,
) -> float:
    """Evaluate integral(W*A du) / integral(W*B du) after weight scaling."""
    u, log_weight, numerator_factor, denominator_factor = _sorted_finite_samples(
        u, log_weight, numerator_factor, denominator_factor
    )
    if u.size == 0:
        return math.nan
    if u.size == 1:
        denominator = float(denominator_factor[0])
        if not math.isfinite(denominator) or denominator == 0.0:
            return math.nan
        return float(numerator_factor[0] / denominator)

    scaled_weight = _scaled_weight_from_log(log_weight)
    numerator = float(np.trapz(scaled_weight * numerator_factor, u))
    denominator = float(np.trapz(scaled_weight * denominator_factor, u))
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return math.nan
    if denominator <= 0.0:
        return math.nan
    result = numerator / denominator
    return float(result) if math.isfinite(result) else math.nan



def _stable_nc_ratio(
    u: np.ndarray,
    log_weight: np.ndarray,
    numerator_factor: np.ndarray,
    denominator_factor: np.ndarray,
) -> float:
    """Newton-Cotes counterpart of _stable_trapz_ratio for scalar tables."""
    u, log_weight, numerator_factor, denominator_factor = _sorted_finite_samples(
        u, log_weight, numerator_factor, denominator_factor
    )
    if u.size == 0:
        return math.nan
    if u.size == 1:
        denominator = float(denominator_factor[0])
        if not math.isfinite(denominator) or denominator == 0.0:
            return math.nan
        return float(numerator_factor[0] / denominator)
    scaled_weight = _scaled_weight_from_log(log_weight)
    numerator = integrate_nc(u, scaled_weight * numerator_factor)
    denominator = integrate_nc(u, scaled_weight * denominator_factor)
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return math.nan
    if denominator <= 0.0:
        return math.nan
    result = numerator / denominator
    return float(result) if math.isfinite(result) else math.nan

def rosseland_weight(hnux: np.ndarray) -> np.ndarray:
    """Return STAR's Rosseland weight where it is representable.

    Collapse means do not use this direct array anymore; they use
    rosseland_log_weight_u and scale the logarithmic weights before integration.
    This function is retained for diagnostics and backward compatibility.
    """
    log_weight = rosseland_log_weight_u(hnux)
    out = np.zeros_like(log_weight)
    representable = np.isfinite(log_weight) & (log_weight >= math.log(np.nextafter(0.0, 1.0)))
    out[representable] = np.exp(log_weight[representable])
    return out

def planck_mean(hnux: np.ndarray, spec: np.ndarray) -> float:
    """Stable Planck-weighted arithmetic mean over the supplied spectrum.

    Computes integral(W_P * kappa du) / integral(W_P du) after removing a
    common logarithmic scale from W_P.  This is used for both absorption and
    scattering grey targets.
    """
    hnux = np.asarray(hnux, dtype=np.float64)
    spec = np.asarray(spec, dtype=np.float64)
    good = np.isfinite(hnux) & (hnux > 0.0) & np.isfinite(spec) & (spec >= 0.0)
    if np.count_nonzero(good) == 0:
        return math.nan
    if np.all(spec[good] == 0.0):
        return 0.0
    return _stable_nc_ratio(
        hnux[good],
        planck_log_weight_u(hnux[good]),
        spec[good],
        np.ones(np.count_nonzero(good), dtype=np.float64),
    )

def rosseland_mean(hnux: np.ndarray, rosw: np.ndarray, spec: np.ndarray) -> float:
    """Stable harmonic Rosseland mean over the supplied spectrum.

    The documented definition is integral(W du) / integral(W/kappa du).
    A common logarithmic scale factor is removed from W before both integrals,
    so exponentially small weights cancel instead of underflowing separately.
    """
    del rosw  # Kept in the signature for compatibility with existing callers.
    hnux = np.asarray(hnux, dtype=np.float64)
    spec = np.asarray(spec, dtype=np.float64)
    good = np.isfinite(hnux) & np.isfinite(spec) & (spec >= 0.0)
    if np.count_nonzero(good) == 0:
        return math.nan
    if np.any(spec[good] == 0.0):
        return 0.0
    result = _stable_nc_ratio(
        hnux[good],
        rosseland_log_weight_u(hnux[good]),
        np.ones(np.count_nonzero(good), dtype=np.float64),
        1.0 / spec[good],
    )
    return result

def rosseland_flux_weighted_absorption(
    hnux: np.ndarray,
    rosw: np.ndarray,
    kappa_a: np.ndarray,
    chi: np.ndarray,
) -> float:
    """Stable flux-weighted absorption opacity.

    Computes integral(W*kappa_a/chi du) / integral(W/chi du) after removing
    a common logarithmic scale from W.  Therefore an exponentially small W
    cannot by itself produce zero.
    """
    del rosw  # Kept in the signature for compatibility with existing callers.
    hnux = np.asarray(hnux, dtype=np.float64)
    kappa_a = np.asarray(kappa_a, dtype=np.float64)
    chi = np.asarray(chi, dtype=np.float64)
    good = (
        np.isfinite(hnux)
        & np.isfinite(kappa_a)
        & np.isfinite(chi)
        & (kappa_a >= 0.0)
        & (chi >= 0.0)
    )
    if np.count_nonzero(good) == 0:
        return math.nan
    if np.all(kappa_a[good] == 0.0):
        return 0.0
    chi_safe = np.maximum(chi[good], CHI_MIN)
    result = _stable_nc_ratio(
        hnux[good],
        rosseland_log_weight_u(hnux[good]),
        kappa_a[good] / chi_safe,
        1.0 / chi_safe,
    )
    return result

def planck_weight_u(hnux: np.ndarray) -> np.ndarray:
    """Return the Planck weight where it is representable.

    Collapse ratios use planck_log_weight_u instead, so this direct form is not
    allowed to decide that a group has zero weight merely because u is large.
    """
    log_weight = planck_log_weight_u(hnux)
    out = np.zeros_like(log_weight)
    representable = np.isfinite(log_weight) & (log_weight >= math.log(np.nextafter(0.0, 1.0)))
    out[representable] = np.exp(log_weight[representable])
    return out

def log_frequency_edges(hnu_min: float, hnu_max: float, n_groups: int) -> np.ndarray:
    if not math.isfinite(hnu_min) or not math.isfinite(hnu_max):
        raise ValueError("hnu limits must be finite")
    if hnu_min <= 0.0 or hnu_max <= hnu_min:
        raise ValueError("require 0 < hnu_min < hnu_max")
    if n_groups < 1:
        raise ValueError("n_groups must be positive")
    return np.logspace(math.log10(hnu_min), math.log10(hnu_max), n_groups + 1)


def log_frequency_centers(edges: np.ndarray) -> np.ndarray:
    return np.sqrt(edges[:-1] * edges[1:])


def planck_mean_bin(u: np.ndarray, spec: np.ndarray, mask: np.ndarray) -> float:
    ub = np.asarray(u[mask], dtype=np.float64)
    sb = np.asarray(spec[mask], dtype=np.float64)
    good = np.isfinite(ub) & np.isfinite(sb) & (sb >= 0.0)
    ub = ub[good]
    sb = sb[good]
    if ub.size == 0:
        return math.nan
    if ub.size == 1:
        return float(sb[0])
    if np.all(sb == 0.0):
        return 0.0
    return _stable_trapz_ratio(
        ub,
        planck_log_weight_u(ub),
        sb,
        np.ones_like(sb),
    )

def rosseland_mean_bin(
    u: np.ndarray,
    rosw: np.ndarray,
    spec: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Stable group harmonic Rosseland mean.

    This fixes the old group implementation, which evaluated only
    1/integral(W/kappa du).  The documented group mean requires
    integral(W du)/integral(W/kappa du); the numerator is essential both
    physically and for cancellation of exponentially small weights.
    """
    del rosw  # Kept in the signature for compatibility with existing callers.
    ub = np.asarray(u[mask], dtype=np.float64)
    sb = np.asarray(spec[mask], dtype=np.float64)
    good = np.isfinite(ub) & np.isfinite(sb) & (sb >= 0.0)
    ub = ub[good]
    sb = sb[good]
    if ub.size == 0:
        return math.nan
    if ub.size == 1:
        return float(sb[0])
    if np.any(sb == 0.0):
        return 0.0
    return _stable_trapz_ratio(
        ub,
        rosseland_log_weight_u(ub),
        np.ones_like(sb),
        1.0 / sb,
    )

def rosseland_absorption_bin(
    u: np.ndarray,
    rosw: np.ndarray,
    kappa_a: np.ndarray,
    chi: np.ndarray,
    mask: np.ndarray,
) -> float:
    del rosw  # Kept in the signature for compatibility with existing callers.
    ub = np.asarray(u[mask], dtype=np.float64)
    kappa_bin = np.asarray(kappa_a[mask], dtype=np.float64)
    chi_bin = np.asarray(chi[mask], dtype=np.float64)
    good = (
        np.isfinite(ub)
        & np.isfinite(kappa_bin)
        & np.isfinite(chi_bin)
        & (kappa_bin >= 0.0)
        & (chi_bin >= 0.0)
    )
    ub = ub[good]
    kappa_bin = kappa_bin[good]
    chi_bin = chi_bin[good]
    if ub.size == 0:
        return math.nan
    if ub.size == 1:
        return float(kappa_bin[0])
    if np.all(kappa_bin == 0.0):
        return 0.0
    chi_safe = np.maximum(chi_bin, CHI_MIN)
    return _stable_trapz_ratio(
        ub,
        rosseland_log_weight_u(ub),
        kappa_bin / chi_safe,
        1.0 / chi_safe,
    )

def _logsumexp_1d(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if not np.any(finite):
        return -math.inf
    maximum = float(np.max(values[finite]))
    return maximum + math.log(float(np.sum(np.exp(values[finite] - maximum))))


def _infer_temperature_from_hnu_u(
    hnu_ev: np.ndarray,
    u: np.ndarray,
) -> float:
    good = (
        np.isfinite(hnu_ev)
        & (hnu_ev > 0.0)
        & np.isfinite(u)
        & (u > 0.0)
    )
    if np.count_nonzero(good) < 2:
        raise ValueError("cannot infer temperature from fewer than two hnu/u samples")
    samples = np.asarray(hnu_ev[good] / u[good], dtype=np.float64)
    temperature = float(np.median(samples))
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("could not infer a finite positive temperature from hnu/u")
    relative_spread = float(np.max(np.abs(samples / temperature - 1.0)))
    if relative_spread > 1.0e-8:
        raise ValueError(
            "raw u is not proportional to photon energy at fixed temperature; "
            f"maximum relative hnu/u spread is {relative_spread:.3e}"
        )
    return temperature


def _scaled_group_weights(log_weight: np.ndarray) -> np.ndarray:
    log_weight = np.asarray(log_weight, dtype=np.float64)
    finite = np.isfinite(log_weight)
    if not np.any(finite):
        raise ValueError("group weight integral is non-finite in every group")
    maximum = float(np.max(log_weight[finite]))
    weight = np.zeros_like(log_weight)
    weight[finite] = np.exp(log_weight[finite] - maximum)
    return weight


def _regroup_collapse_fields(
    collapse: dict[str, np.ndarray],
    log_planck_group_weight: np.ndarray,
    log_rosseland_group_weight: np.ndarray,
) -> dict[str, float]:
    planck_weight = _scaled_group_weights(log_planck_group_weight)
    rosseland_weight_group = _scaled_group_weights(log_rosseland_group_weight)

    kplanck = np.asarray(collapse["kplanck"], dtype=np.float64)
    krosseland = np.asarray(collapse["krosseland"], dtype=np.float64)
    kplanck_scattering = np.asarray(collapse["kplanck_scattering"], dtype=np.float64)
    kross_absorption = np.asarray(
        collapse["krosseland_absorption"], dtype=np.float64
    )

    planck = float(np.sum(planck_weight * kplanck) / np.sum(planck_weight))

    def harmonic(values: np.ndarray) -> float:
        if np.any(values == 0.0):
            return 0.0
        return float(
            np.sum(rosseland_weight_group)
            / np.sum(rosseland_weight_group / values)
        )

    total = harmonic(krosseland)
    scattering = float(
        np.sum(planck_weight * kplanck_scattering) / np.sum(planck_weight)
    )

    if np.any(krosseland <= 0.0):
        raise ValueError(
            "non-positive total Rosseland group prevents absorption regrouping"
        )
    transport_weight = rosseland_weight_group / krosseland
    absorption = float(
        np.sum(transport_weight * kross_absorption)
        / np.sum(transport_weight)
    )

    return {
        "kplanck": planck,
        "krosseland": total,
        "kplanck_scattering": scattering,
        "krosseland_absorption": absorption,
    }


def _sorted_unique_spectrum(
    hnu_ev: np.ndarray,
    u: np.ndarray,
    arrays: dict[str, np.ndarray],
    hnu_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], float]:
    """Return the valid raw samples sorted by energy, plus the temperature."""
    hnu_ev = np.asarray(hnu_ev, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    arrays = {name: np.asarray(values, dtype=np.float64) for name, values in arrays.items()}
    common = np.isfinite(hnu_ev) & (hnu_ev > 0.0) & np.isfinite(u) & (u > 0.0)
    for values in arrays.values():
        common &= np.isfinite(values) & (values >= 0.0)
    if np.count_nonzero(common) < 2:
        raise ValueError("fewer than two valid raw spectral samples")

    order = np.argsort(hnu_ev[common], kind="mergesort")
    hnu_raw = hnu_ev[common][order]
    u_raw = u[common][order]
    sorted_arrays = {name: values[common][order] for name, values in arrays.items()}
    unique = np.concatenate(([True], np.diff(hnu_raw) > 0.0))
    hnu_raw = hnu_raw[unique]
    u_raw = u_raw[unique]
    sorted_arrays = {name: values[unique] for name, values in sorted_arrays.items()}
    if hnu_raw.size < 2:
        raise ValueError("fewer than two distinct raw photon energies")

    tolerance = 64.0 * np.finfo(np.float64).eps
    if hnu_edges[0] < hnu_raw[0] * (1.0 - tolerance):
        raise ValueError("collapse lower edge lies below raw photon-energy range")
    if hnu_edges[-1] > hnu_raw[-1] * (1.0 + tolerance):
        raise ValueError("collapse upper edge lies above raw photon-energy range")
    temperature = _infer_temperature_from_hnu_u(hnu_raw, u_raw)
    return hnu_raw, u_raw, sorted_arrays, temperature


def _gl_group_log_weights(
    hnu_edges: np.ndarray,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return log integral_g W_P du and log integral_g W_R du for every group.

    These are the group weights the browser uses to re-collapse groups
    (opacity_math.mjs), evaluated with the same 32-point Gauss-Legendre rule.
    """
    hnu_edges = np.asarray(hnu_edges, dtype=np.float64)
    low_u = hnu_edges[:-1] / temperature
    high_u = hnu_edges[1:] / temperature
    midpoint_u = 0.5 * (low_u + high_u)
    half_width_u = 0.5 * (high_u - low_u)
    if np.any(~np.isfinite(half_width_u)) or np.any(half_width_u <= 0.0):
        raise ValueError("collapse group edges are not strictly increasing")
    u_nodes = midpoint_u[:, None] + half_width_u[:, None] * EMPTY_BIN_GL_X[None, :]
    log_gl_weight = np.log(EMPTY_BIN_GL_W)[None, :]
    log_planck = planck_log_weight_u(u_nodes) + log_gl_weight
    log_rosseland = rosseland_log_weight_u(u_nodes) + log_gl_weight
    log_planck_group_weight = np.log(half_width_u) + _row_logsumexp(log_planck)
    log_rosseland_group_weight = np.log(half_width_u) + _row_logsumexp(log_rosseland)
    return log_planck_group_weight, log_rosseland_group_weight


def _row_logsumexp(values: np.ndarray) -> np.ndarray:
    maximum = np.max(values, axis=1)
    return maximum + np.log(np.sum(np.exp(values - maximum[:, None]), axis=1))


class NativeGroupQuadrature:
    """Exact group integrals of the native spectrum.

    The raw samples define a piecewise-linear opacity.  Every group interval is
    cut at the raw photon energies it contains, and each sub-interval is
    integrated with SUBINTERVAL_GL_ORDER Gauss-Legendre nodes against the exact
    Planck or Rosseland weight.  Because the opacity is linear on every
    sub-interval, the group integrals partition the integral of the native
    spectrum: no raw sample is skipped, however narrow a line is, and a group
    narrower than the native spacing gets the interpolated value.  Weights are
    scaled by their largest logarithm within each group, so exp(-u) factors
    cancel between numerator and denominator instead of underflowing.
    """

    def __init__(self, u_raw: np.ndarray, edges_u: np.ndarray) -> None:
        self.u_raw = np.asarray(u_raw, dtype=np.float64)
        edges_u = np.asarray(edges_u, dtype=np.float64)
        self.n_groups = edges_u.size - 1
        inner = self.u_raw[(self.u_raw > edges_u[0]) & (self.u_raw < edges_u[-1])]
        points = np.union1d(edges_u, inner)
        left = points[:-1]
        right = points[1:]
        keep = right > left
        left = left[keep]
        right = right[keep]
        midpoint = 0.5 * (left + right)
        half_width = 0.5 * (right - left)
        group = np.clip(np.searchsorted(edges_u, midpoint, side="right") - 1, 0, self.n_groups - 1)
        order = SUBINTERVAL_GL_ORDER
        self.u_nodes = (midpoint[:, None] + half_width[:, None] * SUBINTERVAL_GL_X[None, :]).reshape(-1)
        log_quadrature = (np.log(half_width)[:, None] + np.log(SUBINTERVAL_GL_W)[None, :]).reshape(-1)
        self.node_group = np.repeat(group, order)
        self.starts = np.searchsorted(self.node_group, np.arange(self.n_groups), side="left")
        counts = np.diff(np.append(self.starts, self.node_group.size))
        if np.any(counts <= 0):
            raise ValueError("a collapse group received no quadrature nodes")
        self.log_planck = planck_log_weight_u(self.u_nodes) + log_quadrature
        self.log_rosseland = rosseland_log_weight_u(self.u_nodes) + log_quadrature
        self.planck_scaled = self._scaled(self.log_planck)
        self.log_rosseland_sum = self._log_sum(self.log_rosseland)
        native_counts = np.diff(np.searchsorted(self.u_raw, edges_u, side="left"))
        self.sub_cell = native_counts == 0

    def _group_max(self, log_values: np.ndarray) -> np.ndarray:
        return np.maximum.reduceat(log_values, self.starts)

    def _scaled(self, log_values: np.ndarray) -> np.ndarray:
        return np.exp(log_values - self._group_max(log_values)[self.node_group])

    def _sum(self, values: np.ndarray) -> np.ndarray:
        return np.add.reduceat(values, self.starts)

    def _log_sum(self, log_values: np.ndarray) -> np.ndarray:
        maximum = self._group_max(log_values)
        return maximum + np.log(self._sum(np.exp(log_values - maximum[self.node_group])))

    def interpolate(self, values: np.ndarray) -> np.ndarray:
        return np.interp(self.u_nodes, self.u_raw, np.asarray(values, dtype=np.float64))

    def planck(self, node_values: np.ndarray) -> np.ndarray:
        """Group Planck means of an opacity given at the quadrature nodes."""
        return self._sum(self.planck_scaled * node_values) / self._sum(self.planck_scaled)

    def rosseland(self, chi_nodes: np.ndarray, absorption_nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Group harmonic Rosseland means of chi and flux-weighted means of absorption."""
        log_transport = self.log_rosseland - np.log(np.maximum(chi_nodes, CHI_MIN))
        transport = self._scaled(log_transport)
        transport_sum = self._sum(transport)
        log_transport_sum = self._group_max(log_transport) + np.log(transport_sum)
        total = np.exp(self.log_rosseland_sum - log_transport_sum)
        absorption = self._sum(transport * absorption_nodes) / transport_sum
        return total, absorption


@dataclasses.dataclass
class GroupCollapse:
    """Group opacities of one (T, rho) cell plus the provenance of every adjustment."""

    fields: dict[str, np.ndarray]
    planck_flags: np.ndarray
    rosseland_flags: np.ndarray
    diagnostics: dict[str, float]


def _planck_regroup(values: np.ndarray, log_group_weight: np.ndarray) -> float:
    weight = _scaled_group_weights(log_group_weight)
    return float(np.sum(weight * values) / np.sum(weight))


def _rosseland_regroup(values: np.ndarray, log_group_weight: np.ndarray) -> float:
    weight = _scaled_group_weights(log_group_weight)
    if np.any(values <= 0.0):
        return 0.0
    return float(np.sum(weight) / np.sum(weight / values))


def _residual_factor(field: str, target: float, reconstructed: float) -> float:
    """Factor that moves a regrouped mean onto its target; only quadrature-sized factors are accepted."""
    if not math.isfinite(target) or target < 0.0:
        raise ValueError(f"invalid integrated target for {field}: {target}")
    if not math.isfinite(reconstructed) or reconstructed < 0.0:
        raise ValueError(f"invalid reconstructed mean for {field}: {reconstructed}")
    if target == 0.0 and reconstructed == 0.0:
        return 1.0
    if target == 0.0 or reconstructed == 0.0:
        raise ValueError(f"{field}: target {target:.6e} but groups reconstruct {reconstructed:.6e}")
    factor = target / reconstructed
    if abs(math.log(factor)) > RESIDUAL_LOG_LIMIT:
        raise ValueError(
            f"{field}: groups reconstruct {reconstructed:.10e} against target {target:.10e} "
            f"(factor {factor:.6f} exceeds the quadrature limit)"
        )
    return factor


def _beyond_tolerance(*factors: float) -> bool:
    return any(abs(math.log(factor)) > RESIDUAL_LOG_TOLERANCE for factor in factors)


def _planck_line_scale(analytic: float, grid: float) -> float:
    """STAR line Planck mean over the grid line Planck mean (0 or inf when one side is empty)."""
    if analytic > 0.0:
        return analytic / grid if grid > 0.0 else math.inf
    return 0.0 if grid > 0.0 else 1.0


def compute_group_collapse_v8(
    hnu_ev: np.ndarray,
    u: np.ndarray,
    bb: np.ndarray,
    bf: np.ndarray,
    ff: np.ndarray,
    sc: np.ndarray,
    hnu_edges: np.ndarray,
    target_means: dict[str, float],
    kplanck_lines: float | None = None,
    rosseland_point_lines: bool = False,
    planck_line_array: np.ndarray | None = None,
    rosseland_line_array: np.ndarray | None = None,
) -> GroupCollapse:
    """Collapse one STAR spectrum into groups that reproduce STAR's grey means.

    target_means holds the grey kplanck, krosseland, kplanck_scattering and
    krosseland_absorption the groups must reproduce.  kplanck_lines is STAR's
    analytic line Planck mean (kplnk_bb_analytical): STAR's kplanck sums every
    line strength rather than integrating the grid bb, so only the line part of
    the groups is rescaled to it; continuum groups are never rescaled by a line
    correction.  Where the grid holds less than a tenth or more than ten times
    the line strength, the line placement is unknown: the groups carry the
    continuum only and the line Planck mean is reported separately.
    rosseland_point_lines marks spectra whose STAR Rosseland mean used a
    point-sampled line array that starout.h5 does not store (dilute-LTE); the
    stored bin-averaged lines are then scaled so the groups reproduce STAR's
    Rosseland mean.  STAR files written with STAR_WRITE_BB_PLANCK=1 carry
    planck_line_array (mixspec/bb_planck, every line whole in its centre cell):
    the Planck line groups then come from it, so line placement is exact.  Files
    written with STAR_DILUTE_LTE_WRITE_BROS=1 carry rosseland_line_array
    (mixspec/bb_ros, the lines STAR's Rosseland mean used); their strength is
    fitted only when the groups miss STAR's Rosseland mean by more than 2%.  Any
    remaining factor is quadrature-sized: up to 2% silently, up to 30% with the
    grey_rescaled flag, and larger factors fail.
    STAR has no flux-weighted absorption mean, so that grey value is the one
    these groups define and is returned in diagnostics["krosseland_absorption"].
    """
    hnu_edges = np.asarray(hnu_edges, dtype=np.float64)
    spectra = {"bb": bb, "continuum": np.asarray(bf, dtype=np.float64) + np.asarray(ff, dtype=np.float64), "sc": sc}
    if planck_line_array is not None:
        spectra["bb_planck"] = planck_line_array
    use_rosseland_array = rosseland_point_lines and rosseland_line_array is not None
    if use_rosseland_array:
        spectra["bb_ros"] = rosseland_line_array
    _hnu_raw, u_raw, raw, temperature = _sorted_unique_spectrum(hnu_ev, u, spectra, hnu_edges)
    quadrature = NativeGroupQuadrature(u_raw, hnu_edges / temperature)
    log_planck_group_weight, log_rosseland_group_weight = _gl_group_log_weights(hnu_edges, temperature)
    bb_nodes = quadrature.interpolate(raw["bb"])
    continuum_nodes = quadrature.interpolate(raw["continuum"])
    sc_nodes = quadrature.interpolate(raw["sc"])
    planck_bb_nodes = quadrature.interpolate(raw["bb_planck"]) if planck_line_array is not None else bb_nodes
    rosseland_bb_nodes = quadrature.interpolate(raw["bb_ros"]) if use_rosseland_array else bb_nodes

    planck_lines = quadrature.planck(planck_bb_nodes)
    planck_continuum = quadrature.planck(continuum_nodes)
    planck_scattering = quadrature.planck(sc_nodes)
    grid_lines = _planck_regroup(planck_lines, log_planck_group_weight)
    grid_continuum = _planck_regroup(planck_continuum, log_planck_group_weight)

    kplanck_target = float(target_means["kplanck"])
    if kplanck_lines is None or not math.isfinite(kplanck_lines):
        # No separate line strength: lines and continuum share one quadrature-sized factor.
        continuum_factor = _residual_factor("kplanck", kplanck_target, grid_lines + grid_continuum)
        line_scale = continuum_factor
        star_lines = line_scale * grid_lines
        unresolved = False
    else:
        line_scale = _planck_line_scale(kplanck_lines, grid_lines)
        star_lines = float(kplanck_lines)
        unresolved = not (line_scale > 0.0 and abs(math.log(line_scale)) <= LINE_UNRESOLVED_LOG)
        continuum_factor = _residual_factor(
            "kplanck continuum", max(kplanck_target - star_lines, 0.0), grid_continuum)
    kplanck = continuum_factor * planck_continuum
    if not unresolved:
        kplanck = kplanck + line_scale * planck_lines
    unresolved_lines = star_lines if unresolved else 0.0

    scattering_factor = _residual_factor(
        "kplanck_scattering", float(target_means["kplanck_scattering"]),
        _planck_regroup(planck_scattering, log_planck_group_weight),
    )

    def rosseland_groups(scale: float) -> tuple[np.ndarray, np.ndarray]:
        lines = scale * rosseland_bb_nodes
        return quadrature.rosseland(lines + continuum_nodes + sc_nodes, lines + continuum_nodes)

    krosseland_target = float(target_means["krosseland"])
    rosseland_scale = 1.0
    def grey(scale: float) -> float:
        return _rosseland_regroup(rosseland_groups(scale)[0], log_rosseland_group_weight)

    # bb_ros holds STAR's own point-sampled lines, but on spiky spectra STAR's Newton-Cotes rule and the exact group
    # integrals still differ by up to ~15%; the line strength is then fitted as for the bin-averaged lines.
    fit_rosseland = rosseland_point_lines and (
        not use_rosseland_array or abs(math.log(krosseland_target / grey(1.0))) > RESIDUAL_LOG_TOLERANCE)
    if fit_rosseland:
        without_lines = grey(0.0)
        if krosseland_target <= without_lines:
            rosseland_scale = 0.0
        else:
            upper = 1.0
            while grey(upper) < krosseland_target:
                upper *= 10.0
                if upper > 1.0e6:
                    raise ValueError("no line scale reproduces STAR's Rosseland mean")
            rosseland_scale = float(scipy.optimize.brentq(
                lambda scale: grey(scale) - krosseland_target, 0.0, upper, xtol=1e-14, rtol=1e-12))
    krosseland, krosseland_absorption = rosseland_groups(rosseland_scale)
    rosseland_factor = _residual_factor(
        "krosseland", krosseland_target, _rosseland_regroup(krosseland, log_rosseland_group_weight))
    krosseland = rosseland_factor * krosseland
    # STAR does not compute a flux-weighted absorption mean, so its grey value is defined by these groups;
    # target_means["krosseland_absorption"] is not used.
    transport_weight = _scaled_group_weights(log_rosseland_group_weight) / krosseland
    absorption_target = float(np.sum(transport_weight * krosseland_absorption) / np.sum(transport_weight))
    absorption_factor = 1.0

    planck_flags = np.zeros(quadrature.n_groups, dtype=np.uint8)
    rosseland_flags = np.zeros(quadrature.n_groups, dtype=np.uint8)
    planck_flags[quadrature.sub_cell] |= FLAG_SUB_CELL
    rosseland_flags[quadrature.sub_cell] |= FLAG_SUB_CELL
    if _beyond_tolerance(continuum_factor, scattering_factor):
        planck_flags |= FLAG_GREY_RESCALED
    if _beyond_tolerance(rosseland_factor):
        rosseland_flags |= FLAG_GREY_RESCALED
    if unresolved:
        if unresolved_lines > LINE_UNRESOLVED_FLAG_FRACTION * kplanck_target:
            planck_flags |= FLAG_LINES_UNRESOLVED
    elif abs(math.log(line_scale)) > LINE_ALIASED_LOG:
        with np.errstate(invalid="ignore", divide="ignore"):
            line_fraction = np.where(kplanck > 0.0, line_scale * planck_lines / kplanck, 0.0)
        line_groups = line_fraction > GROUP_LINE_FRACTION_FLAG
        planck_flags[line_groups] |= FLAG_LINES_ALIASED
        if abs(math.log(line_scale)) > LINE_ALIASED_X2_LOG:
            planck_flags[line_groups] |= FLAG_LINES_ALIASED_X2
    if rosseland_point_lines and rosseland_scale != 1.0:
        unscaled = rosseland_groups(1.0)[0] * rosseland_factor
        with np.errstate(invalid="ignore", divide="ignore"):
            moved = np.abs(np.log(krosseland / unscaled)) > ROSSELAND_RESCALE_FLAG_LOG
        rosseland_flags[moved] |= FLAG_ROSSELAND_LINES_RESCALED

    fields = {
        "kplanck_scattering": scattering_factor * planck_scattering,
        "kplanck": kplanck,
        "krosseland": krosseland,
        "krosseland_absorption": krosseland_absorption,
    }
    for field, values in fields.items():
        if np.any(~np.isfinite(values)) or np.any(values < 0.0):
            raise NonFiniteOutputError(f"{field} contains invalid group values")
    diagnostics = {
        "planck_line_scale": float(line_scale),
        "planck_line_fraction": float(star_lines / kplanck_target) if kplanck_target > 0.0 else 0.0,
        "planck_grid_line_fraction": float(grid_lines / (grid_lines + grid_continuum))
        if grid_lines + grid_continuum > 0.0 else 0.0,
        "planck_lines_unresolved": float(unresolved_lines),
        "planck_continuum_factor": float(continuum_factor),
        "planck_scattering_factor": float(scattering_factor),
        "rosseland_line_scale": float(rosseland_scale),
        "rosseland_factor": float(rosseland_factor),
        "rosseland_absorption_factor": float(absorption_factor),
        "krosseland_absorption": float(absorption_target),
        "planck_line_array": float(planck_line_array is not None),
        "rosseland_line_array": float(use_rosseland_array),
    }
    return GroupCollapse(fields, planck_flags, rosseland_flags, diagnostics)


def compute_group_collapse(
    hnu_ev: np.ndarray,
    u: np.ndarray,
    bb: np.ndarray,
    bf: np.ndarray,
    ff: np.ndarray,
    sc: np.ndarray,
    hnu_edges: np.ndarray,
    target_means: dict[str, float],
    kplanck_lines: float | None = None,
    rosseland_point_lines: bool = False,
) -> dict[str, np.ndarray]:
    """Group opacities only; see compute_group_collapse_v8."""
    return compute_group_collapse_v8(
        hnu_ev, u, bb, bf, ff, sc, hnu_edges, target_means, kplanck_lines, rosseland_point_lines
    ).fields

def discover_hnu_range(tasks: list[tuple[str, str, str]]) -> tuple[float, float]:
    for _label, _mode, h5_path_text in tasks:
        h5_path = Path(h5_path_text)
        if not h5_path.is_file() or h5_path.stat().st_size <= 0:
            continue
        try:
            with h5py.File(h5_path, "r") as h5:
                if "mixspec" in h5 and "hnu" in h5["mixspec"]:
                    hnu = np.asarray(h5["mixspec"]["hnu"][:], dtype=np.float64)
                elif "hnu" in h5:
                    hnu = np.asarray(h5["hnu"][:], dtype=np.float64)
                else:
                    continue
            good = np.isfinite(hnu) & (hnu > 0.0)
            if np.count_nonzero(good) < 2:
                continue
            hnu = hnu[good]
            return float(np.min(hnu)), float(np.max(hnu))
        except Exception:
            continue
    raise RuntimeError("could not determine hnu range from any starout.h5")


def scalar_dataset(group, names: tuple[str, ...]) -> float:
    for name in names:
        if name in group:
            value = np.asarray(group[name][()], dtype=np.float64).reshape(-1)
            if value.size:
                return float(value[0])
    return math.nan


def read_conditions(h5_path: Path) -> tuple[float, float]:
    with h5py.File(h5_path, "r") as h5:
        tev = scalar_dataset(h5, ("tev",))
        rho = scalar_dataset(h5, ("rho",))
        if math.isfinite(tev) and math.isfinite(rho):
            return tev, rho
    return math.nan, math.nan


def process_run(task: tuple[str, str, str, np.ndarray | None]) -> dict[str, object]:
    point_label, mode, h5_path_text, hnu_edges = task
    h5_path = Path(h5_path_text)
    record: dict[str, object] = {
        "label": point_label,
        "mode": mode,
        "h5_path": h5_path_text,
        "temp_eV": math.nan,
        "rho_gcc": math.nan,
        "kplanck_scattering": math.nan,
        "kplanck": math.nan,
        "krosseland": math.nan,
        "krosseland_absorption": math.nan,
        "group_collapse": None,
        "status": "missing",
    }

    if not h5_path.is_file() or h5_path.stat().st_size <= 0:
        return record

    try:
        with h5py.File(h5_path, "r") as h5:
            if "mixspec" in h5:
                mixspec = h5["mixspec"]
                scalar_group = mixspec
                hnu_ev = np.asarray(mixspec["hnu"][:], dtype=np.float64)
                u = np.asarray(mixspec["u"][:], dtype=np.float64)
                bb = np.asarray(mixspec["bb"][:], dtype=np.float64)
                bf = np.asarray(mixspec["bf"][:], dtype=np.float64)
                ff = np.asarray(mixspec["ff"][:], dtype=np.float64)
                sc = np.asarray(mixspec["sc"][:], dtype=np.float64)
            else:
                scalar_group = h5
                temp_for_u = scalar_dataset(h5, ("tev",))
                hnu_ev = np.asarray(h5["hnu"][:], dtype=np.float64)
                u = hnu_ev / max(temp_for_u, 1e-300)
                bb = np.asarray(h5["bb"][:], dtype=np.float64)
                bf = np.asarray(h5["bf"][:], dtype=np.float64)
                ff = np.asarray(h5["ff"][:], dtype=np.float64)
                sc = np.asarray(h5["sc"][:], dtype=np.float64)

            temp_ev = scalar_dataset(h5, ("tev",))
            rho_gcc = scalar_dataset(h5, ("rho",))
            kplanck = scalar_dataset(scalar_group, ("kplnk",))
            krosseland = scalar_dataset(scalar_group, ("kros",))
            kplanck_lines = scalar_dataset(scalar_group, ("kplnk_bb_analytical",))
            planck_line_array = (np.asarray(scalar_group["bb_planck"][:], dtype=np.float64)
                                 if "bb_planck" in scalar_group else None)
            rosseland_line_array = (np.asarray(scalar_group["bb_ros"][:], dtype=np.float64)
                                    if "bb_ros" in scalar_group else None)
            haze = scalar_dataset(scalar_group, ("rosseland_line_haze_active",))
            rosseland_point_lines = math.isfinite(haze) and haze > 0.0
            if not math.isfinite(kplanck):
                kplanck = scalar_dataset(h5, ("kplnk",))
            if not math.isfinite(krosseland):
                krosseland = scalar_dataset(h5, ("kros",))

        kappa_a = bb + bf + ff
        chi = kappa_a + sc
        rosw = rosseland_weight(u)
        kplanck_scattering = planck_mean(u, sc)
        kross_absorption = rosseland_flux_weighted_absorption(u, rosw, kappa_a, chi)
        group_collapse = None
        group_flags = None
        collapse_diagnostics = None
        if hnu_edges is not None:
            result = compute_group_collapse_v8(
                hnu_ev,
                u,
                bb,
                bf,
                ff,
                sc,
                hnu_edges,
                target_means={
                    "kplanck": kplanck,
                    "krosseland": krosseland,
                    "kplanck_scattering": kplanck_scattering,
                    "krosseland_absorption": kross_absorption,
                },
                kplanck_lines=kplanck_lines if math.isfinite(kplanck_lines) else None,
                rosseland_point_lines=rosseland_point_lines,
                planck_line_array=planck_line_array,
                rosseland_line_array=rosseland_line_array,
            )
            group_collapse = result.fields
            group_flags = {"planck_flags": result.planck_flags, "rosseland_flags": result.rosseland_flags}
            collapse_diagnostics = result.diagnostics
            kross_absorption = result.diagnostics["krosseland_absorption"]

        if (
            not math.isfinite(kplanck)
            or not math.isfinite(krosseland)
            or kplanck <= 0.0
            or krosseland <= 0.0
        ):
            record.update(
                {
                    "temp_eV": temp_ev,
                    "rho_gcc": rho_gcc,
                    "status": "unreadable",
                }
            )
            return record

        record.update(
            {
                "temp_eV": temp_ev,
                "rho_gcc": rho_gcc,
                "kplanck_scattering": kplanck_scattering,
                "kplanck": kplanck,
                "krosseland": krosseland,
                "krosseland_absorption": kross_absorption,
                "group_collapse": group_collapse,
                "group_flags": group_flags,
                "collapse_diagnostics": collapse_diagnostics,
                "status": "ok",
            }
        )
        return record
    except Exception as exc:
        record["status"] = "error: {}".format(exc)
        return record


def discover_tasks(runs_root: Path, mode: str | None) -> list[tuple[str, str, str]]:
    tasks: list[tuple[str, str, str]] = []
    for entry in sorted(runs_root.iterdir()):
        if not entry.is_dir():
            continue
        direct_h5 = entry / "starout.h5"
        if direct_h5.is_file():
            tasks.append((entry.name, "", str(direct_h5)))
            continue

        mode_dirs = [p for p in entry.iterdir() if p.is_dir()]
        if not mode_dirs:
            continue

        selected_mode = mode
        if selected_mode is None:
            if len(mode_dirs) == 1:
                selected_mode = mode_dirs[0].name
            else:
                preferred = entry / DEFAULT_MODE
                if preferred.is_dir():
                    selected_mode = DEFAULT_MODE
                else:
                    selected_mode = sorted(p.name for p in mode_dirs)[0]

        h5_path = entry / selected_mode / "starout.h5"
        tasks.append((entry.name, selected_mode, str(h5_path)))
    return tasks


def build_group_collapse_grid(
    records: list[dict[str, object]],
    temps: list[float],
    rhos: list[float],
    field: str,
    n_groups: int,
) -> np.ndarray:
    grid = np.full((n_groups, len(rhos), len(temps)), np.nan, dtype=np.float64)
    for record in records:
        if record.get("status") != "ok":
            continue
        collapse = record.get("group_collapse")
        if not isinstance(collapse, dict) or field not in collapse:
            continue
        values = np.asarray(collapse[field], dtype=np.float64).reshape(-1)
        if values.size != n_groups:
            continue
        i_rho = nearest_index(rhos, float(record["rho_gcc"]))
        j_temp = nearest_index(temps, float(record["temp_eV"]))
        grid[:, i_rho, j_temp] = values
    return grid


COLLAPSE_DIAGNOSTICS = (
    "planck_line_scale",
    "planck_line_fraction",
    "planck_grid_line_fraction",
    "planck_lines_unresolved",
    "planck_continuum_factor",
    "planck_scattering_factor",
    "rosseland_line_scale",
    "rosseland_factor",
    "planck_line_array",
    "rosseland_line_array",
)
COLLAPSE_FLAG_ARRAYS = ("planck_flags", "rosseland_flags")


def build_group_flag_grid(
    records: list[dict[str, object]],
    temps: list[float],
    rhos: list[float],
    name: str,
    n_groups: int,
) -> np.ndarray:
    grid = np.zeros((n_groups, len(rhos), len(temps)), dtype=np.uint8)
    for record in records:
        flags = record.get("group_flags")
        if record.get("status") != "ok" or not isinstance(flags, dict):
            continue
        i_rho = nearest_index(rhos, float(record["rho_gcc"]))
        j_temp = nearest_index(temps, float(record["temp_eV"]))
        grid[:, i_rho, j_temp] = np.asarray(flags[name], dtype=np.uint8)
    return grid


def build_diagnostic_grid(
    records: list[dict[str, object]],
    temps: list[float],
    rhos: list[float],
    name: str,
) -> np.ndarray:
    grid = np.full((len(rhos), len(temps)), np.nan, dtype=np.float64)
    for record in records:
        diagnostics = record.get("collapse_diagnostics")
        if record.get("status") != "ok" or not isinstance(diagnostics, dict):
            continue
        i_rho = nearest_index(rhos, float(record["rho_gcc"]))
        j_temp = nearest_index(temps, float(record["temp_eV"]))
        grid[i_rho, j_temp] = float(diagnostics[name])
    return grid


def write_group_collapse_index(
    path: Path,
    hnu_centers: np.ndarray,
    hnu_edges: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "group_index",
                "hnu_center_eV",
                "hnu_low_eV",
                "hnu_high_eV",
            ]
        )
        for index, center in enumerate(hnu_centers):
            writer.writerow(
                [
                    index,
                    fmt_grid(float(center)),
                    fmt_grid(float(hnu_edges[index])),
                    fmt_grid(float(hnu_edges[index + 1])),
                ]
            )


def write_group_collapse_outputs(
    outdir: Path,
    records: list[dict[str, object]],
    temps: list[float],
    rhos: list[float],
    hnu_edges: np.ndarray,
) -> tuple[Path, dict[str, np.ndarray]]:
    collapse_dir = outdir / "group_collapse"
    collapse_dir.mkdir(parents=True, exist_ok=True)
    hnu_centers = log_frequency_centers(hnu_edges)
    n_groups = hnu_centers.size

    npz_payload: dict[str, object] = {
        "hnu_ev_edges": hnu_edges,
        "hnu_ev_centers": hnu_centers,
        "temp_eV": np.asarray(temps, dtype=np.float64),
        "rho_gcc": np.asarray(rhos, dtype=np.float64),
    }
    collapse_grids: dict[str, np.ndarray] = {}
    for field, _stem in COLLAPSE_FIELDS:
        grid = build_group_collapse_grid(records, temps, rhos, field, n_groups)
        collapse_grids[field] = grid
        npz_payload[field] = grid

    validate_opacity_outputs(
        records,
        temps,
        rhos,
        pivot_grids=None,
        collapse_grids=collapse_grids,
        hnu_edges=hnu_edges,
    )
    for name in COLLAPSE_FLAG_ARRAYS:
        npz_payload[name] = build_group_flag_grid(records, temps, rhos, name, n_groups)
    for name in COLLAPSE_DIAGNOSTICS:
        npz_payload[name] = build_diagnostic_grid(records, temps, rhos, name)

    for field, stem in COLLAPSE_FIELDS:
        grid = collapse_grids[field]
        np.savez_compressed(collapse_dir / "{}_groups.npz".format(stem), **{
            "hnu_ev_edges": hnu_edges,
            "hnu_ev_centers": hnu_centers,
            "temp_eV": np.asarray(temps, dtype=np.float64),
            "rho_gcc": np.asarray(rhos, dtype=np.float64),
            field: grid,
        })

    np.savez_compressed(collapse_dir / "opacity_group_collapse.npz", **npz_payload)
    write_group_collapse_index(
        collapse_dir / "frequency_groups.csv",
        hnu_centers,
        hnu_edges,
    )
    manifest = {
        "n_groups": n_groups,
        "hnu_min_eV": float(hnu_edges[0]),
        "hnu_max_eV": float(hnu_edges[-1]),
        "spacing": "log",
        "shape": [n_groups, len(rhos), len(temps)],
        "axis_order": "group,rho,temp",
        "fields": [field for field, _stem in COLLAPSE_FIELDS],
        "flag_arrays": list(COLLAPSE_FLAG_ARRAYS),
        "flag_bits": {
            "lines_aliased": FLAG_LINES_ALIASED,
            "lines_aliased_x2": FLAG_LINES_ALIASED_X2,
            "lines_unresolved": FLAG_LINES_UNRESOLVED,
            "rosseland_lines_rescaled": FLAG_ROSSELAND_LINES_RESCALED,
            "sub_cell": FLAG_SUB_CELL,
            "grey_rescaled": FLAG_GREY_RESCALED,
        },
        "cell_diagnostics": list(COLLAPSE_DIAGNOSTICS),
    }
    (collapse_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return collapse_dir, collapse_grids


def attach_hnu_edges(
    tasks: list[tuple[str, str, str]],
    hnu_edges: np.ndarray | None,
) -> list[tuple[str, str, str, np.ndarray | None]]:
    return [
        (point_label, mode, h5_path_text, hnu_edges)
        for point_label, mode, h5_path_text in tasks
    ]


def fmt_grid(value: float) -> str:
    if not math.isfinite(value):
        return ""
    return "{:.12g}".format(value)


def nearest_index(values: list[float], target: float) -> int:
    arr = np.asarray(values, dtype=np.float64)
    return int(np.argmin(np.abs(arr - target)))


def records_for_grid_cell(
    records: list[dict[str, object]],
    temps: list[float],
    rhos: list[float],
    i_rho: int,
    j_temp: int,
) -> list[dict[str, object]]:
    matched: list[dict[str, object]] = []
    for record in records:
        temp_ev = float(record.get("temp_eV", math.nan))
        rho_gcc = float(record.get("rho_gcc", math.nan))
        if not math.isfinite(temp_ev) or not math.isfinite(rho_gcc):
            continue
        if nearest_index(rhos, rho_gcc) != i_rho:
            continue
        if nearest_index(temps, temp_ev) != j_temp:
            continue
        matched.append(record)
    return matched


def load_run_spectrum(h5_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(h5_path, "r") as h5:
        if "mixspec" in h5:
            mixspec = h5["mixspec"]
            hnu_ev = np.asarray(mixspec["hnu"][:], dtype=np.float64)
            u = np.asarray(mixspec["u"][:], dtype=np.float64)
            bb = np.asarray(mixspec["bb"][:], dtype=np.float64)
            bf = np.asarray(mixspec["bf"][:], dtype=np.float64)
            ff = np.asarray(mixspec["ff"][:], dtype=np.float64)
            sc = np.asarray(mixspec["sc"][:], dtype=np.float64)
        else:
            temp_for_u = scalar_dataset(h5, ("tev",))
            hnu_ev = np.asarray(h5["hnu"][:], dtype=np.float64)
            u = hnu_ev / max(temp_for_u, 1e-300)
            bb = np.asarray(h5["bb"][:], dtype=np.float64)
            bf = np.asarray(h5["bf"][:], dtype=np.float64)
            ff = np.asarray(h5["ff"][:], dtype=np.float64)
            sc = np.asarray(h5["sc"][:], dtype=np.float64)
    return hnu_ev, u, bb, bf, ff, sc


def group_mask(hnu_ev: np.ndarray, hnu_edges: np.ndarray, group: int) -> np.ndarray:
    lo = hnu_edges[group]
    hi = hnu_edges[group + 1]
    if group < hnu_edges.size - 2:
        return (hnu_ev >= lo) & (hnu_ev < hi)
    return (hnu_ev >= lo) & (hnu_ev <= hi)


def diagnose_group_bin_failure(
    field: str,
    group: int,
    hnu_edges: np.ndarray,
    hnu_ev: np.ndarray,
    u: np.ndarray,
    bb: np.ndarray,
    bf: np.ndarray,
    ff: np.ndarray,
    sc: np.ndarray,
) -> list[str]:
    """Explain a non-finite group using the same stable routines as production."""
    lines: list[str] = []
    lo = float(hnu_edges[group])
    hi = float(hnu_edges[group + 1])
    mask = group_mask(hnu_ev, hnu_edges, group)
    n_points = int(np.count_nonzero(mask))
    lines.append(
        "    frequency group {}: hnu in [{:.6g}, {:.6g}] eV has {} spectral point(s)".format(
            group, lo, hi, n_points
        )
    )
    if n_points == 0:
        lines.append(
            "    reason: no starout.h5 frequency samples fall in this bin; "
            "neighbor fill could not find a finite anchor"
        )
        return lines

    ub = np.asarray(u[mask], dtype=np.float64)
    finite_u = ub[np.isfinite(ub) & (ub > 0.0)]
    if finite_u.size:
        lines.append(
            "    dimensionless energy u range: [{:.6g}, {:.6g}]".format(
                float(np.min(finite_u)), float(np.max(finite_u))
            )
        )

    kappa_a = bb + bf + ff
    chi = kappa_a + sc
    rosw = rosseland_weight(u)
    if field == "kplanck":
        value = planck_mean_bin(u, kappa_a, mask)
        log_weight = planck_log_weight_u(ub)
    elif field == "kplanck_scattering":
        value = planck_mean_bin(u, sc, mask)
        log_weight = planck_log_weight_u(ub)
    elif field == "krosseland":
        value = rosseland_mean_bin(u, rosw, chi, mask)
        log_weight = rosseland_log_weight_u(ub)
    elif field == "krosseland_absorption":
        value = rosseland_absorption_bin(u, rosw, kappa_a, chi, mask)
        log_weight = rosseland_log_weight_u(ub)
    else:
        lines.append("    reason: unknown collapse field")
        return lines

    finite_log_weight = log_weight[np.isfinite(log_weight)]
    if finite_log_weight.size:
        lines.append(
            "    log-weight range before common-factor cancellation: "
            "[{:.6g}, {:.6g}]".format(
                float(np.min(finite_log_weight)),
                float(np.max(finite_log_weight)),
            )
        )
    lines.append("    stable recomputed value: {}".format(value))
    if math.isfinite(value):
        lines.append(
            "    reason: stable bin computation is finite; failure likely occurred "
            "during grid assembly or neighbor filling"
        )
    else:
        lines.append(
            "    reason: no finite stable ratio could be formed from the input samples"
        )
    return lines

def diagnose_record_group_collapse(
    record: dict[str, object],
    field: str,
    group: int,
    hnu_edges: np.ndarray,
) -> list[str]:
    lines: list[str] = []
    label = record.get("label", "?")
    h5_path_text = str(record.get("h5_path", ""))
    lines.append(
        "  run label={} status={} T={} rho={}".format(
            label,
            record.get("status", "?"),
            record.get("temp_eV", "?"),
            record.get("rho_gcc", "?"),
        )
    )
    lines.append("  h5_path={}".format(h5_path_text))
    collapse = record.get("group_collapse")
    if not isinstance(collapse, dict):
        lines.append("  reason: run has no group_collapse data")
        return lines

    values = np.asarray(collapse.get(field, []), dtype=np.float64).reshape(-1)
    if group >= values.size:
        lines.append(
            "  reason: collapse field '{}' has {} groups, expected {}".format(
                field, values.size, hnu_edges.size - 1
            )
        )
        return lines

    finite_count = int(np.count_nonzero(np.isfinite(values)))
    lines.append(
        "  collapse field '{}' has {}/{} finite groups before grid assembly".format(
            field, finite_count, values.size
        )
    )
    if finite_count == 0:
        lines.append(
            "  reason: every frequency group is non-finite for this run; "
            "raw-spectrum interpolation failed to produce a finite value"
        )
        return lines

    h5_path = Path(h5_path_text)
    if not h5_path.is_file():
        lines.append("  reason: starout.h5 is missing; cannot inspect spectrum")
        return lines

    try:
        hnu_ev, u, bb, bf, ff, sc = load_run_spectrum(h5_path)
    except Exception as exc:
        lines.append("  reason: failed to read spectrum from starout.h5: {}".format(exc))
        return lines

    lines.extend(
        diagnose_group_bin_failure(
            field, group, hnu_edges, hnu_ev, u, bb, bf, ff, sc
        )
    )
    return lines


def validate_finite_array(
    name: str,
    array: np.ndarray,
    detail_lines: list[str],
) -> None:
    arr = np.asarray(array, dtype=np.float64)
    bad_mask = ~np.isfinite(arr)
    if not np.any(bad_mask):
        return

    nan_count = int(np.isnan(arr).sum())
    inf_count = int(np.isinf(arr).sum())
    bad_count = int(bad_mask.sum())
    lines = [
        "Non-finite values in {}:".format(name),
        "  nan={}, inf={}, bad={}/{} ({:.4f}%)".format(
            nan_count,
            inf_count,
            bad_count,
            arr.size,
            100.0 * bad_count / max(arr.size, 1),
        ),
    ]
    lines.extend(detail_lines)
    raise NonFiniteOutputError("\n".join(lines))


def validate_pivot_grid(
    field: str,
    grid: np.ndarray,
    records: list[dict[str, object]],
    temps: list[float],
    rhos: list[float],
) -> None:
    bad_indices = np.argwhere(~np.isfinite(grid))
    if bad_indices.size == 0:
        return

    detail_lines: list[str] = []
    for i_rho, j_temp in bad_indices[:MAX_VALIDATION_EXAMPLES]:
        rho = rhos[i_rho]
        temp = temps[j_temp]
        detail_lines.append(
            "  cell rho={} g/cc, temp={} eV (grid index rho={}, temp={})".format(
                fmt_grid(rho), fmt_grid(temp), i_rho, j_temp
            )
        )
        matched = records_for_grid_cell(records, temps, rhos, i_rho, j_temp)
        if not matched:
            detail_lines.append(
                "    reason: no run maps to this grid cell after nearest-index snapping"
            )
            continue
        ok_records = [record for record in matched if record.get("status") == "ok"]
        if not ok_records:
            statuses = sorted({str(record.get("status", "?")) for record in matched})
            detail_lines.append(
                "    reason: {} run(s) map here but none have status=ok; statuses={}".format(
                    len(matched), ", ".join(statuses)
                )
            )
            continue
        record = ok_records[0]
        value = float(record.get(field, math.nan))
        detail_lines.append(
            "    mapped run label={} status={} {}={}".format(
                record.get("label", "?"),
                record.get("status", "?"),
                field,
                value,
            )
        )
        if not math.isfinite(value):
            detail_lines.append(
                "    reason: scalar {} is non-finite for this run".format(field)
            )
        else:
            detail_lines.append(
                "    reason: unexpected pivot assembly failure despite finite scalar value"
            )

    remaining = bad_indices.shape[0] - MAX_VALIDATION_EXAMPLES
    if remaining > 0:
        detail_lines.append("  ... and {} more non-finite pivot cell(s)".format(remaining))

    validate_finite_array(field, grid, detail_lines)


def validate_group_collapse_grid(
    field: str,
    grid: np.ndarray,
    records: list[dict[str, object]],
    temps: list[float],
    rhos: list[float],
    hnu_edges: np.ndarray,
) -> None:
    bad_indices = np.argwhere(~np.isfinite(grid))
    if bad_indices.size == 0:
        return

    detail_lines: list[str] = []
    for group, i_rho, j_temp in bad_indices[:MAX_VALIDATION_EXAMPLES]:
        rho = rhos[i_rho]
        temp = temps[j_temp]
        detail_lines.append(
            "  group={}, rho={} g/cc, temp={} eV (hnu {:.6g}..{:.6g} eV)".format(
                group,
                fmt_grid(rho),
                fmt_grid(temp),
                float(hnu_edges[group]),
                float(hnu_edges[group + 1]),
            )
        )
        matched = records_for_grid_cell(records, temps, rhos, i_rho, j_temp)
        ok_records = [record for record in matched if record.get("status") == "ok"]
        if not ok_records:
            if not matched:
                detail_lines.append(
                    "    reason: no run maps to this (rho, temp) grid cell"
                )
            else:
                statuses = sorted({str(record.get("status", "?")) for record in matched})
                detail_lines.append(
                    "    reason: mapped run(s) are not ok; statuses={}".format(
                        ", ".join(statuses)
                    )
                )
            continue
        detail_lines.extend(
            diagnose_record_group_collapse(ok_records[0], field, group, hnu_edges)
        )

    remaining = bad_indices.shape[0] - MAX_VALIDATION_EXAMPLES
    if remaining > 0:
        detail_lines.append(
            "  ... and {} more non-finite group-collapse cell(s)".format(remaining)
        )

    validate_finite_array(field, grid, detail_lines)


def validate_opacity_outputs(
    records: list[dict[str, object]],
    temps: list[float],
    rhos: list[float],
    pivot_grids: dict[str, np.ndarray] | None,
    collapse_grids: dict[str, np.ndarray] | None,
    hnu_edges: np.ndarray | None,
) -> None:
    if pivot_grids is not None:
        for field, grid in pivot_grids.items():
            validate_pivot_grid(field, grid, records, temps, rhos)

    if collapse_grids is not None and hnu_edges is not None:
        for field, grid in collapse_grids.items():
            validate_group_collapse_grid(
                field, grid, records, temps, rhos, hnu_edges
            )

    for axis_name, values in (("temp_eV", temps), ("rho_gcc", rhos)):
        arr = np.asarray(values, dtype=np.float64)
        validate_finite_array(
            axis_name,
            arr,
            ["  reason: axis coordinate is non-finite"],
        )

    if hnu_edges is not None:
        validate_finite_array(
            "hnu_ev_edges",
            np.asarray(hnu_edges, dtype=np.float64),
            ["  reason: frequency-group edge is non-finite"],
        )
        centers = log_frequency_centers(np.asarray(hnu_edges, dtype=np.float64))
        validate_finite_array(
            "hnu_ev_centers",
            centers,
            ["  reason: frequency-group center is non-finite"],
        )


def build_pivot(
    records: list[dict[str, object]],
    temps: list[float],
    rhos: list[float],
    field: str,
) -> np.ndarray:
    grid = np.full((len(rhos), len(temps)), np.nan, dtype=np.float64)
    for record in records:
        if record.get("status") != "ok":
            continue
        temp_ev = float(record["temp_eV"])
        rho_gcc = float(record["rho_gcc"])
        value = float(record[field])
        if not math.isfinite(value):
            continue
        i_rho = nearest_index(rhos, rho_gcc)
        j_temp = nearest_index(temps, temp_ev)
        grid[i_rho, j_temp] = value
    return grid


def write_pivot_csv(
    path: Path,
    temps: list[float],
    rhos: list[float],
    grid: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rho_gcc"] + [fmt_grid(t) for t in temps])
        for rho, row in zip(rhos, grid):
            writer.writerow([fmt_grid(rho)] + [fmt_grid(v) for v in row])


def write_long_csv(path: Path, records: list[dict[str, object]]) -> None:
    columns = [
        "temp_eV",
        "rho_gcc",
        "mode",
        "label",
        "kplanck_scattering",
        "kplanck",
        "krosseland",
        "krosseland_absorption",
        "status",
        "h5_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in sorted(
            records,
            key=lambda item: (
                float(item.get("temp_eV", math.nan))
                if math.isfinite(float(item.get("temp_eV", math.nan)))
                else math.inf,
                float(item.get("rho_gcc", math.nan))
                if math.isfinite(float(item.get("rho_gcc", math.nan)))
                else math.inf,
            ),
        ):
            writer.writerow({key: record.get(key, "") for key in columns})


def progress_line(done: int, total: int, started: float, record: dict[str, object]) -> str:
    elapsed = max(time.time() - started, 1e-9)
    rate = done / elapsed
    remaining = (total - done) / rate if rate > 0 else math.nan
    label = record.get("label", "?")
    status = record.get("status", "?")
    temp = record.get("temp_eV", math.nan)
    rho = record.get("rho_gcc", math.nan)
    return (
        "[{done}/{total}] {pct:5.1f}%  {rate:5.1f} runs/s  ETA {eta:6.0f}s  "
        "last={label} T={temp} rho={rho} status={status}"
    ).format(
        done=done,
        total=total,
        pct=100.0 * done / max(total, 1),
        rate=rate,
        eta=remaining if math.isfinite(remaining) else 0.0,
        label=label,
        temp=fmt_grid(float(temp)) if math.isfinite(float(temp)) else "?",
        rho=fmt_grid(float(rho)) if math.isfinite(float(rho)) else "?",
        status=status,
    )


def collect_records(
    tasks: list[tuple[str, str, str, np.ndarray | None]],
    workers: int,
) -> list[dict[str, object]]:
    total = len(tasks)
    if total == 0:
        return []

    started = time.time()
    records: list[dict[str, object]] = []
    if workers <= 1:
        for index, task in enumerate(tasks, start=1):
            record = process_run(task)
            records.append(record)
            print(progress_line(index, total, started, record), file=sys.stderr, flush=True)
        return records

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_run, task): task for task in tasks}
        done = 0
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            done += 1
            print(progress_line(done, total, started, record), file=sys.stderr, flush=True)
    return records


def unique_sorted(values: list[float]) -> list[float]:
    arr = np.asarray(values, dtype=np.float64)
    good = arr[np.isfinite(arr)]
    if good.size == 0:
        return []
    uniq = np.unique(np.round(good, decimals=12))
    return sorted(float(v) for v in uniq)


def resolve_axes(
    records: list[dict[str, object]],
    runs_root: Path,
) -> tuple[list[float], list[float]]:
    manifest_axes = load_manifest_axes(runs_root)
    if manifest_axes is not None:
        return manifest_axes

    temps: list[float] = []
    rhos: list[float] = []
    for record in records:
        temp = float(record.get("temp_eV", math.nan))
        rho = float(record.get("rho_gcc", math.nan))
        if math.isfinite(temp):
            temps.append(temp)
        if math.isfinite(rho):
            rhos.append(rho)
    return unique_sorted(temps), unique_sorted(rhos)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build opacity tables from a STAR FAC grid runs directory "
            "(output of run_tops_star_fac_grid.py)."
        )
    )
    parser.add_argument(
        "runs_dir",
        type=Path,
        help="Grid output directory or its runs/ subdirectory",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory (default: <grid>/opacity_tables)",
    )
    parser.add_argument(
        "--mode",
        default=None,
        help="Mode subdirectory name (default: auto-detect)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 1))),
        help="Parallel worker processes (default: up to 8)",
    )
    parser.add_argument(
        "--freq-groups",
        type=int,
        default=DEFAULT_FREQ_GROUPS,
        help=(
            "Number of log-spaced frequency groups for group collapse "
            "(default: {}, 0 disables)".format(DEFAULT_FREQ_GROUPS)
        ),
    )
    parser.add_argument(
        "--hnu-min",
        type=float,
        default=None,
        help="Minimum photon energy [eV] for frequency groups (default: from data)",
    )
    parser.add_argument(
        "--hnu-max",
        type=float,
        default=None,
        help="Maximum photon energy [eV] for frequency groups (default: from data)",
    )
    parser.add_argument(
        "--skip-mean-tables",
        action="store_true",
        help="Only build frequency-group collapse tables",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runs_root = resolve_runs_root(args.runs_dir)
    if not runs_root.is_dir():
        print("Runs directory not found: {}".format(runs_root), file=sys.stderr)
        return 1

    outdir = resolve_outdir(runs_root, args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tasks = discover_tasks(runs_root, args.mode)
    if not tasks:
        print("No run directories found under {}".format(runs_root), file=sys.stderr)
        return 1

    hnu_edges = None
    if args.freq_groups > 0:
        hnu_min, hnu_max = discover_hnu_range(tasks)
        if args.hnu_min is not None:
            hnu_min = float(args.hnu_min)
        if args.hnu_max is not None:
            hnu_max = float(args.hnu_max)
        hnu_edges = log_frequency_edges(hnu_min, hnu_max, args.freq_groups)
        print(
            "Frequency groups: {} log-spaced bins from {:.6g} to {:.6g} eV".format(
                args.freq_groups, hnu_min, hnu_max
            ),
            file=sys.stderr,
            flush=True,
        )

    worker_tasks = attach_hnu_edges(tasks, hnu_edges)

    print(
        "Processing {} runs from {} with {} worker(s)".format(
            len(worker_tasks), runs_root, args.workers
        ),
        file=sys.stderr,
        flush=True,
    )

    records = collect_records(worker_tasks, args.workers)
    temps, rhos = resolve_axes(records, runs_root)
    if not temps or not rhos:
        print("Could not determine temperature/density axes.", file=sys.stderr)
        return 1

    if not args.skip_mean_tables:
        pivot_specs = (
            ("kplanck_scattering", "scattering_planck.csv"),
            ("kplanck", "planck.csv"),
            ("krosseland", "rosseland.csv"),
            ("krosseland_absorption", "rosseland_absorption.csv"),
        )

        npz_payload: dict[str, object] = {
            "temp_eV": np.asarray(temps, dtype=np.float64),
            "rho_gcc": np.asarray(rhos, dtype=np.float64),
        }
        pivot_grids: dict[str, np.ndarray] = {}
        for field, filename in pivot_specs:
            grid = build_pivot(records, temps, rhos, field)
            pivot_grids[field] = grid

        try:
            validate_opacity_outputs(
                records,
                temps,
                rhos,
                pivot_grids=pivot_grids,
                collapse_grids=None,
                hnu_edges=None,
            )
        except NonFiniteOutputError as exc:
            print(str(exc), file=sys.stderr)
            return 1

        for field, filename in pivot_specs:
            write_pivot_csv(outdir / filename, temps, rhos, pivot_grids[field])
            npz_payload[field] = pivot_grids[field]

        write_long_csv(outdir / "opacity_table.csv", records)
        np.savez_compressed(outdir / "opacity_tables.npz", **npz_payload)

    collapse_dir = None
    if hnu_edges is not None:
        try:
            collapse_dir, _collapse_grids = write_group_collapse_outputs(
                outdir, records, temps, rhos, hnu_edges
            )
        except NonFiniteOutputError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    ok_count = sum(1 for record in records if record.get("status") == "ok")
    if collapse_dir is not None:
        print(
            "Wrote mean tables to {} and group collapse to {} ({} / {} runs ok)".format(
                outdir, collapse_dir, ok_count, len(records)
            ),
            file=sys.stderr,
        )
    else:
        print(
            "Wrote tables to {} ({} / {} runs ok)".format(
                outdir, ok_count, len(records)
            ),
            file=sys.stderr,
        )
    return 0 if ok_count > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
