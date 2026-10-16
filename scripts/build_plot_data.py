#!/usr/bin/env python3
"""Build compact full-frequency opacity arrays for GLOW line plots.

The normal browser data are chunked by native energy group.  Plotting a grey
(full-frequency integrated) opacity curve would otherwise require downloading
all 1024 groups.  This script reconstructs the conservative one-group means
from the split v7 group-collapse files and writes one small gzip-compressed
rho-by-temperature array per opacity field.

Output layout
-------------
<web-data>/plot/manifest.json
<web-data>/plot/kplanck_grey.f64.gz
<web-data>/plot/krosseland_grey.f64.gz
<web-data>/plot/krosseland_absorption_grey.f64.gz
<web-data>/plot/kplanck_scattering_grey.f64.gz

Each binary is little-endian float64 with axis order [rho, temp].
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

FIELDS = (
    "kplanck",
    "krosseland",
    "krosseland_absorption",
    "kplanck_scattering",
)
ROSS_C = 15.0 / (4.0 * math.pi**4)
FLAG_ARRAYS = ("planck_flags", "rosseland_flags")
# Bits of each flag array that apply to each field (matches build_web_data.FIELD_FLAGS).
FIELD_FLAGS = {
    "kplanck": ("planck_flags", 1 | 2 | 4 | 16 | 32),
    "kplanck_scattering": ("planck_flags", 16 | 32),
    "krosseland": ("rosseland_flags", 8 | 16 | 32),
    "krosseland_absorption": ("rosseland_flags", 8 | 16 | 32),
}
# Cell-level quality ignores sub_cell (bit 16): it marks groups narrower than STAR's grid, not a defect of the cell.
CELL_FLAG_MASK = 1 | 2 | 4 | 8 | 32
BROWSER_DTYPES = {
    "float32": np.dtype("<f4"),
    "float64": np.dtype("<f8"),
}


def logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    maximum = np.max(values, axis=axis, keepdims=True)
    finite = np.isfinite(maximum)
    shifted = np.where(finite, values - maximum, -np.inf)
    total = np.sum(np.exp(shifted), axis=axis, keepdims=True)
    result = np.where(finite & (total > 0.0), maximum + np.log(total), -np.inf)
    return np.squeeze(result, axis=axis)


def log_planck_weight(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=np.float64)
    result = np.full_like(u, -np.inf)
    good = np.isfinite(u) & (u > 0.0)
    x = u[good]
    log_expm1 = np.empty_like(x)
    small = x < 50.0
    log_expm1[small] = np.log(np.expm1(x[small]))
    log_expm1[~small] = x[~small] + np.log1p(-np.exp(-x[~small]))
    result[good] = 3.0 * np.log(x) - log_expm1
    return result


def log_rosseland_weight(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=np.float64)
    result = np.full_like(u, -np.inf)
    good = np.isfinite(u) & (u > 0.0)
    x = u[good]
    result[good] = (
        math.log(ROSS_C)
        - x
        + 4.0 * np.log(x)
        - 2.0 * np.log(-np.expm1(-x))
    )
    return result


def integrated_log_weights(
    energy_edges: np.ndarray,
    temperatures: np.ndarray,
    kind: str,
    order: int,
) -> np.ndarray:
    """Return log integrated weights with shape [group, temp]."""

    nodes, quadrature_weights = np.polynomial.legendre.leggauss(order)
    low = energy_edges[:-1, None]
    high = energy_edges[1:, None]
    energy = 0.5 * (high - low) * nodes[None, :] + 0.5 * (high + low)
    half_width = 0.5 * (high - low)
    log_qw = np.log(quadrature_weights)[None, :]

    output = np.empty((energy_edges.size - 1, temperatures.size), dtype=np.float64)
    for j, temperature in enumerate(temperatures):
        if not math.isfinite(float(temperature)) or temperature <= 0.0:
            raise ValueError(f"Invalid temperature at index {j}: {temperature}")
        u = energy / temperature
        if kind == "planck":
            kernel = log_planck_weight(u)
        elif kind == "rosseland":
            kernel = log_rosseland_weight(u)
        else:
            raise ValueError(kind)
        output[:, j] = np.log(half_width[:, 0]) + logsumexp(kernel + log_qw, axis=1)
    return output


def load_parts(parts_dir: Path, pattern: str):
    paths = sorted(parts_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files match {parts_dir / pattern}")

    edges = None
    rho = None
    temperatures = []
    field_parts = {field: [] for field in FIELDS}
    extra_parts = {name: [] for name in (*FLAG_ARRAYS, "planck_lines_unresolved")}

    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            required = {"hnu_ev_edges", "temp_eV", "rho_gcc", *FIELDS}
            missing = required.difference(data.files)
            if missing:
                raise KeyError(f"{path.name} is missing {sorted(missing)}")

            part_edges = np.asarray(data["hnu_ev_edges"], dtype=np.float64)
            part_rho = np.asarray(data["rho_gcc"], dtype=np.float64)
            part_temp = np.asarray(data["temp_eV"], dtype=np.float64)

            if edges is None:
                edges = part_edges
                rho = part_rho
            else:
                if not np.array_equal(edges, part_edges):
                    raise ValueError(f"Energy edges differ in {path.name}")
                if not np.array_equal(rho, part_rho):
                    raise ValueError(f"Density axis differs in {path.name}")

            temperatures.append(part_temp)
            expected = (part_edges.size - 1, part_rho.size, part_temp.size)
            for field in FIELDS:
                values = np.asarray(data[field], dtype=np.float64)
                if values.shape != expected:
                    raise ValueError(
                        f"{path.name}:{field} has shape {values.shape}; expected {expected}"
                    )
                if np.any(values < 0.0) or np.any(~np.isfinite(values)):
                    raise ValueError(f"{path.name}:{field} contains invalid values")
                field_parts[field].append(values)
            for name in extra_parts:
                if name in data.files:
                    extra_parts[name].append(np.asarray(data[name]))

    assert edges is not None and rho is not None
    temp = np.concatenate(temperatures)
    fields = {
        field: np.concatenate(parts, axis=2)
        for field, parts in field_parts.items()
    }
    for name, parts in extra_parts.items():
        if len(parts) == len(paths):
            fields[name] = np.concatenate(parts, axis=parts[0].ndim - 1)
        elif parts:
            raise ValueError(f"{name} is present in only some parts")
    return edges, rho, temp, fields


def arithmetic_mean(values: np.ndarray, log_weights: np.ndarray) -> np.ndarray:
    expanded = np.broadcast_to(log_weights[:, None, :], values.shape)
    terms = np.full_like(values, -np.inf)
    positive = values > 0.0
    terms[positive] = expanded[positive] + np.log(values[positive])
    log_num = logsumexp(terms, axis=0)
    log_den = logsumexp(expanded, axis=0)
    result = np.zeros_like(log_den)
    good = np.isfinite(log_num) & np.isfinite(log_den)
    result[good] = np.exp(log_num[good] - log_den[good])
    return result


def harmonic_mean(values: np.ndarray, log_weights: np.ndarray) -> np.ndarray:
    expanded = np.broadcast_to(log_weights[:, None, :], values.shape)
    has_zero = np.any(values == 0.0, axis=0)
    terms = np.full_like(values, -np.inf)
    positive = values > 0.0
    terms[positive] = expanded[positive] - np.log(values[positive])
    log_num = logsumexp(expanded, axis=0)
    log_den = logsumexp(terms, axis=0)
    result = np.zeros_like(log_num)
    good = ~has_zero & np.isfinite(log_num) & np.isfinite(log_den)
    result[good] = np.exp(log_num[good] - log_den[good])
    return result


def rosseland_absorption_mean(
    absorption: np.ndarray,
    total_rosseland: np.ndarray,
    log_weights: np.ndarray,
) -> np.ndarray:
    if np.any(total_rosseland <= 0.0):
        raise ValueError(
            "krosseland contains zero values; cannot construct the transport denominator"
        )
    expanded = np.broadcast_to(log_weights[:, None, :], total_rosseland.shape)
    effective = expanded - np.log(total_rosseland)
    numerator_terms = np.full_like(absorption, -np.inf)
    positive = absorption > 0.0
    numerator_terms[positive] = effective[positive] + np.log(absorption[positive])
    log_num = logsumexp(numerator_terms, axis=0)
    log_den = logsumexp(effective, axis=0)
    result = np.zeros_like(log_den)
    good = np.isfinite(log_num) & np.isfinite(log_den)
    result[good] = np.exp(log_num[good] - log_den[good])
    return result


def write_gzip_values(
    path: Path,
    values: np.ndarray,
    dtype: np.dtype,
    level: int,
) -> dict[str, object]:
    little = np.asarray(values, dtype=dtype, order="C")
    raw = little.tobytes(order="C")
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(filename=str(path), mode="wb", compresslevel=level, mtime=0) as handle:
        handle.write(raw)
    compressed = path.read_bytes()
    return {
        "file": path.name,
        "shape": list(little.shape),
        "axis_order": ["rho", "temp"],
        "dtype": f"{dtype.name}-le",
        "uncompressed_bytes": len(raw),
        "compressed_bytes": len(compressed),
        "sha256": hashlib.sha256(compressed).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parts-dir",
        type=Path,
        default=Path("solar_final/group_collapse"),
    )
    parser.add_argument(
        "--pattern",
        default="opacity_group_collapse_part*.npz",
    )
    parser.add_argument(
        "--web-data-dir",
        type=Path,
        default=Path("solar_final/web_data"),
    )
    parser.add_argument("--quadrature-order", type=int, default=32)
    parser.add_argument("--gzip-level", type=int, default=9)
    parser.add_argument("--dtype", choices=sorted(BROWSER_DTYPES), default="float64")
    args = parser.parse_args()

    edges, rho, temperatures, fields = load_parts(
        args.parts_dir.resolve(), args.pattern
    )
    browser_dtype = BROWSER_DTYPES[args.dtype]
    if fields["kplanck"].shape != (1024, 128, 128):
        print(f"WARNING: unexpected table shape {fields['kplanck'].shape}")

    print("Computing full-frequency Planck and Rosseland weights...")
    log_planck = integrated_log_weights(
        edges, temperatures, "planck", args.quadrature_order
    )
    log_rosseland = integrated_log_weights(
        edges, temperatures, "rosseland", args.quadrature_order
    )

    grey = {
        "kplanck": arithmetic_mean(fields["kplanck"], log_planck),
        "krosseland": harmonic_mean(fields["krosseland"], log_rosseland),
        "kplanck_scattering": arithmetic_mean(
            fields["kplanck_scattering"], log_planck
        ),
        "krosseland_absorption": rosseland_absorption_mean(
            fields["krosseland_absorption"],
            fields["krosseland"],
            log_rosseland,
        ),
    }

    # Line Planck strength the groups could not place (lines unresolved on STAR's grid) belongs to the grey mean.
    unresolved = fields.get("planck_lines_unresolved")
    if unresolved is not None:
        grey["kplanck"] = grey["kplanck"] + np.nan_to_num(unresolved, nan=0.0)

    output_dir = args.web_data_dir.resolve() / "plot"
    output_dir.mkdir(parents=True, exist_ok=True)
    quality_fields = {}
    if all(name in fields for name in FLAG_ARRAYS):
        for field, (name, mask) in FIELD_FLAGS.items():
            cell_flags = np.bitwise_or.reduce(fields[name] & (mask & CELL_FLAG_MASK), axis=0).astype(np.uint8)
            metadata = write_gzip_values(output_dir / f"{field}_quality.u8.gz", cell_flags, np.dtype("u1"), args.gzip_level)
            metadata["flagged_cells"] = int(np.count_nonzero(cell_flags))
            quality_fields[field] = metadata
    manifest_fields = {}
    for field in FIELDS:
        values = grey[field]
        if values.shape != (rho.size, temperatures.size):
            raise ValueError(f"{field} grey shape is {values.shape}")
        if np.any(values < 0.0) or np.any(~np.isfinite(values)):
            raise ValueError(f"{field} grey array contains invalid values")
        filename = f"{field}_grey.f64.gz"
        metadata = write_gzip_values(
            output_dir / filename, values, browser_dtype, args.gzip_level
        )
        metadata.update(
            {
                "zero_count": int(np.count_nonzero(values == 0.0)),
                "positive_count": int(np.count_nonzero(values > 0.0)),
            }
        )
        manifest_fields[field] = metadata
        print(
            f"{field:28s} zeros={metadata['zero_count']:8d} "
            f"compressed={metadata['compressed_bytes'] / 1024:.1f} KiB"
        )

    manifest = {
        "version": 1,
        "description": "Full-frequency weighted opacity arrays for line plots",
        "source_groups": int(edges.size - 1),
        "quadrature_order": args.quadrature_order,
        "dimensions": {
            "densities": int(rho.size),
            "temperatures": int(temperatures.size),
        },
        "axis_order": ["rho", "temp"],
        "fields": manifest_fields,
        "quality": quality_fields or None,
        "kplanck_includes_unresolved_lines": unresolved is not None,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
