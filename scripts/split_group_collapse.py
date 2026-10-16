#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

DEFAULT_PART_SIZES = (26, 26, 26, 25, 25)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Split a combined group-collapse NPZ along temperature.'
    )
    parser.add_argument('source', type=Path)
    parser.add_argument('output_directory', type=Path)
    parser.add_argument(
        '--part-sizes',
        default=','.join(str(x) for x in DEFAULT_PART_SIZES),
        help='Comma-separated temperature counts per output part.',
    )
    args = parser.parse_args()

    sizes = tuple(int(x) for x in args.part_sizes.split(',') if x.strip())
    if not sizes or any(x <= 0 for x in sizes):
        raise ValueError('All part sizes must be positive integers.')

    source = args.source.expanduser().resolve()
    output = args.output_directory.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    with np.load(source, allow_pickle=False) as data:
        payload = {key: np.asarray(data[key]) for key in data.files}

    required = {
        'hnu_ev_edges', 'temp_eV', 'rho_gcc',
        'kplanck', 'kplanck_scattering',
        'krosseland', 'krosseland_absorption',
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise KeyError(f'Missing required arrays: {missing}')
    if 'kross_scattering' in payload:
        raise ValueError(
            'Stale kross_scattering is present. Use the v7 generator and do not '
            'mix old and new scattering definitions.'
        )

    temperatures = payload['temp_eV']
    densities = payload['rho_gcc']
    edges = payload['hnu_ev_edges']
    if sum(sizes) != temperatures.size:
        raise ValueError(
            f'Part sizes sum to {sum(sizes)}, but table has '
            f'{temperatures.size} temperatures.'
        )

    group_fields = []
    cell_fields = []
    expected = (edges.size - 1, densities.size, temperatures.size)
    for key, values in payload.items():
        if values.ndim == 3:
            if values.shape != expected:
                raise ValueError(f'{key}: expected {expected}, got {values.shape}')
            group_fields.append(key)
        elif values.ndim == 2:
            if values.shape != expected[1:]:
                raise ValueError(f'{key}: expected {expected[1:]}, got {values.shape}')
            cell_fields.append(key)

    start = 0
    for part_index, count in enumerate(sizes):
        stop = start + count
        part_payload = {}
        for key, values in payload.items():
            if key == 'temp_eV':
                part_payload[key] = values[start:stop]
            elif key in group_fields:
                part_payload[key] = values[:, :, start:stop]
            elif key in cell_fields:
                part_payload[key] = values[:, start:stop]
            else:
                part_payload[key] = values

        path = output / f'opacity_group_collapse_part{part_index:02d}.npz'
        np.savez_compressed(path, **part_payload)
        print(
            f'{path.name}: temperatures {start}:{stop}; '
            f'T=[{temperatures[start]:.16e}, {temperatures[stop - 1]:.16e}]'
        )
        start = stop


if __name__ == '__main__':
    main()
