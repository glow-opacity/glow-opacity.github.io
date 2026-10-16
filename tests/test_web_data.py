#!/usr/bin/env python3
from __future__ import annotations

import gzip
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def main() -> None:
    package = Path(__file__).resolve().parent.parent
    builder = package / 'scripts' / 'build_web_data.py'

    with tempfile.TemporaryDirectory() as tmp_text:
        tmp = Path(tmp_text)
        parts = tmp / 'parts'
        out = tmp / 'web_data'
        parts.mkdir()

        edges = np.logspace(-2, 2, 9)
        rho = np.logspace(-12, -10, 3)
        temp = np.logspace(-1, 1, 4)
        fields = {
            'kplanck': 2.0,
            'kplanck_scattering': 0.4,
            'krosseland': 3.0,
            'krosseland_absorption': 1.2,
        }
        for part_index, (start, stop) in enumerate(((0, 2), (2, 4))):
            payload = {
                'hnu_ev_edges': edges,
                'rho_gcc': rho,
                'temp_eV': temp[start:stop],
            }
            for field, value in fields.items():
                payload[field] = np.full(
                    (8, 3, stop - start), value, dtype=np.float64
                )
            payload['planck_flags'] = np.zeros((8, 3, stop - start), dtype=np.uint8)
            payload['planck_flags'][1, 2, 0] = 4
            payload['rosseland_flags'] = np.zeros((8, 3, stop - start), dtype=np.uint8)
            np.savez_compressed(
                parts / f'opacity_group_collapse_part{part_index:02d}.npz',
                **payload,
            )

        subprocess.run(
            [
                sys.executable,
                str(builder),
                '--parts-dir',
                str(parts),
                '--output-dir',
                str(out),
                '--group-chunk-size',
                '3',
                '--dtype',
                'float32',
            ],
            check=True,
        )

        manifest = json.loads((out / 'manifest.json').read_text())
        axes = json.loads((out / 'axes.json').read_text())
        assert manifest['schema'] == 'glow-planck-scattering-v7'
        assert manifest['dimensions'] == {
            'groups': 8, 'densities': 3, 'temperatures': 4
        }
        assert 'kplanck_scattering' in manifest['field_metadata']
        assert 'kross_scattering' not in manifest['field_metadata']
        assert manifest['storage']['dtype'] == 'float32-le'
        assert len(axes['temp_eV']) == 4

        first_part = manifest['parts'][0]
        chunk = next(
            item for item in first_part['chunks']
            if item['field'] == 'kplanck_scattering'
            and item['group_start'] == 0
        )
        with gzip.open(out / chunk['file'], 'rb') as handle:
            values = np.frombuffer(handle.read(), dtype='<f4').reshape(chunk['shape'])
        np.testing.assert_allclose(values, 0.4, rtol=2e-7, atol=0.0)

        assert manifest['quality_flags']['fields']['kplanck']['array'] == 'planck_flags'
        flag_chunk = next(
            item for item in first_part['flag_chunks']
            if item['flags'] == 'planck_flags' and item['group_start'] == 0
        )
        assert flag_chunk['dtype'] == 'uint8-le'
        with gzip.open(out / flag_chunk['file'], 'rb') as handle:
            flags = np.frombuffer(handle.read(), dtype='u1').reshape(flag_chunk['shape'])
        assert flags[1, 2, 0] == 4 and np.count_nonzero(flags) == 1

    print('Browser-data builder test passed.')


if __name__ == '__main__':
    main()
