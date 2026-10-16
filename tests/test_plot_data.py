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
    builder = package / 'scripts' / 'build_plot_data.py'

    with tempfile.TemporaryDirectory() as tmp_text:
        tmp = Path(tmp_text)
        parts = tmp / 'parts'
        web = tmp / 'web'
        parts.mkdir()

        edges = np.logspace(-2, 3, 9)
        rho = np.logspace(-12, -8, 3)
        temp = np.logspace(-1, 2, 4)
        expected = {
            'kplanck': 2.0,
            'kplanck_scattering': 0.5,
            'krosseland': 4.0,
            'krosseland_absorption': 1.5,
        }

        for part, (start, stop) in enumerate(((0, 2), (2, 4))):
            payload = {
                'hnu_ev_edges': edges,
                'hnu_ev_centers': np.sqrt(edges[:-1] * edges[1:]),
                'rho_gcc': rho,
                'temp_eV': temp[start:stop],
            }
            for field, value in expected.items():
                payload[field] = np.full(
                    (edges.size - 1, rho.size, stop - start),
                    value,
                    dtype=np.float64,
                )
            payload['planck_lines_unresolved'] = np.zeros((rho.size, stop - start))
            payload['planck_flags'] = np.zeros((edges.size - 1, rho.size, stop - start), dtype=np.uint8)
            payload['rosseland_flags'] = np.zeros((edges.size - 1, rho.size, stop - start), dtype=np.uint8)
            if part == 1:
                payload['planck_lines_unresolved'][0, 1] = 0.75
                payload['planck_flags'][:, 0, 1] = 4 | 16
            np.savez_compressed(
                parts / f'opacity_group_collapse_part{part:02d}.npz',
                **payload,
            )

        subprocess.run(
            [
                sys.executable,
                str(builder),
                '--parts-dir',
                str(parts),
                '--web-data-dir',
                str(web),
                '--dtype',
                'float32',
            ],
            check=True,
        )

        plot_dir = web / 'plot'
        manifest = json.loads((plot_dir / 'manifest.json').read_text())
        assert manifest['axis_order'] == ['rho', 'temp']
        assert manifest['dimensions'] == {'densities': 3, 'temperatures': 4}
        assert 'kross_scattering' not in manifest['fields']

        for field, target in expected.items():
            info = manifest['fields'][field]
            assert info['dtype'] == 'float32-le'
            with gzip.open(plot_dir / info['file'], 'rb') as handle:
                values = np.frombuffer(handle.read(), dtype='<f4').reshape(info['shape'])
            if field == 'kplanck':
                # The cell whose lines are unresolved carries them in its grey Planck mean.
                assert abs(values[0, 3] - (target + 0.75)) < 1e-6
                values = values.copy()
                values[0, 3] = target
            np.testing.assert_allclose(values, target, rtol=2e-7, atol=0.0)

        quality = manifest['quality']['kplanck']
        with gzip.open(plot_dir / quality['file'], 'rb') as handle:
            cells = np.frombuffer(handle.read(), dtype='u1').reshape(quality['shape'])
        assert cells[0, 3] == 4 and np.count_nonzero(cells) == 1  # sub_cell is not a cell-level flag

    print('Planck-scattering line-plot data test passed.')


if __name__ == '__main__':
    main()
