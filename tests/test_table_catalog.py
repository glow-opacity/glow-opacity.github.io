#!/usr/bin/env python3
"""Validate the selectable STAR opacity-table catalog and its browser assets."""

from __future__ import annotations

import gzip
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
FIELDS = {
    "kplanck",
    "kplanck_scattering",
    "krosseland",
    "krosseland_absorption",
}
EXPECTED_IDS = {"z0p1", "z1", "z10"}


def read_gzip_size(path: Path) -> int:
    with gzip.open(path, "rb") as handle:
        return len(handle.read())


def main() -> None:
    catalog = json.loads((ROOT / "tables.json").read_text(encoding="utf-8"))
    entries = catalog["tables"]
    assert catalog["default_table"] == "z1"
    assert {entry["id"] for entry in entries} == EXPECTED_IDS

    for entry in entries:
        expected_data_root = (
            "https://raw.githubusercontent.com/glow-opacity/glow-opacity.github.io/main/tables/"
            f"{entry['id']}/web_data"
        )
        assert entry["data_root"] == expected_data_root
        data_root = ROOT / "tables" / entry["id"] / "web_data"
        manifest = json.loads((data_root / "manifest.json").read_text())
        axes = json.loads((data_root / "axes.json").read_text())
        plot_manifest = json.loads((data_root / "plot" / "manifest.json").read_text())

        assert manifest["dimensions"] == {
            "groups": 1024,
            "densities": 128,
            "temperatures": 128,
        }
        assert set(manifest["field_metadata"]) == FIELDS
        assert manifest["storage"]["dtype"] == "float32-le"
        assert len(axes["hnu_ev_edges"]) == 1025
        assert len(axes["rho_gcc"]) == 128
        assert len(axes["temp_eV"]) == 128
        assert len(manifest["parts"]) == 1

        chunks = manifest["parts"][0]["chunks"]
        assert len(chunks) == 64
        for field in FIELDS:
            field_chunks = [chunk for chunk in chunks if chunk["field"] == field]
            assert len(field_chunks) == 16
            for chunk in (field_chunks[0], field_chunks[-1]):
                path = data_root / chunk["file"]
                assert path.is_file()
                assert read_gzip_size(path) == chunk["uncompressed_bytes"]

        assert set(plot_manifest["fields"]) == FIELDS
        for field, metadata in plot_manifest["fields"].items():
            path = data_root / "plot" / metadata["file"]
            assert path.is_file()
            assert metadata["shape"] == [128, 128]
            assert metadata["dtype"] == "float32-le"
            assert read_gzip_size(path) == metadata["uncompressed_bytes"]


if __name__ == "__main__":
    main()
