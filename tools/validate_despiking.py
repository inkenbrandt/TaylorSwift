"""Validate all rows of a local Parquet file without loading it all at once."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from TaylorSwift.despike import despike_dataframe  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--block-rows", type=int, default=36_000)
    args = parser.parse_args()
    if args.block_rows <= 0:
        parser.error("--block-rows must be positive")
    parquet = pq.ParquetFile(args.path)
    channels = [c for c in ["Ux", "Uy", "Uz", "T_SONIC", "CO2_density",
                            "H2O_density"] if c in parquet.schema.names]
    if not channels:
        parser.error("No supported sensor columns found")
    settings = {"default": {}, "gentle": {"prob_threshold": 1e-6, "bulk_iqr": 8.0}}
    counts = {name: dict.fromkeys(channels, 0) for name in settings}
    rows = 0
    started = time.perf_counter()
    for batch in parquet.iter_batches(batch_size=args.block_rows, columns=channels):
        raw = pl.from_arrow(batch)
        for name, options in settings.items():
            clean = despike_dataframe(raw, channels, **options)
            assert clean.shape == raw.shape and clean.columns == raw.columns
            for col in channels:
                before, after = raw[col].to_numpy(), clean[col].to_numpy()
                changed = np.isfinite(before) & (~np.isfinite(after) | (before != after))
                counts[name][col] += int(changed.sum())
                assert not np.isinf(after).any(), (rows, col)
                # Linear interpolation must not leave any internal gaps.
                valid = np.flatnonzero(np.isfinite(after))
                if len(valid):
                    assert np.isfinite(after[valid[0]:valid[-1] + 1]).all(), (rows, col)
        rows += len(raw)
    assert rows == parquet.metadata.num_rows
    print(json.dumps({"rows": rows, "block_rows": args.block_rows,
                      "seconds": round(time.perf_counter() - started, 2),
                      "replaced": counts}, indent=2))


if __name__ == "__main__":
    main()
