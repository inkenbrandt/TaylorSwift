from __future__ import annotations
from .frame_utils import normalize_input_frame


def run_irga(df, config, *, rename_map=None, ts_col=None):
    df = normalize_input_frame(df, rename_map=rename_map, ts_col=ts_col)
    raise NotImplementedError('This skeleton package now includes extracted thermo, cleaning, and covariance modules, but the full numerical IRGA pipeline has not been migrated yet.')


def run_kh20(df, config):
    df = normalize_input_frame(df)
    raise NotImplementedError('This skeleton package now includes extracted thermo, cleaning, and covariance modules, but the full numerical KH20 pipeline has not been migrated yet.')
