"""Crisscross symmetric combine: symmetry by construction, side selection,
validation, and block/entrywise parity, through the bindings."""

import numpy as np
import pytest

import ellipsoid_psf as ep

from test_ellipsoid_psf_py import build_field, make_grid_mesh


def crisscross_config(k=4):
    return ep.EvalConfig(frame=ep.Frame.translation, scaling=ep.Scaling.volume,
                         support=ep.Support.ellipsoid, tau=3.0, num_neighbors=k,
                         symmetric_combine=ep.SymmetricCombine.crisscross)


def test_config_roundtrip_and_repr():
    cfg = crisscross_config()
    assert cfg.symmetric_combine == ep.SymmetricCombine.crisscross
    assert "crisscross" in repr(cfg)
    assert ep.EvalConfig().symmetric_combine == ep.SymmetricCombine.pooled


def test_requires_row_field():
    F, _ = build_field(seed=3)
    with pytest.raises(Exception):
        ep.KernelEvaluator(F, config=crisscross_config())


def test_symmetric_by_construction_and_block_parity():
    F, data = build_field(seed=5)
    K = ep.KernelEvaluator(F, F, config=crisscross_config())

    rng = np.random.default_rng(11)
    P = rng.uniform(0.2, 0.8, (7, 2))
    P[:2] = data["sample_points"][:2]          # include exact samples (ties)
    B = K.block(P, P)
    assert np.array_equal(B, B.T)              # exact symmetry, ties included

    # block matches entrywise evaluation
    for i in range(P.shape[0]):
        for j in range(P.shape[0]):
            assert B[i, j] == pytest.approx(
                K.block(P[i:i + 1], P[j:j + 1])[0, 0], rel=1e-12, abs=1e-14)


def test_side_selection_at_samples():
    F, data = build_field(seed=7)
    cfg = crisscross_config()
    K_cc = ep.KernelEvaluator(F, F, config=cfg)
    cfg_cols = ep.EvalConfig(frame=ep.Frame.translation, scaling=ep.Scaling.volume,
                             support=ep.Support.ellipsoid, tau=3.0, num_neighbors=4)
    K_cols = ep.KernelEvaluator(F, config=cfg_cols)

    xs = data["sample_points"][0]              # a sample: source distance 0
    y = np.array([0.47, 0.53])                 # not a sample
    a = K_cc.block(y[None, :], xs[None, :])[0, 0]
    b = K_cols.block(y[None, :], xs[None, :])[0, 0]
    assert a == b                              # column side selected exactly

    # transposed entry: row side selected == cols-only of the transpose
    c = K_cc.block(xs[None, :], y[None, :])[0, 0]
    assert c == b
