"""Unit tests for lingbotvla.utils.normalize (norm-stat computation).

These statistics directly drive action/state normalization during training
and inference, so mean/std/min/max correctness is on the critical path of
every post-training run.
"""

import json

import numpy as np
import pytest

pytest.importorskip("numpydantic")

from lingbotvla.utils.normalize import (  # noqa: E402
    RunningStats,
    RunningStatsState,
    deserialize_json,
    load,
    save,
    serialize_json,
)


def _batches(rng, n_batches, rows, dims):
    return [rng.normal(size=(rows, dims)) for _ in range(n_batches)]


def _stats_from(batches):
    stats = RunningStats()
    for batch in batches:
        stats.update(batch)
    return stats


def test_running_stats_mean_std_matches_numpy():
    rng = np.random.default_rng(0)
    batches = _batches(rng, n_batches=4, rows=50, dims=3)
    stats = _stats_from(batches)

    data = np.concatenate(batches, axis=0)
    result = stats.get_statistics()

    np.testing.assert_allclose(result.mean, data.mean(axis=0), rtol=1e-8)
    np.testing.assert_allclose(result.std, data.std(axis=0), rtol=1e-8)
    np.testing.assert_array_equal(result.min, data.min(axis=0))
    np.testing.assert_array_equal(result.max, data.max(axis=0))
    assert stats._count == data.shape[0]


def test_running_stats_accepts_1d_batch():
    stats = RunningStats()
    stats.update(np.array([1.0, 2.0, 3.0, 4.0]))
    result = stats.get_statistics()
    assert result.mean.shape == (1,)
    np.testing.assert_allclose(result.mean, [2.5])
    np.testing.assert_allclose(result.min, [1.0])
    np.testing.assert_allclose(result.max, [4.0])


def test_get_statistics_requires_two_vectors():
    stats = RunningStats()
    stats.update(np.zeros((1, 3)))
    with pytest.raises(ValueError, match="less than 2 vectors"):
        stats.get_statistics()


def test_update_rejects_mismatched_vector_length():
    stats = RunningStats()
    stats.update(np.zeros((4, 3)))
    with pytest.raises(ValueError, match="does not match"):
        stats.update(np.zeros((4, 5)))


def test_quantiles_ordered_and_within_range():
    rng = np.random.default_rng(1)
    stats = _stats_from(_batches(rng, n_batches=5, rows=400, dims=2))

    result = stats.get_statistics()
    assert np.all(result.q01 <= result.q02)
    assert np.all(result.q02 <= result.q98)
    assert np.all(result.q98 <= result.q99)
    assert np.all(result.min <= result.q01)
    assert np.all(result.q99 <= result.max)


def test_quantiles_approximate_numpy_on_uniform_data():
    rng = np.random.default_rng(2)
    data = rng.uniform(0.0, 1.0, size=(20_000, 1))
    stats = RunningStats()
    stats.update(data)
    result = stats.get_statistics()
    # 5000 histogram bins over [0, 1] -> resolution 2e-4 per bin.
    np.testing.assert_allclose(result.q01, np.quantile(data, 0.01, axis=0), atol=5e-3)
    np.testing.assert_allclose(result.q99, np.quantile(data, 0.99, axis=0), atol=5e-3)


def test_merge_equals_single_pass():
    rng = np.random.default_rng(3)
    data = rng.normal(size=(600, 4))
    full = RunningStats()
    full.update(data)

    a, b = RunningStats(), RunningStats()
    a.update(data[:250])
    b.update(data[250:])
    merged = RunningStats.merge([a, b])

    expected = full.get_statistics()
    got = merged.get_statistics()
    assert merged._count == full._count
    np.testing.assert_allclose(got.mean, expected.mean, rtol=1e-8)
    np.testing.assert_allclose(got.std, expected.std, rtol=1e-6)
    np.testing.assert_array_equal(got.min, expected.min)
    np.testing.assert_array_equal(got.max, expected.max)


def test_merge_requires_at_least_one_non_empty():
    with pytest.raises(ValueError, match="at least one non-empty"):
        RunningStats.merge([RunningStats(), None])


def test_state_roundtrip_preserves_statistics():
    rng = np.random.default_rng(4)
    stats = _stats_from(_batches(rng, n_batches=3, rows=40, dims=2))
    before = stats.get_statistics()

    payload = stats.get_state().model_dump_json()
    restored = RunningStats.from_state(RunningStatsState(**json.loads(payload)))
    after = restored.get_statistics()

    assert restored._count == stats._count
    np.testing.assert_allclose(after.mean, before.mean, rtol=1e-10)
    np.testing.assert_allclose(after.std, before.std, rtol=1e-10)
    np.testing.assert_array_equal(after.min, before.min)
    np.testing.assert_array_equal(after.max, before.max)


def test_serialize_deserialize_json_roundtrip():
    rng = np.random.default_rng(5)
    stats = _stats_from(_batches(rng, n_batches=2, rows=30, dims=2))
    norm = {"action.arm.position": stats.get_statistics()}

    restored = deserialize_json(serialize_json(norm, count=60))
    assert set(restored) == {"action.arm.position"}
    np.testing.assert_allclose(restored["action.arm.position"].mean, norm["action.arm.position"].mean)
    np.testing.assert_allclose(restored["action.arm.position"].std, norm["action.arm.position"].std)


def test_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(6)
    stats = _stats_from(_batches(rng, n_batches=2, rows=30, dims=2))
    norm = {"action.arm.position": stats.get_statistics()}

    # Current contract: save() writes to the exact *file* path given,
    # while load() treats its argument as a *directory* and appends
    # norm_stats.json. The two functions are asymmetric on purpose of this
    # pin; see test_save_load_path_asymmetry before "fixing" either side.
    save(tmp_path / "norm_stats.json", norm, count=60)
    loaded = load(tmp_path)
    np.testing.assert_allclose(loaded["action.arm.position"].mean, norm["action.arm.position"].mean)


def test_save_load_path_asymmetry(tmp_path):
    """Pins a known API inconsistency: save(p) writes a file at p, but
    load(p) looks for p/norm_stats.json. scripts/compute_norm_stats.py
    only uses save() with a full file path, so the asymmetry is latent.
    If this test starts failing, someone changed the contract — update
    compute_norm_stats.py and the data README together."""
    rng = np.random.default_rng(7)
    stats = _stats_from(_batches(rng, n_batches=2, rows=30, dims=2))
    norm = {"action.arm.position": stats.get_statistics()}

    file_path = tmp_path / "stats.json"
    save(file_path, norm, count=60)
    assert file_path.is_file()

    with pytest.raises(FileNotFoundError):
        load(file_path)
