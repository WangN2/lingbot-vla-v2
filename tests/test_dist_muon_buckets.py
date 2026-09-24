"""Unit tests for lingbotvla.optim.dist_muon bucket construction.

Buckets decide which transformer layers' Muon updates overlap with each
other; grouping regressions here change communication/compute overlap
without any error, so the grouping is pinned.
"""

import pytest

from lingbotvla.optim.dist_muon import build_dist_muon_bucket_configs


def test_adjacent_layers_share_a_bucket():
    names = [f"backbone.layers.{i}.attn.q_proj.weight" for i in range(4)]
    configs = build_dist_muon_bucket_configs(names, layers_per_bucket=2)

    assert len(configs) == 2
    assert configs[0].patterns == tuple(names[:2])
    assert configs[1].patterns == tuple(names[2:])
    # Names encode order, layer prefix, and bucket index.
    assert configs[0].name.startswith("000-")
    assert "layers" in configs[0].name
    assert configs[1].name.startswith("001-")


def test_one_layer_per_bucket():
    names = [f"backbone.layers.{i}.w" for i in range(3)]
    configs = build_dist_muon_bucket_configs(names, layers_per_bucket=1)
    assert [tuple(c.patterns) for c in configs] == [(n,) for n in names]


def test_separate_prefixes_never_share_buckets():
    names = [
        "vlm.layers.0.w",
        "vlm.layers.1.w",
        "expert.blocks.0.w",
        "expert.blocks.1.w",
    ]
    configs = build_dist_muon_bucket_configs(names, layers_per_bucket=2)

    assert len(configs) == 2
    assert configs[0].patterns == ("vlm.layers.0.w", "vlm.layers.1.w")
    assert configs[1].patterns == ("expert.blocks.0.w", "expert.blocks.1.w")


def test_non_layer_params_lands_in_other_bucket():
    configs = build_dist_muon_bucket_configs(["model.norm.weight"], layers_per_bucket=2)
    assert len(configs) == 1
    assert configs[0].patterns == ("model.norm.weight",)
    assert "other" in configs[0].name


def test_first_seen_order_is_preserved():
    names = [
        "expert.blocks.0.w",
        "vlm.layers.0.w",
        "expert.blocks.1.w",
        "vlm.layers.1.w",
    ]
    configs = build_dist_muon_bucket_configs(names, layers_per_bucket=1)
    # Buckets appear in the order their first member appears in `names`.
    assert [c.patterns[0] for c in configs] == [
        "expert.blocks.0.w",
        "vlm.layers.0.w",
        "expert.blocks.1.w",
        "vlm.layers.1.w",
    ]


def test_empty_names_yield_empty_configs():
    assert build_dist_muon_bucket_configs([], layers_per_bucket=2) == ()


@pytest.mark.parametrize("bad", [0, -1])
def test_invalid_layers_per_bucket_raises(bad):
    with pytest.raises(ValueError, match="positive"):
        build_dist_muon_bucket_configs(["a.layers.0.w"], layers_per_bucket=bad)
