"""Unit tests for lingbotvla.optim.dist_muon_params (DistMuon parameter selection).

Getting the Muon/AdamW split wrong silently routes weights through the wrong
update rule, so the routing predicates are pinned here.
"""

import torch
import torch.nn as nn

from lingbotvla.optim.dist_muon_params import (
    _is_adamw_by_name,
    _is_muon_eligible_shape,
    split_muon_adamw_params,
)


class ToyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(10, 8)  # nn.Embedding -> AdamW
        self.fc = nn.Linear(16, 8)  # 2D weight -> Muon, 1D bias -> AdamW
        self.table = nn.Parameter(torch.zeros(1, 4, 8))  # single learned table -> AdamW
        self.stack = nn.Parameter(torch.zeros(3, 4, 8))  # expert stack -> Muon
        self.frozen = nn.Parameter(torch.zeros(4, 4), requires_grad=False)  # skipped


def test_adamw_by_default_patterns():
    for name in (
        "model.embed_tokens.weight",
        "backbone.pos_embed",
        "backbone.cls_token",
        "backbone.mask_token",
        "backbone.register_tokens",
        "lm_head.weight",
        "decoder.output_layer.weight",
    ):
        assert _is_adamw_by_name(name, ()), name
    # Case-insensitive matching.
    assert _is_adamw_by_name("Foo.Embed_Tokens.x", ())
    # A routed expert weight must stay Muon-eligible.
    assert not _is_adamw_by_name("layers.0.mlp.experts.0.gate_proj.weight", ())


def test_adamw_by_extra_patterns():
    assert _is_adamw_by_name("resampler.query_tokens", ("resampler",))
    # Extra patterns are case-insensitive too.
    assert _is_adamw_by_name("some.module.weight", ("SOME.MODULE",))
    assert not _is_adamw_by_name("other.weight", ("some.module",))
    # Empty / None patterns are ignored, not matched.
    assert not _is_adamw_by_name("anything.weight", ("", None))


def test_muon_eligible_shapes():
    assert _is_muon_eligible_shape(torch.zeros(64, 32))  # 2D matrix
    assert not _is_muon_eligible_shape(torch.zeros(1, 8, 32))  # [1, N, D] learned table
    assert _is_muon_eligible_shape(torch.zeros(4, 8, 32))  # [K>1, N, D] expert stack
    assert not _is_muon_eligible_shape(torch.zeros(32))  # 1D bias / norm scale
    assert not _is_muon_eligible_shape(torch.zeros(2, 4, 8, 16))  # 4D conv kernel


def test_split_routes_toy_module():
    muon_params, adamw_params, muon_names, adamw_names = split_muon_adamw_params(ToyModule())

    assert set(muon_names) == {"fc.weight", "stack"}
    assert set(adamw_names) == {"embed_tokens.weight", "table", "fc.bias"}
    assert len(muon_params) == len(muon_names)
    assert len(adamw_params) == len(adamw_names)
    # requires_grad=False params are dropped from both groups.
    assert "frozen" not in muon_names + adamw_names


def test_split_with_extra_adamw_patterns():
    muon_params, _, muon_names, _ = split_muon_adamw_params(
        ToyModule(), extra_adamw_name_patterns=("stack",)
    )
    assert set(muon_names) == {"fc.weight"}


def test_split_with_no_decay_modules():
    # no_decay_modules matches module CLASS names (module.__class__.__name__),
    # not FQN paths.
    _, _, _, adamw_names = split_muon_adamw_params(ToyModule(), no_decay_modules=["Linear"])
    assert set(adamw_names) == {"embed_tokens.weight", "table", "fc.weight", "fc.bias"}
