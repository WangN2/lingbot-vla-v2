"""T-001 (Phase 2): config drift protection.

Guards three drift surfaces:

1. Architecture-critical values in ``configs/vla/robotwin/robotwin.yaml``
   against the documented recipe in ``configs/vla/Training_Config.md``.
   Deployment knobs (paths, batch sizes, lr schedule, checkpointing, and the
   MoE balancing *strategy*) are intentionally exempt; strategy legality is
   enforced separately below.
2. MoE balancing hyperparameter combinations: sign/mode validity plus the
   code-enforced DistMuon constraints (``lingbotvla/optim/dist_muon.py``).
3. ``norm_type`` values against the enum actually implemented by
   ``Normalizer.normalize`` (``lingbotvla/data/vla_data/transform.py``),
   including the sincos state-only restriction and the per-joint coverage
   that ``lingbotvla/data/vla_data/utils.py`` asserts at runtime.
"""

import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from lingbotvla.data.vla_data.transform import Normalizer
from lingbotvla.optim import build_flex_shard_dist_muon_optimizer

REPO_ROOT = Path(__file__).resolve().parents[1]
VLA_CONFIG_DIR = REPO_ROOT / "configs" / "vla"
TRAINING_DOC = VLA_CONFIG_DIR / "Training_Config.md"
REPO_ROBOTWIN = VLA_CONFIG_DIR / "robotwin" / "robotwin.yaml"
REPO_ROBOTWIN_DIST_MUON = VLA_CONFIG_DIR / "robotwin" / "robotwin_dist_muon.yaml"
REPO_REAL_ROBOT = VLA_CONFIG_DIR / "real_robot" / "real_robot.yaml"
ALL_VLA_CONFIGS = [REPO_ROBOTWIN, REPO_ROBOTWIN_DIST_MUON, REPO_REAL_ROBOT]


def load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get(dotted, cfg):
    """Dotted-path lookup, e.g. get('train.token_num_experts', cfg)."""
    node = cfg
    for part in dotted.split("."):
        node = node[part]
    return node


def doc_text():
    return TRAINING_DOC.read_text(encoding="utf-8")


def doc_config_blocks():
    """Parse the full ```yaml config examples embedded in Training_Config.md.

    Full examples are rooted at model/data/train; inline snippets (gradient
    accumulation) are skipped.
    """
    blocks = []
    for m in re.finditer(r"```yaml\n(.*?)```", doc_text(), re.S):
        try:
            parsed = yaml.safe_load(m.group(1))
        except yaml.YAMLError:
            continue
        if isinstance(parsed, dict) and {"model", "data", "train"} <= set(parsed):
            blocks.append(parsed)
    return blocks


def doc_robotwin_example():
    """The RoboTwin example is the only doc block with prompt_type: global."""
    matches = [b for b in doc_config_blocks() if b["data"].get("prompt_type") == "global"]
    assert len(matches) == 1, "Training_Config.md must contain exactly one RoboTwin example"
    return matches[0]


def declared_joints(cfg):
    joints = {}
    for entry in cfg["data"]["joints"]:
        assert isinstance(entry, dict) and len(entry) == 1, entry
        joints.update(entry)
    return joints


def joint_norm_types(cfg):
    norms = {}
    for entry in cfg["data"].get("norm_type") or []:
        assert isinstance(entry, dict) and len(entry) == 1, entry
        norms.update(entry)
    return norms


# ---------------------------------------------------------------------------
# 1. doc <-> config alignment (architecture-critical values only)
# ---------------------------------------------------------------------------

# Values that shape the model / data pipeline and must not drift silently
# between the documented recipe and the shipped robotwin config.
ARCH_KEYS = [
    "model.post_training",
    "model.adanorm_time",
    "model.config_key",
    "model.moe_implementation",
    "data.joints",
    "data.cameras",
    "data.norm_type",
    "train.use_moe",
    "train.token_moe_layers",
    "train.token_num_experts",
    "train.token_top_k",
    "train.token_moe_intermediate_size",
    "train.token_shared_intermediate_size",
    "train.router_activation",
    "train.routed_scaling_factor",
    "train.use_shared_expert_gate",
    "train.use_moe_expert_lr",
    "train.loss_type",
    "train.tokenizer_max_length",
    "train.action_dim",
    "train.max_action_dim",
    "train.max_state_dim",
]

# Deployment knobs the local config intentionally overrides (paths, scale,
# schedule, checkpointing cadence, balancing strategy, attention extras).
# Listed here so every *other* difference between doc and repo is a failure.
EXEMPT_KEYS = {
    "model.model_path",
    "model.tokenizer_path",
    "data.train_path",
    "data.norm_stats_file",
    "data.num_workers",
    "train.output_dir",
    "train.enable_gradient_checkpointing",
    "train.use_compile",
    "train.use_wandb",
    "train.bias_update_speed",
    "train.sequence_wise_mode",
    "train.sequence_wise_loss_coeff",
    "train.router_z_loss_coeff",
    "train.lr",
    "train.lr_min",
    "train.lr_decay_style",
    "train.optimizer",
    "train.micro_batch_size",
    "train.global_batch_size",
    "train.max_steps",
    "train.save_steps",
    "train.save_epochs",
    "train.num_train_epochs",
}


def test_doc_contains_two_full_config_examples():
    blocks = doc_config_blocks()
    assert len(blocks) == 2, "expected real_robot + robotwin examples in Training_Config.md"


@pytest.mark.parametrize("key", ARCH_KEYS)
def test_robotwin_arch_key_matches_doc(key):
    doc_cfg = doc_robotwin_example()
    repo_cfg = load_yaml(REPO_ROBOTWIN)
    assert get(key, doc_cfg) == get(key, repo_cfg), (
        f"{key} drifted between Training_Config.md and robotwin.yaml; "
        "update the doc or the config intentionally (and adjust ARCH_KEYS/EXEMPT_KEYS)"
    )


def test_repo_robotwin_keys_are_doc_or_exempt():
    """Every train-section knob in the repo config must either match the doc
    example or be explicitly exempted — no silent undocumented knobs."""
    doc_cfg = doc_robotwin_example()
    repo_cfg = load_yaml(REPO_ROBOTWIN)
    repo_train = repo_cfg["train"]
    doc_train = doc_cfg.get("train") or {}
    for key in repo_train:
        if key in doc_train:
            continue
        assert f"train.{key}" in EXEMPT_KEYS, (
            f"robotwin.yaml sets train.{key!r} which is neither in the "
            "Training_Config.md recipe nor in EXEMPT_KEYS — document it or "
            "extend the exemption list"
        )


@pytest.mark.parametrize(
    "param",
    [
        "bias_update_speed",
        "sequence_wise_mode",
        "sequence_wise_loss_coeff",
        "router_z_loss_coeff",
        "dist_muon",
        "prompt_type",
    ],
)
def test_doc_documents_balancing_and_optimizer_knobs(param):
    """Repo configs use these knobs; the doc must keep documenting them."""
    assert f"`{param}`" in doc_text(), f"Training_Config.md no longer documents {param}"


def test_doc_prompt_type_options_match_code():
    """Code allows only global/subtask (train_lingbotvla.py Literal);
    the doc must not advertise an unsupported default or option."""
    row = next(line for line in doc_text().splitlines() if line.startswith("| `prompt_type`"))
    options = set(re.findall(r'"([a-z_]+)"', row))
    assert options == {"global", "subtask"}, (
        "prompt_type options drifted from the Literal['global', 'subtask'] in "
        "tasks/vla/train_lingbotvla.py"
    )


# ---------------------------------------------------------------------------
# 2. MoE balancing hyperparameter combination legality
# ---------------------------------------------------------------------------

# modeling_lingbot_vla_v2.py: 'global' -> whole batch, otherwise per-sequence.
VALID_SEQUENCE_WISE_MODES = {"per_sequence", "global"}
# DistMuon constraints (lingbotvla/optim/dist_muon.py, raised at build time).
DIST_MUON_REQUIRED_DP_MODE = "fsdp2"


def moe_balancing_violations(bias_update_speed, sequence_wise_loss_coeff,
                             router_z_loss_coeff, sequence_wise_mode):
    """Rules the training stack actually enforces:

    - balancing weights must be non-negative (negative flips the update
      direction of the bias / aux losses);
    - sequence_wise_mode must be one of the two modes modeling supports;
    - bias_update_speed == 0 disables loss-free balancing (allowed).
    """
    problems = []
    if bias_update_speed < 0:
        problems.append("bias_update_speed must be >= 0")
    if sequence_wise_loss_coeff < 0:
        problems.append("sequence_wise_loss_coeff must be >= 0")
    if router_z_loss_coeff < 0:
        problems.append("router_z_loss_coeff must be >= 0")
    if sequence_wise_mode not in VALID_SEQUENCE_WISE_MODES:
        problems.append(
            f"sequence_wise_mode must be one of {sorted(VALID_SEQUENCE_WISE_MODES)}"
        )
    return problems


def test_repo_balancing_combos_are_legal():
    for path in ALL_VLA_CONFIGS:
        t = load_yaml(path)["train"]
        # PyYAML parses "1e-3" (no decimal point) as the *string* '1e-3'.
        # The training stack type-coerces via dataclass fields, but any raw
        # yaml.safe_load consumer would get a string — so require unambiguous
        # float literals in the shipped configs.
        for key in ("bias_update_speed", "sequence_wise_loss_coeff",
                    "router_z_loss_coeff"):
            assert isinstance(t[key], (int, float)), (
                f"{path.name}: train.{key} parses as {type(t[key]).__name__!r}; "
                "write it as 1.0e-3 (with a decimal point), not 1e-3"
            )
        problems = moe_balancing_violations(
            t["bias_update_speed"],
            t["sequence_wise_loss_coeff"],
            t["router_z_loss_coeff"],
            t["sequence_wise_mode"],
        )
        assert not problems, f"{path.name}: {problems}"


def test_vla_configs_avoid_string_scientific_notation():
    """Guard the yaml footgun everywhere: a bare 1e-3 literal parses as a
    string under yaml.safe_load (PyYAML needs the decimal point)."""
    bad = []
    for path in sorted(VLA_CONFIG_DIR.rglob("*.yaml")):
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r":\s*[0-9]+e[-+][0-9]+\s*$", line):
                bad.append(f"{path.relative_to(REPO_ROOT)}:{i}: {line.strip()}")
    assert not bad, "values parse as strings, add a decimal point:\n" + "\n".join(bad)


@pytest.mark.parametrize(
    "bias,sw,z,mode",
    [
        (-1e-4, 1e-3, 1e-4, "per_sequence"),   # negative bias speed
        (0.0, -1e-3, 0.0, "per_sequence"),     # negative sequence-wise coeff
        (0.0, 1e-3, -1e-4, "global"),          # negative z-loss coeff
        (0.0, 0.0, 0.0, "per_batch"),          # invalid mode name
    ],
)
def test_illegal_balancing_combos_are_rejected(bias, sw, z, mode):
    assert moe_balancing_violations(bias, sw, z, mode)


def _dist_muon_args(**overrides):
    base = {"data_parallel_mode": "fsdp2", "use_moe_expert_lr": False}
    base.update(overrides)
    return SimpleNamespace(**base)


def test_dist_muon_requires_fsdp2():
    with pytest.raises(ValueError, match="fsdp2"):
        build_flex_shard_dist_muon_optimizer(
            None, _dist_muon_args(data_parallel_mode="ddp"), lr=1e-4)


def test_dist_muon_rejects_moe_expert_lr():
    with pytest.raises(NotImplementedError, match="expert LR"):
        build_flex_shard_dist_muon_optimizer(
            None, _dist_muon_args(use_moe_expert_lr=True), lr=1e-4)


def test_repo_configs_do_not_violate_dist_muon_constraints():
    for path in ALL_VLA_CONFIGS:
        t = load_yaml(path)["train"]
        optimizer = t.get("optimizer", "adamw")
        if optimizer == "dist_muon":
            assert t.get("data_parallel_mode") == DIST_MUON_REQUIRED_DP_MODE, (
                f"{path.name}: dist_muon requires fsdp2"
            )
            assert not t.get("use_moe_expert_lr", False), (
                f"{path.name}: dist_muon is incompatible with use_moe_expert_lr"
            )


# ---------------------------------------------------------------------------
# 3. norm_type enum validation
# ---------------------------------------------------------------------------

# Implemented by Normalizer.normalize (lingbotvla/data/vla_data/transform.py).
SUPPORTED_NORM_TYPES = {
    "meanstd", "bounds_98", "bounds_99", "bounds_98_woclip", "bounds_99_woclip",
    "std", "minmax", "minmax_woclip", "sincos", "identity",
}


def test_doc_norm_type_options_match_implementation():
    """The doc's norm_type row must list exactly the code-supported options."""
    row = next(line for line in doc_text().splitlines() if line.startswith("| `norm_type`"))
    documented = set(re.findall(r'"([a-z0-9_]+)"', row))
    assert documented == SUPPORTED_NORM_TYPES


@pytest.mark.parametrize("path", [REPO_ROBOTWIN, REPO_REAL_ROBOT])
def test_norm_types_are_supported(path):
    for joint, norm in joint_norm_types(load_yaml(path)).items():
        assert norm in SUPPORTED_NORM_TYPES, (
            f"{path.name}: {joint} uses unknown norm_type {norm!r}"
        )


@pytest.mark.parametrize("path", [REPO_ROBOTWIN, REPO_REAL_ROBOT])
def test_norm_type_covers_all_declared_joints(path):
    """utils.py asserts at runtime that every state/action joint has a norm
    type; check it here so failures surface before training starts."""
    cfg = load_yaml(path)
    assert set(declared_joints(cfg)) == set(joint_norm_types(cfg)), (
        f"{path.name}: data.joints and data.norm_type key sets differ"
    )


def _make_normalizer():
    stats = {
        "observation.state.fake": {"mean": 0.0, "std": 1.0, "q01": -1.0, "q99": 1.0,
                                   "q02": -1.0, "q98": 1.0, "min": -1.0, "max": 1.0},
        "action.fake": {"mean": 0.0, "std": 1.0, "q01": -1.0, "q99": 1.0,
                        "q02": -1.0, "q98": 1.0, "min": -1.0, "max": 1.0},
    }
    return Normalizer(stats, norm_type={})


def test_all_documented_norm_types_run_on_state_keys():
    """Every documented norm_type is accepted by Normalizer.normalize.

    Values are torch tensors, matching what the data pipeline feeds in
    (torch.clamp in the clipping branches does not accept numpy arrays).
    """
    import torch

    normalizer = _make_normalizer()
    value = torch.zeros(4)
    for norm in SUPPORTED_NORM_TYPES:
        normalizer.norm_type = {"observation.state.fake": norm}
        out = normalizer.normalize({"observation.state.fake": value.clone()})
        assert "observation.state.fake" in out, f"norm_type {norm!r} dropped the key"


def test_unknown_norm_type_raises():
    import torch

    normalizer = _make_normalizer()
    normalizer.norm_type = {"observation.state.fake": "bogus"}
    with pytest.raises(ValueError, match="Unknown normalization type"):
        normalizer.normalize({"observation.state.fake": torch.zeros(4)})


def test_sincos_rejected_for_action_keys():
    import torch

    normalizer = _make_normalizer()
    normalizer.norm_type = {"action.fake": "sincos"}
    with pytest.raises(ValueError, match="sincos"):
        normalizer.normalize({"action.fake": torch.zeros(4)})
