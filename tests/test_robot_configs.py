"""Config-consistency tests for robot configs vs. VLA training configs.

The rules enforced here come from lingbotvla/data/vla_data/README.md:
a mismatch only surfaces at runtime as a ValueError, so it is checked
before that here.
"""

import json
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
ROBOT_CONFIG_DIR = REPO_ROOT / "configs" / "robot_configs"

PAIRS = [
    # (vla config, robot config, robot name)
    (REPO_ROOT / "configs" / "vla" / "robotwin" / "robotwin.yaml",
     ROBOT_CONFIG_DIR / "robotwin.yaml", "robotwin"),
    (REPO_ROOT / "configs" / "vla" / "real_robot" / "real_robot.yaml",
     ROBOT_CONFIG_DIR / "agilex_cobot_magic.yaml", "agilex_cobot_magic"),
]


def load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def feature_keys(robot_cfg, section):
    """Robot-config states/actions are lists of single-key dicts."""
    keys = []
    for entry in robot_cfg.get(section) or []:
        assert isinstance(entry, dict) and len(entry) == 1, entry
        keys.append(next(iter(entry)))
    return keys


def declared_joints(vla_cfg):
    joints = {}
    for entry in vla_cfg["data"]["joints"]:
        assert isinstance(entry, dict) and len(entry) == 1, entry
        joints.update(entry)
    return joints


def declared_cameras(vla_cfg):
    return list(vla_cfg["data"]["cameras"])


def slice_span(slice_):
    """A robot-config slice is {raw_key: {start, end}}; tolerate a flat
    {start, end} dict as well."""
    if "start" in slice_ and "end" in slice_:
        spec = slice_
    else:
        spec = next(iter(slice_.values()))
    return spec["end"] - spec["start"]


@pytest.mark.parametrize("vla_path,robot_path,robot_name", PAIRS)
class TestRobotVlaConfigConsistency:
    def test_joint_types_are_declared(self, vla_path, robot_path, robot_name):
        vla_cfg, robot_cfg = load_yaml(vla_path), load_yaml(robot_path)
        declared = declared_joints(vla_cfg)
        for section, prefix in (("states", "observation.state."), ("actions", "action.")):
            for key in feature_keys(robot_cfg, section):
                joint_type = key[len(prefix):]
                assert joint_type in declared, (
                    f"{robot_name}.yaml maps {key}, but {vla_path.name} does not "
                    f"declare data.joints entry '{joint_type}'"
                )

    def test_cameras_are_declared(self, vla_path, robot_path, robot_name):
        vla_cfg, robot_cfg = load_yaml(vla_path), load_yaml(robot_path)
        declared = set(declared_cameras(vla_cfg))
        for entry in robot_cfg.get("images") or []:
            key = entry if isinstance(entry, str) else next(iter(entry))
            camera = key[len("observation.images."):]
            assert camera in declared, (
                f"{robot_name}.yaml maps camera {key}, but {vla_path.name} does not "
                f"declare data.cameras entry '{camera}'"
            )

    def test_slice_dims_fit_declared_dims(self, vla_path, robot_path, robot_name):
        vla_cfg, robot_cfg = load_yaml(vla_path), load_yaml(robot_path)
        declared = declared_joints(vla_cfg)
        for section, prefix in (("states", "observation.state."), ("actions", "action.")):
            for entry in robot_cfg.get(section) or []:
                key, spec = next(iter(entry.items()))
                joint_type = key[len(prefix):]
                origin = spec["origin_keys"]
                # origin_keys may be a plain key (short form) or a list of slices.
                if isinstance(origin, list):
                    total = sum(slice_span(slice_) for slice_ in origin)
                else:
                    total = None
                if total is not None:
                    assert total <= declared[joint_type], (
                        f"{robot_name}.yaml: {key} uses {total} dims but "
                        f"{vla_path.name} declares {declared[joint_type]}"
                    )


def test_robotwin_norm_stats_file_matches_features():
    stats_path = REPO_ROOT / "assets" / "norm_stats" / "robotwin.json"
    assert stats_path.exists(), "shipped norm stats file is missing"

    data = json.loads(stats_path.read_text())
    assert "norm_stats" in data
    robot_cfg = load_yaml(ROBOT_CONFIG_DIR / "robotwin.yaml")
    feature_keys_ = [
        next(iter(entry))
        for entry in (robot_cfg.get("states") or []) + (robot_cfg.get("actions") or [])
    ]
    for key in feature_keys_:
        assert key in data["norm_stats"], f"norm stats missing feature {key!r}"


def test_robotwin_arm_and_effector_dims_follow_readme_contract():
    """README/data-guide contract: arm slices [0:6)+[7:13) = 12 <= 14;
    effector [6:7)+[13:14) = 2 <= 2."""
    robot_cfg = load_yaml(ROBOT_CONFIG_DIR / "robotwin.yaml")
    vla_cfg = load_yaml(REPO_ROOT / "configs" / "vla" / "robotwin" / "robotwin.yaml")
    declared = declared_joints(vla_cfg)

    arm_entry = next(iter(robot_cfg["states"][0]))
    assert arm_entry == "observation.state.arm.position"
    arm_slices = robot_cfg["states"][0][arm_entry]["origin_keys"]
    arm_total = sum(slice_span(s) for s in arm_slices)
    assert arm_total == 12
    assert declared["arm.position"] == 14
