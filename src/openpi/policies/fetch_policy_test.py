import numpy as np
import pytest

from openpi.policies import fetch_policy
from openpi.training import config


def test_fetch_inputs() -> None:
    inputs = fetch_policy.FetchInputs()(fetch_policy.make_fetch_example())

    assert inputs["state"].shape == (15,)
    assert inputs["image"]["base_0_rgb"].shape == (224, 224, 3)
    assert inputs["image"]["left_wrist_0_rgb"].shape == (224, 224, 3)
    assert inputs["image_mask"]["right_wrist_0_rgb"] == np.False_
    assert inputs["prompt"] == fetch_policy.PROMPT


def test_fetch_prompt_is_fixed() -> None:
    example = fetch_policy.make_fetch_example()
    example["prompt"] = "open the cabinet door"

    with pytest.raises(ValueError, match="Expected prompt"):
        fetch_policy.FetchInputs()(example)

    example = fetch_policy.make_fetch_example()
    del example["prompt"]
    with pytest.raises(ValueError, match="Prompt is required"):
        fetch_policy.FetchInputs()(example)


def test_fetch_state_and_action_dims_are_fixed() -> None:
    example = fetch_policy.make_fetch_example()
    example["observation/state"] = np.zeros(14, dtype=np.float32)
    with pytest.raises(ValueError, match="15-D state"):
        fetch_policy.FetchInputs()(example)

    example = fetch_policy.make_fetch_example()
    example["actions"] = np.zeros((50, 14), dtype=np.float32)
    with pytest.raises(ValueError, match="13-D actions"):
        fetch_policy.FetchInputs()(example)


def test_fetch_outputs_slice_action_dim() -> None:
    actions = np.zeros((50, 32), dtype=np.float32)
    outputs = fetch_policy.FetchOutputs()({"actions": actions})

    assert outputs["actions"].shape == (50, 13)


def test_pi05_hex_config() -> None:
    train_config = config.get_config("pi05_hex")
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)

    assert train_config.model.pi05
    assert data_config.repo_id == "local/hex_cabinet_open"
    assert data_config.prompt_from_task
    assert data_config.action_sequence_keys == ("actions",)
