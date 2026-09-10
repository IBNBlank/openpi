import dataclasses

import einops
import numpy as np

from openpi import transforms


PROMPT = "Open the cabinet door."
STATE_DIM = 15
ACTION_DIM = 13


def make_fetch_example() -> dict:
    """Create an example matching the Hex ManiSkill LeRobot export."""
    return {
        "observation/state": np.random.rand(STATE_DIM),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": PROMPT,
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected an RGB image, got shape {image.shape}")
    return image


def _validate_prompt(prompt) -> str:
    if not isinstance(prompt, str):
        prompt = prompt.item()
    if prompt != PROMPT:
        raise ValueError(f"Expected prompt {PROMPT!r}, got {prompt!r}")
    return prompt


@dataclasses.dataclass(frozen=True)
class FetchInputs(transforms.DataTransformFn):
    """Map Fetch qpos and two RGB cameras to the pi0.5 input contract."""

    def __call__(self, data: dict) -> dict:
        head_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])
        state = np.asarray(data["observation/state"], dtype=np.float32)
        if state.shape[-1] != STATE_DIM:
            raise ValueError(f"Expected {STATE_DIM}-D state, got shape {state.shape}")
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": head_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(head_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
        }
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.shape[-1] != ACTION_DIM:
                raise ValueError(f"Expected {ACTION_DIM}-D actions, got shape {actions.shape}")
            inputs["actions"] = actions
        if "prompt" not in data:
            raise ValueError(f"Prompt is required and must be {PROMPT!r}")
        inputs["prompt"] = _validate_prompt(data["prompt"])
        return inputs


@dataclasses.dataclass(frozen=True)
class FetchOutputs(transforms.DataTransformFn):
    """Drop openpi's padded action dimensions for the 13-D Fetch controller."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        if actions.shape[-1] < ACTION_DIM:
            raise ValueError(f"Expected at least {ACTION_DIM} action dimensions, got shape {actions.shape}")
        return {"actions": actions[..., :ACTION_DIM]}
