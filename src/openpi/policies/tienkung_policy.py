import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_tienkung_example(
    *,
    state_dim: int = 16,
    image_hw: tuple[int, int] = (224, 224),
) -> dict:
    """Creates a random input example for the TienKung policy.

    Keys match what ``TienkungInputs`` / ``Policy.infer`` expect after the
    training-only ``RepackTransform``. Real-robot clients should send the same
    intermediate keys (not LeRobot flat keys such as ``observation.state``).

    Common layouts:
      - EVT50: ``state_dim=16``
      - EVT276 reduced hand: ``state_dim=24``
      - EVT276 full hand: ``state_dim=34``
    """
    height, width = image_hw
    return {
        "state": np.random.rand(state_dim).astype(np.float32),
        "image": np.random.randint(256, size=(height, width, 3), dtype=np.uint8),
        "prompt": "move box to the green conveyor",
    }


def _parse_image(image) -> np.ndarray:
    """Parse image to uint8 (H, W, C) format."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class TienkungInputs(transforms.DataTransformFn):
    """
    Converts TienKung dataset / runtime inputs to the model-expected format.

    This transform is used for both training and inference. It does not hardcode
    state/action dimensions; those come from the dataset (train) or robot client
    (infer). Embodiment-specific settings live in ``LeRobotTienkungDataConfig``:
      - EVT50: state/action 16D, image key ``observation.images.head``
      - EVT276 reduced hand: state/action 24D, image key ``observation.images.camera_head``
      - EVT276 full hand: state 34D / action 24D, same camera key as reduced hand

    Intermediate keys expected here (after training ``RepackTransform``, or sent
    directly by a real-robot client):
      - ``image``: (H, W, 3) or (3, H, W) head camera
      - ``state``: (state_dim,) float32
      - ``actions``: (horizon, action_dim) float32, training only
      - ``prompt``: str task instruction

    Model expects:
      - state
      - image: {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
      - image_mask
      - actions (during training)
      - prompt
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Read from intermediate keys (after RepackTransform, or from robot client).
        base_image = _parse_image(data["image"])

        # Pi0/pi05 models support up to 3 image views: base + left wrist + right wrist.
        # TienKung currently uses one head camera, so wrist views are zero-padded.
        inputs = {
            "state": data["state"],
            "image": {
                "base_0_rgb": base_image,
                #"left_wrist_0_rgb": np.zeros_like(base_image),
                #"right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                # Mask out the missing wrist images. For pi0/pi05, non-existent
                # images should be masked with False.
                #"left_wrist_0_rgb": (
                #    np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_
                #),
                #"right_wrist_0_rgb": (
                #    np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_
                #),
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class TienkungOutputs(transforms.DataTransformFn):
    """
    Converts model outputs back to the configured TienKung action space.

    The model produces ``action_dim=32``. Only the leading embodiment dimensions
    are returned. Set ``action_dim`` from ``LeRobotTienkungDataConfig.embodiment_action_dim``
    (16 for EVT50, 24 for EVT276). Default 16 is only a fallback for EVT50.
    """

    action_dim: int = 16

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., : self.action_dim])}


@dataclasses.dataclass(frozen=True)
class FitStateToModelDim(transforms.DataTransformFn):
    """Fit the continuous state tensor to the model interface after tokenization.

    PI0.5 consumes state through the discrete prompt tokens, but its Observation
    interface still requires a tensor whose last dimension equals action_dim.
    This transform is intentionally placed after TokenizePrompt so a state wider
    than the model interface (e.g. EVT276 full-hand 34D) is fully represented in
    the prompt before truncation. Shorter states are left unchanged for
    PadStatesAndActions to zero-pad.
    """

    model_dim: int

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["state"])
        if state.shape[-1] > self.model_dim:
            data["state"] = state[..., : self.model_dim]
        return data
