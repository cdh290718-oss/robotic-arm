"""Official MolmoAct2 legacy <-> LeRobot >=0.5 SO100/101 frame bridge.

Pure numerical adapter: no camera, serial, calibration or model-stat writes.
Use ONLY for the original MolmoAct2 checkpoint with a raw inference path.
Do not apply twice to converted LeRobot checkpoints/processors.
Source: huggingface/lerobot, processor_molmoact2.py,
MolmoAct2StateFrameTransformStep and MolmoAct2ActionFrameTransformStep.
"""
import numpy as np

NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
SIGNS = np.array([1, -1, 1, 1, 1, 1], dtype=np.float64)
OFFSETS = np.array([0, 90, 90, 0, 0, 0], dtype=np.float64)


def _six(values):
    a = np.asarray(values, dtype=np.float64)
    if a.ndim < 1 or a.shape[-1] != 6 or not np.isfinite(a).all():
        raise ValueError("Expected finite (..., 6) joint values; gripper in percentage points")
    return a


def arm_to_model(values):
    """Apply BEFORE model state normalization: model = signs * arm + offsets."""
    return _six(values) * SIGNS + OFFSETS


def model_to_arm(values):
    """Apply AFTER action unnormalization: arm = signs * (model - offsets)."""
    return (_six(values) - OFFSETS) * SIGNS


def action_bounds_in_arm(q01, q99):
    a, b = model_to_arm(q01), model_to_arm(q99)
    return np.minimum(a, b), np.maximum(a, b)
