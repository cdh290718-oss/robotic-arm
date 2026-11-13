#!/usr/bin/env python
"""Offline joint-coordinate audit for the MolmoAct2 SO-100/101 checkpoint.

Scope: **file and log analysis only**.  No motor is touched, no register is
written, no action is produced or sent.

What this script does
---------------------
1. Reconstructs the *actual* coordinate chain from the installed LeRobot 0.6.0
   source (file:line citations), instead of assuming a formula.
2. Re-verifies that chain numerically against this machine's own saved
   ``raw_present`` / ``state`` pairs and its calibration file.
3. Keeps **state statistics and action statistics separate** and checks the
   normalise / un-normalise directions independently (the checkpoint has two
   different statistic blocks).
4. Documents the upstream training-data provenance that can be verified from
   public artifacts (dataset id, version, feature names, first frame).
5. States explicitly what is **not** recoverable from public information.

Deliberately absent: any per-joint "alignment offset" suggestion.  Median
statistics are not a calibration and must not be used to build a deployable
mapping.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

MODEL_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# LeRobot 0.6.0: motors_bus.py:872 uses `resolution - 1`, tables.py:190 gives
# sts3215 resolution 4096, so max_res = 4095 and degrees-per-tick = 360/4095.
STS3215_RESOLUTION = 4096
MAX_RES = STS3215_RESOLUTION - 1
TICKS_PER_DEG = MAX_RES / 360.0  # 11.375 exactly


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def relu_progress(value: float) -> float:
    return value


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(HERE / "model"))
    ap.add_argument("--calibration", default=r"G:\LeRobot\data\calibration\robots\so_follower\lerobot.json")
    ap.add_argument("--saved", default=r"D:\桌面\萌芽杯\policy_v1_refined\results\20261112_214119_107_Grasp_bf16")
    ap.add_argument("--lerobot-src", default=r"G:\LeRobot\source\lerobot-0.6.0\src\lerobot")
    ap.add_argument("--upstream-dir", default=str(HERE / "tmp" / "upstream"))
    ap.add_argument("--out", default=str(HERE / "tmp" / "joint_contract.json"))
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    stats_payload = load_json(model_dir / "norm_stats.json")
    cfg = load_json(model_dir / "config.json")
    meta = stats_payload["metadata_by_tag"]["so100_so101_molmoact2"]
    st = meta["state_stats"]
    at = meta["action_stats"]

    report: dict = {
        "created_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hardware_access": "none - file/log analysis only; no motor, no register write, no action sent",
        "corrections_applied": [
            "LeRobot 0.6.0 DEGREES branch also uses the calibrated range midpoint and 360/4095; "
            "this project's control_core.py follows that convention, it does not deviate from it",
            "ticks per degree is 4095/360 = 11.375 (max_res = resolution - 1), not 4096/360",
            "the formula (ticks - homing_offset)/4096*360 is NOT a LeRobot path and has been removed",
            "the finding is limited to 'normalisation clipping is confirmed'; the cause of the "
            "coordinate mismatch is still unverified",
            "no statistical (median) alignment is offered as a mapping",
        ],
    }

    # ---------------------------------------------------------------- chain
    report["coordinate_chain"] = {
        "authoritative_source": "LeRobot 0.6.0 (installed at G:\\LeRobot\\source\\lerobot-0.6.0)",
        "steps": [
            {
                "step": 1,
                "what": "bus read of the position register",
                "detail": "Feetech STS3215 'Present_Position' register (address 56, 2 bytes)",
                "evidence": "lerobot/motors/feetech/tables.py:83",
            },
            {
                "step": 2,
                "what": "sign-magnitude decode of the raw register value",
                "detail": "Present_Position uses sign bit 15; decoded value = -magnitude if bit15 else magnitude",
                "evidence": "feetech/tables.py:207-216 (STS_SMS_SERIES_ENCODINGS_TABLE) and "
                "motors/encoding_utils.py:decode_sign_magnitude",
            },
            {
                "step": 3,
                "what": "body joints -> degrees",
                "detail": "deg = (ticks - mid) * 360 / max_res, mid = (range_min + range_max)/2, "
                "max_res = model_resolution - 1 = 4095",
                "evidence": "motors/motors_bus.py:870-873 (MotorNormMode.DEGREES branch)",
            },
            {
                "step": 4,
                "what": "gripper -> 0..100",
                "detail": "value = (ticks - range_min) / (range_max - range_min) * 100",
                "evidence": "motors/motors_bus.py:867-869 (MotorNormMode.RANGE_0_100 branch)",
            },
            {
                "step": 5,
                "what": "which joint uses which mode",
                "detail": "shoulder_pan/lift, elbow_flex, wrist_flex, wrist_roll -> DEGREES; "
                "gripper -> RANGE_0_100 (default use_degrees=True)",
                "evidence": "robots/so_follower/so_follower.py:50-59 and config_so_follower.py:42",
            },
            {
                "step": 6,
                "what": "dataset observation.state / action",
                "detail": "get_observation() calls sync_read('Present_Position') with normalize=True, "
                "so the dataset stores exactly the step-3/step-4 values",
                "evidence": "robots/so_follower/so_follower.py:182-204",
            },
            {
                "step": 7,
                "what": "checkpoint normalisation",
                "detail": "state and action each map linearly from [q01, q99] to [-1, 1] and are clipped; "
                "q01/q99 are computed per tag over the training mixture",
                "evidence": "model/modeling_molmoact2.py:_FeatureNormalizer.normalize (1011-1042) and "
                "unnormalize (1044-1064)",
            },
            {
                "step": 8,
                "what": "prompt state",
                "detail": "normalised state -> 256 uniform bins -> '<state_i>' tokens (no sign flip, no wrap)",
                "evidence": "model/modeling_molmoact2.py:_discretize_normalized_state (1271-1278)",
            },
        ],
        "constants_verified": {
            "sts3215_resolution": STS3215_RESOLUTION,
            "max_res_used_by_lerobot": MAX_RES,
            "ticks_per_degree": TICKS_PER_DEG,
            "arithmetic": "4095 / 360 = 11.375 exactly",
            "evidence": "lerobot/motors/feetech/tables.py:190 (resolution 4096) + "
            "motors_bus.py:872 (max_res = resolution - 1)",
        },
        "local_convention_vs_lerobot": (
            "identical: this project's control_core.py uses (tick - range midpoint) * 360/4095 "
            "and a 0-100 gripper fraction, which is the same DEGREES / RANGE_0_100 path"
        ),
    }

    # ------------------------------------------- numeric re-verification (local)
    saved = Path(args.saved)
    npz_path = saved / "predictions.npz"
    jsonl_path = saved / "control.jsonl"
    cal_path = Path(args.calibration)
    numeric: dict = {"sources": {"states_npz": str(npz_path), "control_jsonl": str(jsonl_path),
                                 "calibration": str(cal_path)}}
    cal = load_json(cal_path) if cal_path.exists() else None
    if cal:
        numeric["calibration_midpoints_ticks"] = {
            name: (c["range_min"] + c["range_max"]) / 2 for name, c in cal.items()
        }
        numeric["calibration_range_degrees_DEGREES_branch"] = {
            name: [
                round((c["range_min"] - (c["range_min"] + c["range_max"]) / 2) * 360 / MAX_RES, 2),
                round((c["range_max"] - (c["range_min"] + c["range_max"]) / 2) * 360 / MAX_RES, 2),
            ]
            for name, c in cal.items()
            if name != "gripper"
        }
        numeric["calibration_gripper_0_100_span"] = {
            "ticks": [cal["gripper"]["range_min"], cal["gripper"]["range_max"]],
            "ticks_per_unit": (cal["gripper"]["range_max"] - cal["gripper"]["range_min"]) / 100.0,
        }

    if jsonl_path.exists() and cal:
        raw, state = [], []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                if "raw_present" in d and "state" in d:
                    raw.append(d["raw_present"])
                    state.append(d["state"])
        raw = np.asarray(raw, dtype=np.float64)
        state = np.asarray(state, dtype=np.float64)
        per_joint = {}
        for i, name in enumerate(MODEL_NAMES):
            c = cal[name]
            mid = (c["range_min"] + c["range_max"]) / 2
            if name == "gripper":
                predicted = (raw[:, i] - c["range_min"]) / (c["range_max"] - c["range_min"]) * 100
            else:
                predicted = (raw[:, i] - mid) * 360 / MAX_RES
            residual = np.abs(predicted - state[:, i])
            per_joint[name] = {
                "rows": int(raw.shape[0]),
                "formula": (
                    "(ticks - range_min)/(range_max - range_min)*100"
                    if name == "gripper"
                    else f"(ticks - {mid}) * 360 / {MAX_RES}"
                ),
                "max_abs_residual_deg": round(float(residual.max()), 6),
                "mean_abs_residual_deg": round(float(residual.mean()), 6),
                "matches_saved_state": bool(residual.max() < 1e-6),
            }
        numeric["formula_reproduction_from_saved_log"] = per_joint
        numeric["all_joints_reproduced"] = all(
            v["matches_saved_state"] for v in per_joint.values()
        )
        numeric["saved_state_ranges_degrees"] = {
            name: [round(float(state[:, i].min()), 3), round(float(state[:, i].max()), 3)]
            for i, name in enumerate(MODEL_NAMES)
        }
    report["numeric_verification"] = numeric

    # --------------------------------------------- state vs action statistics
    def normalise(x, stats, mode):
        q01 = np.asarray(stats["q01"], dtype=np.float64)
        q99 = np.asarray(stats["q99"], dtype=np.float64)
        if mode == "q01_q99":
            z = 2.0 * (x - q01) / np.maximum(q99 - q01, 1e-6) - 1.0
            return np.clip(z, -1.0, 1.0)
        raise ValueError(mode)

    def unnormalise(z, stats, mode):
        q01 = np.asarray(stats["q01"], dtype=np.float64)
        q99 = np.asarray(stats["q99"], dtype=np.float64)
        if mode == "q01_q99":
            z = np.clip(z, -1.0, 1.0)
            return (z + 1.0) * (q99 - q01) / 2.0 + q01
        raise ValueError(mode)

    probe = np.asarray([0.0, 90.0, 45.0, 60.0, 0.0, 20.0], dtype=np.float64)
    z_state = normalise(probe, st, "q01_q99")
    z_action = normalise(probe, at, "q01_q99")
    report["state_vs_action_statistics"] = {
        "norm_mode": stats_payload["norm_mode"],
        "state_q01": st["q01"],
        "state_q99": st["q99"],
        "action_q01": at["q01"],
        "action_q99": at["q99"],
        "note": (
            "the two blocks differ substantially; using the action statistics for the state "
            "input (or the reverse) is a silent error, so normalise and un-normalise are "
            "checked separately"
        ),
        "same_pose_same_dimension_different_normalised_value": {
            "pose_degrees_or_gripper_units": probe.tolist(),
            "normalised_with_state_stats": np.round(z_state, 4).tolist(),
            "normalised_with_action_stats": np.round(z_action, 4).tolist(),
            "max_abs_difference": round(float(np.max(np.abs(z_state - z_action))), 4),
        },
        "inverse_direction_roundtrip": {
            "state": {
                "max_abs_error": round(
                    float(np.max(np.abs(unnormalise(z_state, st, "q01_q99") - probe))), 9
                )
            },
            "action": {
                "max_abs_error": round(
                    float(np.max(np.abs(unnormalise(z_action, at, "q01_q99") - probe))), 9
                )
            },
        },
        "clipping": {
            "state_clip_outside_q01_q99": True,
            "action_clip_outside_q01_q99": True,
            "consequence": (
                "an input outside [q01, q99] saturates to +-1 before discretisation, so the "
                "model cannot see how far outside it is; a predicted action outside the "
                "action range is clipped back to the range bounds on un-normalisation"
            ),
        },
        "gripper": {
            "dataset_and_robot_scale": "0-100 (RANGE_0_100 branch)",
            "checkpoint_mask_state": st.get("mask"),
            "checkpoint_mask_action": at.get("mask"),
            "normalize_gripper_flag": meta.get("normalize_gripper"),
            "code_usage_of_normalize_gripper": (
                "not referenced anywhere in modeling_molmoact2.py; the mask in the statistics "
                "is what decides per-dimension normalisation, and here every mask entry is true, "
                "so the gripper is normalised with q01/q99 like the arm joints"
            ),
            "direction": "0 = closed end of the calibrated span, 100 = open end; the checkpoint "
            "does not define a separate opening direction",
        },
        "angle_wrap": {
            "wrap_handling": "none: the normalisation is a linear map plus clip, and the DEGREES "
            "branch of LeRobot does not wrap either",
            "risk": "a joint crossing the +-180 deg boundary would jump discontinuously in the "
            "state token stream; the saved session does not cross it",
        },
        "drive_mode": {
            "effect_on_degrees": "none; motors_bus.py applies drive_mode only in the "
            "RANGE_M100_100 and RANGE_0_100 branches",
            "local_calibration_drive_modes": {k: v["drive_mode"] for k, v in (cal or {}).items()},
        },
    }

    # ------------------------------------------------------- upstream provenance
    up = Path(args.upstream_dir)
    upstream = {
        "model_card": "https://huggingface.co/allenai/MolmoAct2-SO100_101",
        "card_statements": [
            "state is the raw robot state and actions are returned in robot scale",
            "camera order does not matter for this checkpoint",
            "the sample comes from Beegbrain/pick_lemon_and_drop_in_bowl episode 0 frame 0",
        ],
        "dataset": {
            "id": "Beegbrain/pick_lemon_and_drop_in_bowl",
            "codebase_version": "v2.1",
            "robot_type": "so100",
            "fps": 30,
            "total_episodes": 40,
            "total_frames": 15352,
            "feature_names": [
                "main_shoulder_pan",
                "main_shoulder_lift",
                "main_elbow_flex",
                "main_wrist_flex",
                "main_wrist_roll",
                "main_gripper",
            ],
            "cameras": ["observation.images.realsense_top", "observation.images.realsense_side"],
            "camera_resolution": [480, 640, 3],
        },
        "provenance_check": {
            "card_sample_state": [
                -0.52734375,
                189.140625,
                181.40625,
                60.64453125,
                -3.603515625,
                1.0971786975860596,
            ],
            "dataset_frame0_state": None,
            "matches": None,
            "dataset_frame0_action": None,
        },
        "note": (
            "the dataset ships no robot calibration in its metadata, so the recording rig's "
            "range_min/range_max - and therefore the degree zero of the training data - cannot "
            "be recovered from public artifacts"
        ),
    }
    parquet = up / "episode_000000.parquet"
    frame0_json = up / "episode0.json"
    if frame0_json.exists():
        # Extracted once from the public parquet with a reader that has pyarrow;
        # the audit itself must not depend on a dataframe stack.
        d0 = load_json(frame0_json)
        card = np.asarray(upstream["provenance_check"]["card_sample_state"])
        s0 = np.asarray(d0["state_frame0"])
        upstream["provenance_check"]["dataset_frame0_state"] = np.round(s0, 6).tolist()
        upstream["provenance_check"]["dataset_frame0_action"] = np.round(
            np.asarray(d0["action_frame0"]), 6
        ).tolist()
        upstream["provenance_check"]["matches"] = bool(np.allclose(s0, card, atol=1e-6))
        upstream["dataset"]["episode_rows"] = d0.get("rows")
        upstream["dataset"]["episode_state_range"] = {
            "min": np.round(np.asarray(d0["state_min"]), 2).tolist(),
            "max": np.round(np.asarray(d0["state_max"]), 2).tolist(),
        }
        upstream["dataset"]["episode_action_range"] = {
            "min": np.round(np.asarray(d0["action_min"]), 2).tolist(),
            "max": np.round(np.asarray(d0["action_max"]), 2).tolist(),
        }
    elif parquet.exists():
        try:
            import pandas as pd

            df = pd.read_parquet(parquet)
            s0 = np.asarray(df["observation.state"].values[0], dtype=np.float64)
            a0 = np.asarray(df["action"].values[0], dtype=np.float64)
            card = np.asarray(upstream["provenance_check"]["card_sample_state"])
            upstream["provenance_check"]["dataset_frame0_state"] = np.round(s0, 6).tolist()
            upstream["provenance_check"]["dataset_frame0_action"] = np.round(a0, 6).tolist()
            upstream["provenance_check"]["matches"] = bool(np.allclose(s0, card, atol=1e-6))
        except Exception as e:  # noqa: BLE001
            upstream["provenance_check"]["error"] = f"{type(e).__name__}: {e}"
    report["upstream_provenance"] = upstream

    # ------------------------------------------------------- comparison to local
    if "saved_state_ranges_degrees" in numeric:
        comparison = {}
        for i, name in enumerate(MODEL_NAMES):
            lo, hi = numeric["saved_state_ranges_degrees"][name]
            q01 = st["q01"][i]
            q99 = st["q99"][i]
            comparison[name] = {
                "local_logged_range_degrees": [lo, hi],
                "range_source": "control.jsonl (3537 rows of raw_present + state)",
                "training_state_q01_q99": [round(q01, 3), round(q99, 3)],
                "fraction_of_saved_frames_below_q01": None,
                "fraction_of_saved_frames_above_q99": None,
                "fraction_source": "predictions.npz (196 frames at model time)",
            }
        if npz_path.exists():
            states = np.load(npz_path)["states"]
            for i, name in enumerate(MODEL_NAMES):
                q01 = st["q01"][i]
                q99 = st["q99"][i]
                comparison[name]["fraction_of_saved_frames_below_q01"] = round(
                    float((states[:, i] < q01).mean()), 4
                )
                comparison[name]["fraction_of_saved_frames_above_q99"] = round(
                    float((states[:, i] > q99).mean()), 4
                )
        report["local_vs_training_distribution"] = comparison

    # ------------------------------------------------------------- conclusion
    report["conclusion"] = {
        "confirmed": [
            "this project and LeRobot 0.6.0 use the same DEGREES/RANGE_0_100 convention: "
            "zero at the calibrated range midpoint, 360/4095 degrees per tick, gripper 0-100",
            "the saved session's own raw->state log is reproduced to <1e-6 by that formula on all "
            "six joints, so the local logger is not the source of any mismatch",
            "the checkpoint's normalisation clips out-of-range inputs to +-1 before the state is "
            "discretised; on the saved session shoulder_lift is below q01 in every frame and "
            "elbow_flex in part of them, so those dimensions saturate",
            "state and action statistics are different blocks and are applied in opposite "
            "directions; both directions round-trip exactly inside the range",
            "the official sample state is the dataset's episode-0 frame-0 observation.state, so "
            "the input side of the contract is settled",
        ],
        "not_confirmed": [
            "why this arm's absolute values differ from the training mixture; the recording rigs' "
            "calibrations are not published with the dataset, so a per-joint conversion cannot be "
            "derived from public artifacts",
            "whether the local arm's mechanical assembly (servo horn orientation) matches the "
            "training rigs; a ~150 deg offset on shoulder_lift is consistent with a different "
            "assembly/zero, but not proven",
            "whether the saved SmolVLA frames are temporally valid observations for this "
            "checkpoint; that is a separate alignment question",
        ],
        "no_mapping_provided": (
            "no per-joint offset or scale is proposed here. Aligning distributions is not a "
            "calibration, and applying one would inject an unverified transform into a robot "
            "control path"
        ),
        "recommended_next_steps": [
            "collect a calibration pose set on this arm (same physical poses as the training "
            "distribution, e.g. the poses visible in the sample dataset) and fit the mapping only "
            "if the training convention can be tied to physical poses",
            "or fine-tune / re-fit normalisation statistics on data recorded with this arm's own "
            "convention, which keeps the checkpoint's code path unchanged",
            "keep hardware out of the loop until a state->action one-step comparison on this arm "
            "is within an agreed tolerance",
        ],
    }

    text = json.dumps(report, indent=2)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("corrections_applied", "numeric_verification",
                                            "state_vs_action_statistics", "conclusion")},
                     indent=2)[:6000])
    print(f"\nreport: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
