#!/usr/bin/env python
"""Analyse a live read-only session: synchronisation, clipping, target delta, range, continuity.

Inputs
------
* a ``live_predict*`` run directory (``samples.jsonl`` + ``report.json``)
* the capture spool (``capture_status.json`` + per-sample ``meta.json``)
* the robot calibration file

Outputs a single JSON with the numbers the report needs, and never touches
hardware.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MOTORS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
MAX_RES = 4095  # LeRobot: resolution 4096 - 1


def pct(values, q):
    s = sorted(values)
    if not s:
        return None
    idx = max(0, min(len(s) - 1, int(np.ceil(q * len(s))) - 1))
    return round(float(s[idx]), 4)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="live_predict run directory")
    ap.add_argument("--spool", required=True, help="capture spool directory")
    ap.add_argument("--model-dir", default=str(HERE / "model"))
    ap.add_argument("--calibration", default=r"G:\LeRobot\data\calibration\robots\so_follower\lerobot.json")
    ap.add_argument("--out", default=str(HERE / "tmp" / "live_analysis.json"))
    args = ap.parse_args()

    run_dir = Path(args.run)
    spool = Path(args.spool)
    model_dir = Path(args.model_dir)
    cal = json.loads(Path(args.calibration).read_text(encoding="utf-8-sig"))
    norm = json.loads((model_dir / "norm_stats.json").read_text(encoding="utf-8"))
    st = norm["metadata_by_tag"]["so100_so101_molmoact2"]["state_stats"]
    at = norm["metadata_by_tag"]["so100_so101_molmoact2"]["action_stats"]
    q01 = np.asarray(st["q01"], dtype=np.float64)
    q99 = np.asarray(st["q99"], dtype=np.float64)

    rows = [json.loads(line) for line in (run_dir / "samples.jsonl").read_text("utf-8").splitlines() if line.strip()]
    capture_status = json.loads((spool / "capture_status.json").read_text("utf-8"))
    metas = []
    for d in sorted(spool.glob("sample_*")):
        try:
            metas.append(json.loads((d / "meta.json").read_text("utf-8")))
        except Exception:  # noqa: BLE001
            continue

    report: dict = {
        "run_dir": str(run_dir),
        "spool": str(spool),
        "capture": {
            "status": capture_status.get("status"),
            "samples_captured": capture_status.get("samples"),
            "blocked_packets": capture_status.get("blocked_packets"),
            "torque_before": capture_status.get("torque_before"),
            "torque_after": capture_status.get("torque_after"),
            "torque_unchanged": capture_status.get("torque_unchanged"),
            "allowed_packets": capture_status.get("allowed_packets"),
            "cameras": capture_status.get("cameras"),
        },
        "predictions": len(rows),
        "hardware_command": "none - read-only capture; no goal position, torque, gain or calibration write",
    }

    # ---- synchronisation
    if metas:
        skew = [abs(m["image_state_skew_ms"]) for m in metas]
        view = [abs(m["view_skew_ms"]) for m in metas]
        report["synchronisation"] = {
            "samples": len(metas),
            "image_state_skew_ms": {
                "median": round(statistics.median(skew), 3),
                "p95": pct(skew, 0.95),
                "max": round(max(skew), 3),
            },
            "left_right_view_skew_ms": {
                "median": round(statistics.median(view), 3),
                "max": round(max(view), 3),
            },
            "capture_rate_hz": round(
                len(metas) / max(metas[-1]["elapsed_s"], 1e-6), 3
            ),
        }

    # ---- state: clipping and coverage
    states = np.asarray([r["state"] for r in rows], dtype=np.float64)
    clipped = np.asarray([r["state_clipped"] for r in rows], dtype=bool)
    z = np.asarray([r["state_normalized"] for r in rows], dtype=np.float64)
    per_joint = {}
    for i, name in enumerate(MOTORS):
        per_joint[name] = {
            "state_range_deg": [round(float(states[:, i].min()), 3), round(float(states[:, i].max()), 3)],
            "training_state_q01_q99": [round(float(q01[i]), 3), round(float(q99[i]), 3)],
            "normalized_range": [round(float(z[:, i].min()), 4), round(float(z[:, i].max()), 4)],
            "clipped_fraction": round(float(clipped[:, i].mean()), 4),
            "clipped_in_all_samples": bool(clipped[:, i].all()),
        }
    report["state_clipping"] = {
        "samples": len(rows),
        "note": "clipped means the normalised value saturated at +-1, i.e. outside the training q01..q99",
        "per_joint": per_joint,
        "clipped_dimensions_per_sample": {
            "median": statistics.median(int(c.sum()) for c in clipped),
            "max": int(clipped.sum(axis=1).max()),
        },
    }

    # ---- first target vs measured position
    first = np.asarray([r["first_action"] for r in rows], dtype=np.float64)
    delta = first - states
    report["first_target_vs_measured"] = {
        "units": "degrees for the five arm joints, 0-100 for the gripper",
        "per_joint": {
            name: {
                "mean_delta": round(float(delta[:, i].mean()), 3),
                "std_delta": round(float(delta[:, i].std()), 3),
                "min_delta": round(float(delta[:, i].min()), 3),
                "max_delta": round(float(delta[:, i].max()), 3),
                "mean_abs_delta": round(float(np.abs(delta[:, i]).mean()), 3),
                "max_abs_delta": round(float(np.abs(delta[:, i]).max()), 3),
            }
            for i, name in enumerate(MOTORS)
        },
        "max_abs_delta_overall": round(float(np.abs(delta).max()), 3),
        "note": "the arm was stationary with torque disabled, so this is a one-pose comparison",
    }

    # ---- do predicted targets stay inside the calibrated travel?
    ranges = {}
    for name in MOTORS:
        c = cal[name]
        if name == "gripper":
            ranges[name] = (0.0, 100.0)
        else:
            mid = (c["range_min"] + c["range_max"]) / 2
            ranges[name] = (
                (c["range_min"] - mid) * 360 / MAX_RES,
                (c["range_max"] - mid) * 360 / MAX_RES,
            )
    all_chunks = np.asarray([r["chunk"] for r in rows], dtype=np.float64)  # (n, 30, 6)
    outside = np.zeros((all_chunks.shape[0], 30, 6), dtype=bool)
    for i, name in enumerate(MOTORS):
        lo, hi = ranges[name]
        outside[:, :, i] = (all_chunks[:, :, i] < lo) | (all_chunks[:, :, i] > hi)
    report["travel_range_check"] = {
        "calibrated_ranges": {k: [round(v[0], 2), round(v[1], 2)] for k, v in ranges.items()},
        "predicted_action_values": int(all_chunks.size),
        "fraction_outside_calibrated_range": round(float(outside.mean()), 5),
        "per_joint_outside_fraction": {
            name: round(float(outside[:, :, i].mean()), 5) for i, name in enumerate(MOTORS)
        },
        "per_joint_worst_excursion": {
            name: round(
                float(
                    np.maximum(
                        0.0,
                        np.maximum(ranges[name][0] - all_chunks[:, :, i], all_chunks[:, :, i] - ranges[name][1]),
                    ).max()
                ),
                3,
            )
            for i, name in enumerate(MOTORS)
        },
    }

    # ---- action-range saturation (how often the un-normaliser clips the chunk)
    aq01 = np.asarray(at["q01"], dtype=np.float64)
    aq99 = np.asarray(at["q99"], dtype=np.float64)
    action_z = 2.0 * (all_chunks - aq01) / (aq99 - aq01) - 1.0
    report["action_range_saturation"] = {
        "fraction_at_or_below_minus1": round(float((action_z <= -0.999).mean()), 4),
        "fraction_at_or_above_plus1": round(float((action_z >= 0.999).mean()), 4),
        "per_joint_fraction_saturated": {
            name: round(
                float(
                    ((action_z[:, :, i] <= -0.999) | (action_z[:, :, i] >= 0.999)).mean()
                ),
                4,
            )
            for i, name in enumerate(MOTORS)
        },
        "note": "the checkpoint clips actions to the action q01..q99 range on un-normalisation",
    }

    # ---- continuity across consecutive predictions
    if len(first) > 1:
        step = np.abs(np.diff(first, axis=0))
        report["continuity"] = {
            "samples": len(first),
            "first_target_std": {
                name: round(float(first[:, i].std()), 4) for i, name in enumerate(MOTORS)
            },
            "consecutive_delta": {
                name: {
                    "mean_abs": round(float(step[:, i].mean()), 4),
                    "max_abs": round(float(step[:, i].max()), 4),
                }
                for i, name in enumerate(MOTORS)
            },
            "max_consecutive_jump_overall": round(float(step.max()), 4),
            "note": (
                "arm and scene were static during the session, so this measures repeatability "
                "under identical observations; it does not test continuity under motion"
            ),
        }

    # ---- latency
    lat = [r["latency_ms"] for r in rows]
    if lat:
        report["latency_ms"] = {
            "median": round(statistics.median(lat), 1),
            "p95": pct(lat, 0.95),
            "max": round(max(lat), 1),
            "min": round(min(lat), 1),
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2)
    out.write_text(text, encoding="utf-8")
    print(text[:7000])
    print(f"\nreport: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
