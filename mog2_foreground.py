"""Adaptive foreground detection and CDnet-style evaluation for data/highway."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent


def numbered_files(folder: Path, pattern: str) -> dict[int, Path]:
    found: dict[int, Path] = {}
    for path in folder.glob(pattern):
        match = re.search(r"(\d+)", path.stem)
        if match:
            found[int(match.group(1))] = path
    return found


def read_temporal_roi(path: Path) -> tuple[int, int]:
    try:
        values = [int(x) for x in path.read_text(encoding="utf-8").split()]
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read temporal ROI file {path}: {exc}") from exc
    if len(values) < 2 or values[0] < 1 or values[1] < values[0]:
        raise ValueError(f"Expected 'first_frame last_frame' in {path}")
    return values[0], values[1]


def safe_metrics(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run adaptive OpenCV MOG2 foreground detection and evaluate against CDnet masks."
    )
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "highway", help="Sequence folder")
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "mog2-highway")
    parser.add_argument("--start", type=int, help="First frame to evaluate (default: temporal ROI start)")
    parser.add_argument("--end", type=int, help="Last frame to evaluate (default: temporal ROI end)")
    parser.add_argument("--warmup", type=int, default=469, help="Number of frames immediately before evaluation used for warmup")
    parser.add_argument("--history", type=int, default=500, help="MOG2 history length")
    parser.add_argument("--variance-threshold", type=float, default=25.0, help="MOG2 squared-distance threshold")
    parser.add_argument("--learning-rate", type=float, default=-1.0, help="MOG2 learning rate; -1 selects automatic adaptation")
    parser.add_argument("--detect-shadows", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--morphology", choices=("none", "open", "close", "open-close"), default="none",
                        help="Optional binary-mask cleanup; none preserves raw MOG2 output")
    parser.add_argument("--kernel-size", type=int, default=3, help="Odd morphology kernel size")
    parser.add_argument("--video-fps", type=float, default=30.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sequence = args.data.resolve()
    inputs = numbered_files(sequence / "input", "in*")
    truths = numbered_files(sequence / "groundtruth", "gt*")
    if not inputs:
        print(f"Error: no input frames found in {sequence / 'input'}", file=sys.stderr)
        return 2
    if args.warmup < 0 or args.history < 1 or args.video_fps <= 0 or args.kernel_size < 1 or args.kernel_size % 2 == 0:
        print("Error: warmup must be nonnegative; history/FPS/kernel size must be positive, and kernel size must be odd", file=sys.stderr)
        return 2
    temporal_path = sequence / "temporalROI.txt"
    if not temporal_path.exists():
        print(f"Error: temporal ROI file not found: {temporal_path}", file=sys.stderr)
        return 2
    try:
        temporal_start, temporal_end = read_temporal_roi(temporal_path)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    requested_start = args.start if args.start is not None else temporal_start
    requested_end = args.end if args.end is not None else temporal_end
    eval_start = max(requested_start, temporal_start)
    eval_end = min(requested_end, temporal_end)
    if eval_start < 1:
        print("Error: evaluation interval must be positive and start <= end", file=sys.stderr)
        return 2
    if eval_end < eval_start:
        print(f"Error: requested interval does not overlap temporal ROI {temporal_start}-{temporal_end}", file=sys.stderr)
        return 2
    warmup_start = max(1, eval_start - args.warmup)
    frame_numbers = sorted(n for n in inputs if warmup_start <= n <= eval_end)
    if not frame_numbers or eval_start not in inputs:
        print(f"Error: requested evaluation start frame {eval_start} is unavailable", file=sys.stderr)
        return 2
    missing_gt = [n for n in range(eval_start, eval_end + 1) if n in inputs and n not in truths]
    if missing_gt:
        print(f"Error: missing ground-truth masks for frames {missing_gt[:8]}", file=sys.stderr)
        return 2
    roi_path = sequence / "ROI.bmp"
    roi = cv2.imread(str(roi_path), cv2.IMREAD_GRAYSCALE) if roi_path.exists() else None
    if roi is None:
        print(f"Error: could not read ROI mask: {roi_path}", file=sys.stderr)
        return 2

    detector = cv2.createBackgroundSubtractorMOG2(
        history=args.history, varThreshold=args.variance_threshold, detectShadows=args.detect_shadows
    )
    args.output.mkdir(parents=True, exist_ok=True)
    video_writers: dict[str, cv2.VideoWriter] = {}
    rows: list[dict[str, float | int]] = []
    pooled_tp = pooled_fp = pooled_fn = 0
    try:
        for frame_no in frame_numbers:
            frame = cv2.imread(str(inputs[frame_no]), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError(f"Could not read frame {inputs[frame_no]}")
            if frame.shape[:2] != roi.shape:
                raise ValueError(f"ROI size {roi.shape[::-1]} does not match frame {frame_no} size {frame.shape[1::-1]}")
            raw_mask = detector.apply(frame, learningRate=args.learning_rate)
            # OpenCV MOG2 uses 127 for shadow and 255 for foreground; shadows are excluded.
            predicted_mask = np.uint8(raw_mask == 255) * 255
            if args.morphology != "none":
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.kernel_size, args.kernel_size))
                operations = (cv2.MORPH_OPEN, cv2.MORPH_CLOSE) if args.morphology == "open-close" else (
                    cv2.MORPH_OPEN if args.morphology == "open" else cv2.MORPH_CLOSE,
                )
                for operation in operations:
                    predicted_mask = cv2.morphologyEx(predicted_mask, operation, kernel)
            predicted_fg = predicted_mask > 0
            if frame_no < eval_start or frame_no > eval_end:
                continue
            truth_img = cv2.imread(str(truths[frame_no]), cv2.IMREAD_UNCHANGED)
            if truth_img is None:
                raise ValueError(f"Could not read ground-truth mask {truths[frame_no]}")
            if truth_img.ndim == 3:
                truth_img = truth_img[:, :, 0]
            if truth_img.shape != frame.shape[:2]:
                raise ValueError(f"Ground-truth size does not match frame {frame_no}: {truths[frame_no]}")
            roi_valid = roi > 0
            # CDnet labels: 0 background, 50 shadow, 85 unknown, 170/255 foreground.
            valid = roi_valid & np.isin(truth_img, [0, 170, 255])
            actual_fg = np.isin(truth_img, [170, 255])
            tp = int(np.count_nonzero(predicted_fg & actual_fg & valid))
            fp = int(np.count_nonzero(predicted_fg & ~actual_fg & valid))
            fn = int(np.count_nonzero(~predicted_fg & actual_fg & valid))
            row = {"frame": frame_no, **safe_metrics(tp, fp, fn),
                   "valid_pixels": int(np.count_nonzero(valid)),
                   "ignored_shadow_pixels": int(np.count_nonzero(roi_valid & (truth_img == 50))),
                   "ignored_unknown_pixels": int(np.count_nonzero(roi_valid & ~np.isin(truth_img, [0, 50, 170, 255]))),
                   "ignored_outside_roi_pixels": int(np.count_nonzero(~roi_valid))}
            rows.append(row)
            pooled_tp += tp
            pooled_fp += fp
            pooled_fn += fn
            if not video_writers:
                height, width = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                for name, size in (("input", (width, height)), ("foreground", (width, height)),
                                   ("input_mog2", (width * 2, height)),
                                   ("side_by_side", (width * 3, height))):
                    writer = cv2.VideoWriter(str(args.output / f"{name}.avi"), fourcc, args.video_fps, size, isColor=True)
                    if not writer.isOpened():
                        writer.release()
                        raise ValueError(f"Could not create {name}.avi; try a different output path")
                    video_writers[name] = writer
            visual = np.zeros(frame.shape[:2], dtype=np.uint8)
            visual[predicted_fg] = 255
            foreground_bgr = cv2.cvtColor(visual, cv2.COLOR_GRAY2BGR)
            input_bgr = frame.copy()
            cv2.putText(input_bgr, "INPUT", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            labeled_fg = foreground_bgr.copy()
            cv2.putText(labeled_fg, "MOG2 FOREGROUND", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            gt_visual = np.zeros_like(frame)
            gt_visual[truth_img == 50] = (127, 127, 127)
            gt_visual[np.isin(truth_img, [170, 255])] = (255, 255, 255)
            unknown = ~np.isin(truth_img, [0, 50, 170, 255])
            gt_visual[unknown] = (255, 0, 255)
            gt_visual[~roi_valid] = (96, 32, 0)
            cv2.putText(gt_visual, "GROUND TRUTH", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            video_writers["input"].write(frame)
            video_writers["foreground"].write(foreground_bgr)
            video_writers["input_mog2"].write(np.hstack((input_bgr, labeled_fg)))
            video_writers["side_by_side"].write(np.hstack((input_bgr, labeled_fg, gt_visual)))
    except (OSError, ValueError, cv2.error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    finally:
        for writer in video_writers.values():
            writer.release()
    if not rows:
        print("Error: no frames were evaluated", file=sys.stderr)
        return 2
    with (args.output / "per_frame_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "method": "OpenCV MOG2",
        "evaluation_interval": [eval_start, eval_end],
        "temporal_roi": [temporal_start, temporal_end],
        "warmup_frames_before_evaluation": eval_start - warmup_start,
        "requested_warmup_frames": args.warmup,
        "parameters": {"history": args.history, "variance_threshold": args.variance_threshold,
                       "learning_rate": args.learning_rate, "detect_shadows": args.detect_shadows,
                       "morphology": args.morphology, "kernel_size": args.kernel_size},
        "evaluated_frames": len(rows),
        "pooled_metrics": safe_metrics(pooled_tp, pooled_fp, pooled_fn),
        "per_frame_mean": {key: float(np.mean([row[key] for row in rows])) for key in ("precision", "recall", "f1")},
        "label_policy": "FG=170/255; background=0; shadows=50, unknown labels, and outside-ROI pixels ignored",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Results: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
