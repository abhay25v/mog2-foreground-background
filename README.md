# Adaptive Foreground Detection in Dynamic Environments

A small, runnable academic baseline using OpenCV MOG2 on the included CDnet-style `highway` sequence. It is independent of the repository's original LTS model implementation.

## Windows setup and run

Use Python 3.10 or newer. From PowerShell in this repository:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-mog2.txt
python mog2_foreground.py
```

The default run warms MOG2 on frames 1–469, then evaluates frames 470–1700 from `temporalROI.txt`, applying `ROI.bmp` to spatial evaluation. Results are saved under `output/mog2-highway/`:

- `input.avi`: source frames in the evaluation interval
- `foreground.avi`: binary MOG2 foreground output
- `input_mog2.avi`: input and MOG2 prediction in two panels
- `side_by_side.avi`: input, detected foreground, and ground truth in three panels
- `per_frame_metrics.csv`: per-frame counts and metrics
- `summary.json`: pooled metrics and run parameters

Open AVI files with VLC or Windows Media Player. Run `python mog2_foreground.py --help` for options.

## Tune then evaluate on a held-out interval

For a more careful experiment, tune only on frames 470–1084, then freeze the
settings and score frames 1085–1700. The held-out labels are never used during
parameter selection. During the second run, MOG2 still processes frames 1–1084
without scoring them so it can adapt before evaluation.

```powershell
# Parameter selection on the first interval (keep outputs separate)
python mog2_foreground.py --start 470 --end 1084 --warmup 469 --variance-threshold 16 --output output\mog2-tune-vt16
python mog2_foreground.py --start 470 --end 1084 --warmup 469 --variance-threshold 25 --output output\mog2-tune-vt25

# Locked evaluation on the later interval
python mog2_foreground.py --start 1085 --end 1700 --warmup 1084 --variance-threshold 25 --morphology none --output output\mog2-highway-heldout
```

For the included highway sequence, the tuning interval selected variance
threshold 25 over 16 by pooled F1 (0.694 vs 0.631). With the selected settings
frozen, the held-out interval produced pooled precision 0.951, recall 0.582,
and F1 0.722. This is a temporal holdout within one sequence, not an
independent test on another scene; MOG2 also adapts online using the frames it
processes.

Optional `--morphology open`, `close`, or `open-close` applies a small binary
mask cleanup. On the tuning interval, a 3×3 opening reduced F1 from 0.694 to
0.686, while open-close raised it slightly to 0.699 but reduced recall from
0.627 to 0.576. The locked held-out run therefore keeps the simpler raw MOG2
mask (`none`).

## Method and evaluation

MOG2 is an adaptive, pixel-wise Gaussian-mixture background model. It updates its per-pixel appearance model over time and marks observations that do not fit the background as foreground. The current defaults are history 500, variance threshold 25, automatic learning rate, and shadow detection enabled. Only MOG2 mask value 255 is treated as predicted foreground.

Ground-truth labels 170 and 255 count as foreground; 0 counts as background. Shadow label 50, unknown labels such as 85, and pixels outside the ROI are excluded. The program reports precision, recall, and F1 per frame and pooled across all evaluated pixels.

## Scope and contribution

The dataset folder contains 1,700 input frames and 1,700 matching masks, but only one highway scene. The selected parameters were compared on this same sequence, so the measured improvement is sample-specific, not independent evidence of generalization. MOG2 remains vulnerable to camera motion, abrupt lighting changes, camouflage, and dynamic backgrounds.

This project does not introduce MOG2. Its contribution is a reproducible baseline application to the included data, with warmup, ROI-aware and label-aware evaluation, video comparisons, metrics, and student-facing documentation. The original LTS code, weights, notebook, and outputs have been removed from this working tree; they remain in the Git history.
