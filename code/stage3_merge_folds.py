"""
Merge partial fold results produced by fold-level parallel Stage 3 runs.

Usage:
    python code/stage3_merge_folds.py \
        --input-dirs results/stage3_speech_eeg_clip/clip_wav2vec2_p0 \
                     results/stage3_speech_eeg_clip/clip_wav2vec2_p1 \
                     results/stage3_speech_eeg_clip/clip_wav2vec2_p2 \
        --output-dir results/stage3_speech_eeg_clip/clip_wav2vec2_full
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge partial Stage-3 fold results.")
    p.add_argument("--input-dirs", nargs="+", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir: Path = args.output_dir
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{out_dir} exists and is not empty; use --overwrite.")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: list[dict] = []
    fold_names_seen: set[str] = set()

    for src in args.input_dirs:
        src = Path(src)
        metrics_file = src / "fold_metrics.csv"
        if not metrics_file.exists():
            print(f"[WARN] No fold_metrics.csv in {src}, skipping.")
            continue
        rows = read_csv(metrics_file)
        for row in rows:
            fold = row.get("fold", "")
            if fold in fold_names_seen:
                print(f"[WARN] Duplicate fold '{fold}' from {src}, skipping.")
                continue
            fold_names_seen.add(fold)
            all_metrics.append(row)

        # copy per-fold subdirs
        folds_src = src / "folds"
        if folds_src.exists():
            folds_dst = out_dir / "folds"
            folds_dst.mkdir(exist_ok=True)
            for fold_dir in folds_src.iterdir():
                if fold_dir.is_dir():
                    dst = folds_dst / fold_dir.name
                    if dst.exists() and not args.overwrite:
                        print(f"[WARN] {dst} already exists, skipping.")
                        continue
                    if dst.exists():
                        shutil.rmtree(dst)
                    shutil.copytree(fold_dir, dst)

    # sort by fold name for reproducibility
    all_metrics.sort(key=lambda r: r.get("fold", ""))
    write_csv(out_dir / "fold_metrics.csv", all_metrics)
    print(f"Merged {len(all_metrics)} folds → {out_dir / 'fold_metrics.csv'}")

    # recompute summary.json
    numeric_summary: dict[str, float] = {}
    if all_metrics:
        keys = [k for k, v in all_metrics[0].items() if k != "fold"]
        for key in keys:
            values = []
            for row in all_metrics:
                try:
                    values.append(float(row[key]))
                except (ValueError, KeyError):
                    pass
            if values:
                numeric_summary[f"mean_{key}"] = float(np.nanmean(values))
                numeric_summary[f"median_{key}"] = float(np.nanmedian(values))

    (out_dir / "summary.json").write_text(
        json.dumps(numeric_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Summary → {out_dir / 'summary.json'}")

    # copy first run_config.json as reference
    for src in args.input_dirs:
        rc = Path(src) / "run_config.json"
        if rc.exists():
            shutil.copy(rc, out_dir / "run_config_ref.json")
            break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
