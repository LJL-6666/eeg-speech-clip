#!/usr/bin/env bash
# Phase 2 only: speech cache already built, directly launch fold-parallel training
# GPU 0,1,2 → wav2vec2 ; GPU 3,4,5 → HuBERT ; GPU 6(ERR!)/7 unused

set -euo pipefail

export HF_ENDPOINT="https://hf-mirror.com"

PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJ_ROOT}/code:${PYTHONPATH:-}"
CODE_DIR="$PROJ_ROOT/code"
LOG_DIR="$PROJ_ROOT/logs"
RESULTS_DIR="$PROJ_ROOT/results/stage3_speech_eeg_clip"
CONDA_ENV="eeg_analysis"

mkdir -p "$LOG_DIR"

COMMON_TRAIN_ARGS=(
    --split-unit material
    --window-sec 4.0
    --stride-sec 2.0
    --eeg-target-fs 250.0
    --epochs 30
    --batch-size 128
    --lr 1e-3
    --weight-decay 0.0
    --patience 5
    --min-delta 1e-5
    --d-proj 128
    --init-temperature 0.07
    --speech-source cache
    --require-cuda
)

launch_fold_slice() {
    local backend=$1
    local part=$2
    local gpu=$3
    local fold_start=$4
    local fold_end=$5
    local run_name="clip_${backend}_p${part}"
    local log_file="$LOG_DIR/stage3_${backend}_p${part}.log"

    echo "[$(date +%T)] Starting $run_name on GPU $gpu  folds [$fold_start:$fold_end)"
    CUDA_VISIBLE_DEVICES=$gpu conda run -n "$CONDA_ENV" --no-capture-output \
        python "$CODE_DIR/stage3_speech_eeg_clip.py" \
        --speech-backend "$backend" \
        --run-name "$run_name" \
        --fold-start "$fold_start" \
        --fold-end   "$fold_end" \
        --device cuda \
        "${COMMON_TRAIN_ARGS[@]}" \
        --output-root "$RESULTS_DIR" \
        --overwrite \
        2>&1 | tee "$log_file"
    echo "[$(date +%T)] $run_name done."
}

echo "===== Training: 6 parallel processes ====="

launch_fold_slice wav2vec2 0 0  0 18 &
launch_fold_slice wav2vec2 1 1 18 36 &
launch_fold_slice wav2vec2 2 2 36 52 &
launch_fold_slice hubert   0 3  0 18 &
launch_fold_slice hubert   1 4 18 36 &
launch_fold_slice hubert   2 5 36 52 &

wait
echo ""
echo "===== Merging results ====="

for backend in wav2vec2 hubert; do
    echo "[$(date +%T)] Merging $backend ..."
    conda run -n "$CONDA_ENV" --no-capture-output \
        python "$CODE_DIR/stage3_merge_folds.py" \
        --input-dirs \
            "$RESULTS_DIR/clip_${backend}_p0" \
            "$RESULTS_DIR/clip_${backend}_p1" \
            "$RESULTS_DIR/clip_${backend}_p2" \
        --output-dir "$RESULTS_DIR/clip_${backend}_full" \
        --overwrite \
        2>&1 | tee "$LOG_DIR/stage3_${backend}_merge.log"
    echo "[$(date +%T)] $backend merged."
done

echo ""
echo "===== All done ====="
for backend in whisper wav2vec2 hubert; do
    f="$RESULTS_DIR/clip_${backend}_existing_full/summary.json"
    [ -f "$f" ] || f="$RESULTS_DIR/clip_${backend}_full/summary.json"
    [ -f "$f" ] && echo "$backend: $(python -c "import json,sys; d=json.load(open('$f')); print('S->E top1=%.1f%%' % (d.get('mean_speech_to_eeg_top1',0)*100), 'E->S top1=%.1f%%' % (d.get('mean_eeg_to_speech_top1',0)*100))" 2>/dev/null || echo "(no summary)")"
done
