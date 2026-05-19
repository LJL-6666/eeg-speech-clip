#!/usr/bin/env bash
# Stage 3 safe launch: 1 process per backend, 2 processes total
# RAM estimate: 120 subjects × 300MB pkl × 2 procs ≈ 72GB  (safe on 125GB machine)
# GPU: wav2vec2 → GPU 0 ;  HuBERT → GPU 3  (GPU 6 ERR! skipped, GPU 1-5,7 idle)

set -euo pipefail

export HF_ENDPOINT="https://hf-mirror.com"
export PYTHONUNBUFFERED=1       # 让 Python 日志实时写出，不等缓冲区满

# 跨模态对比学习项目根目录
PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJ_ROOT}/code:${PYTHONPATH:-}"
CODE_DIR="$PROJ_ROOT/code"
LOG_DIR="$PROJ_ROOT/logs"
RESULTS_DIR="$PROJ_ROOT/results/stage3_speech_eeg_clip"
CONDA_ENV="eeg_analysis"

mkdir -p "$LOG_DIR"

COMMON_ARGS=(
    --split-unit material
    --window-sec 4.0
    --stride-sec 2.0
    --eeg-target-fs 250.0
    --epochs 30
    --batch-size 64          # 降低 batch size 进一步控制 GPU 显存峰值
    --lr 1e-3
    --weight-decay 0.0
    --patience 5
    --min-delta 1e-5
    --d-proj 128
    --init-temperature 0.07
    --speech-source cache
    --require-cuda
    --output-root "$RESULTS_DIR"
    --overwrite
)

run_backend() {
    local backend=$1
    local gpu=$2
    local run_name="clip_${backend}_full"
    local log_file="$LOG_DIR/stage3_${backend}_full.log"
    echo "[$(date +%T)] START $backend on GPU $gpu → $log_file"
    CUDA_VISIBLE_DEVICES=$gpu conda run -n "$CONDA_ENV" --no-capture-output \
        python "$CODE_DIR/stage3_speech_eeg_clip.py" \
        --speech-backend "$backend" \
        --run-name "$run_name" \
        --device cuda \
        "${COMMON_ARGS[@]}" \
        2>&1 | tee "$log_file"
    echo "[$(date +%T)] DONE $backend"
}

echo "===== Launching 2 backends in parallel ====="
run_backend wav2vec2 0 &
PID_W=$!
run_backend hubert   3 &
PID_H=$!

wait $PID_W
RC_W=$?
wait $PID_H
RC_H=$?

[ $RC_W -eq 0 ] || { echo "[ERROR] wav2vec2 failed (rc=$RC_W)"; exit 1; }
[ $RC_H -eq 0 ] || { echo "[ERROR] HuBERT failed (rc=$RC_H)"; exit 1; }

echo ""
echo "===== Results summary ====="
for backend in whisper_existing wav2vec2 hubert; do
    f="$RESULTS_DIR/clip_${backend}_full/summary.json"
    [ -f "$f" ] && echo "$backend: $(conda run -n $CONDA_ENV python -c "
import json; d=json.load(open('$f'))
print('S->E top1=%.1f%%  E->S top1=%.1f%%  S->E top5=%.1f%%' % (
    d.get('mean_speech_to_eeg_top1',0)*100,
    d.get('mean_eeg_to_speech_top1',0)*100,
    d.get('mean_speech_to_eeg_top5',0)*100))" 2>/dev/null || echo "(parse error)")"
done
