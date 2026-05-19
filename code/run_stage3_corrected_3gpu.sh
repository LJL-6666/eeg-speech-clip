#!/usr/bin/env bash
# 修正版 EEG 数据 + 三语音基座（Whisper / Wav2Vec2 / HuBERT）
# 建议：等当前 EEG/code 下原版数据的 wav2vec2+hubert 跑完、内存释放后再启动。
# 用法：bash code/run_stage3_corrected_3gpu.sh

set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONUNBUFFERED=1

PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJ_ROOT}/code:${PYTHONPATH:-}"

CODE_DIR="$PROJ_ROOT/code"
LOG_DIR="$PROJ_ROOT/logs"
RESULTS_DIR="$PROJ_ROOT/results"
CACHE_DIR="$PROJ_ROOT/cache"
EEG_ROOT="$PROJ_ROOT/data/eeg_corrected"
CONDA_ENV="${CONDA_ENV:-eeg_analysis}"

mkdir -p "$LOG_DIR"

if [ ! -d "$EEG_ROOT" ]; then
  echo "[ERROR] 修正数据目录不存在: $EEG_ROOT"
  exit 1
fi

COMMON_ARGS=(
  --eeg-root "$EEG_ROOT"
  --split-unit material
  --window-sec 4.0
  --stride-sec 2.0
  --eeg-target-fs 250.0
  --epochs 30
  --batch-size 64
  --lr 1e-3
  --weight-decay 0.0
  --patience 5
  --min-delta 1e-5
  --d-proj 128
  --init-temperature 0.07
  --require-cuda
  --output-root "$RESULTS_DIR"
)

run_backend() {
  local backend=$1
  local gpu=$2
  local source=$3
  local run_name="clip_${backend}_corrected_full"
  local log_file="$LOG_DIR/stage3_${backend}_corrected_full.log"
  echo "[$(date +%T)] START $backend (eeg_corrected) on GPU $gpu → $log_file"
  CUDA_VISIBLE_DEVICES=$gpu conda run -n "$CONDA_ENV" --no-capture-output \
    python "$CODE_DIR/stage3_speech_eeg_clip.py" \
    --speech-backend "$backend" \
    --speech-source "$source" \
    --run-name "$run_name" \
    --device cuda \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee "$log_file"
  echo "[$(date +%T)] DONE $backend"
}

echo "===== Phase 1: 语音特征缓存（wav2vec2 + hubert，如已有可跳过）====="
for backend in wav2vec2 hubert; do
  log="$LOG_DIR/stage3_${backend}_corrected_cache_prep.log"
  echo "[$(date +%T)] prepare cache: $backend"
  CUDA_VISIBLE_DEVICES=0 conda run -n "$CONDA_ENV" --no-capture-output \
    python "$CODE_DIR/stage3_speech_eeg_clip.py" \
    --speech-backend "$backend" \
    --speech-source cache \
    --eeg-root "$EEG_ROOT" \
    --prepare-speech-cache \
    --split-unit material \
    --window-sec 4.0 \
    --stride-sec 2.0 \
    2>&1 | tee "$log"
done

echo ""
echo "===== Phase 2: 三基座训练（各 1 卡；内存不足时改为串行）====="
run_backend whisper   0 existing-whisper &
PID_W=$!
run_backend wav2vec2  1 cache &
PID_V=$!
run_backend hubert    2 cache &
PID_H=$!

wait $PID_W; RC_W=$?
wait $PID_V; RC_V=$?
wait $PID_H; RC_H=$?

[ $RC_W -eq 0 ] || { echo "[ERROR] whisper corrected failed"; exit 1; }
[ $RC_V -eq 0 ] || { echo "[ERROR] wav2vec2 corrected failed"; exit 1; }
[ $RC_H -eq 0 ] || { echo "[ERROR] hubert corrected failed"; exit 1; }

echo ""
echo "===== 完成；结果目录: $RESULTS_DIR ====="
