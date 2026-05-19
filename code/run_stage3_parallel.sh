#!/usr/bin/env bash
# Stage 3 multi-backend parallel launch
#
# 布局（nvidia-smi 实测 2026-05-18）：
#   GPU 0,1,2  → wav2vec2-xls-r-300m  (3进程各跑 ~17 fold)
#   GPU 3,4,5  → mHuBERT-147          (3进程各跑 ~17 fold)
#   GPU 6      → ERR! 风扇/功耗传感器故障，跳过
#   GPU 7      → 预留空闲
#
# 总 folds = 52 → 分段 [0:18), [18:36), [36:52)
#
# 用法（在项目根目录运行）：
#   cd /data/liujialing/TY/建模/EEG与语音联动/EEG
#   bash code/run_stage3_parallel.sh 2>&1 | tee logs/stage3_parallel_$(date +%Y%m%d_%H%M%S).log

set -euo pipefail

# 国内 HuggingFace 镜像（huggingface.co 不通时自动走此处）
export HF_ENDPOINT="https://hf-mirror.com"
export HUGGINGFACE_HUB_VERBOSITY="warning"

PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJ_ROOT}/code:${PYTHONPATH:-}"
CODE_DIR="$PROJ_ROOT/code"
LOG_DIR="$PROJ_ROOT/logs"
RESULTS_DIR="$PROJ_ROOT/results/stage3_speech_eeg_clip"
CONDA_ENV="eeg_analysis"   # ty_eeg_speech_stage1 的 PyTorch 编译为 cu130，驱动不兼容；eeg_analysis cu118 可用

mkdir -p "$LOG_DIR"

# ---------- 公共训练参数（与 clip_whisper_existing_full 保持一致） ----------
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

# ---------- 检查 cache，若不存在则先准备 ----------
W2V2_CACHE="$PROJ_ROOT/cache/stage3_speech_eeg_clip/speech_features/wav2vec2_facebook-wav2vec2-xls-r-300m/win4.000_stride2.000"
HUBERT_CACHE="$PROJ_ROOT/cache/stage3_speech_eeg_clip/speech_features/hubert_utter-project-mHuBERT-147-base-3rd-iter/win4.000_stride2.000"

prepare_cache_if_needed() {
    local backend=$1
    local cache_dir=$2
    local gpu=$3
    if ls "$cache_dir"/*.npz 2>/dev/null | head -1 | grep -q npz; then
        echo "[$(date +%T)] $backend cache already exists, skipping prepare."
    else
        echo "[$(date +%T)] Preparing speech cache for $backend on GPU $gpu ..."
        CUDA_VISIBLE_DEVICES=$gpu conda run -n "$CONDA_ENV" --no-capture-output \
            python "$CODE_DIR/stage3_speech_eeg_clip.py" \
            --speech-backend "$backend" \
            --speech-source cache \
            --prepare-speech-cache \
            --device cuda \
            --require-cuda \
            2>&1 | tee "$LOG_DIR/stage3_${backend}_cache_prep.log"
        echo "[$(date +%T)] $backend cache done."
    fi
}

# Phase 1: 串行准备 cache（每个 backend 只需要一块卡，很快）
echo "===== Phase 1: Prepare speech caches ====="
prepare_cache_if_needed wav2vec2 "$W2V2_CACHE" 0 &
PID_W2V2_CACHE=$!
prepare_cache_if_needed hubert "$HUBERT_CACHE" 3 &
PID_HUBERT_CACHE=$!

wait $PID_W2V2_CACHE; [ $? -eq 0 ] && echo "[$(date +%T)] wav2vec2 cache ready." || { echo "[ERROR] wav2vec2 cache prep failed, aborting."; exit 1; }
wait $PID_HUBERT_CACHE; [ $? -eq 0 ] && echo "[$(date +%T)] HuBERT cache ready." || { echo "[ERROR] HuBERT cache prep failed, aborting."; exit 1; }

# ---------- Phase 2: 训练，fold-level 3并行 × 2 backend = 6进程 ----------
echo ""
echo "===== Phase 2: Training (6 parallel processes) ====="

launch_fold_slice() {
    local backend=$1
    local part=$2       # 0, 1, 2
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

# wav2vec2: GPU 0, 1, 2
launch_fold_slice wav2vec2 0 0  0 18 &
launch_fold_slice wav2vec2 1 1 18 36 &
launch_fold_slice wav2vec2 2 2 36 52 &

# HuBERT: GPU 3, 4, 5
launch_fold_slice hubert 0 3  0 18 &
launch_fold_slice hubert 1 4 18 36 &
launch_fold_slice hubert 2 5 36 52 &

# 等待全部完成
wait
echo ""
echo "===== Phase 3: Merging results ====="

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
    echo "[$(date +%T)] $backend merged → $RESULTS_DIR/clip_${backend}_full"
done

echo ""
echo "===== All done. ====="
echo "Results:"
echo "  $RESULTS_DIR/clip_wav2vec2_full/summary.json"
echo "  $RESULTS_DIR/clip_hubert_full/summary.json"
echo "  Compare with Whisper: $RESULTS_DIR/clip_whisper_existing_full/summary.json"
