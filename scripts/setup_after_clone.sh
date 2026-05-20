#!/usr/bin/env bash
# 克隆仓库后一键准备运行环境（不含 35GB 脑电，需另执行 fetch_eeg_data.sh）
set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJ_ROOT"
export PYTHONPATH="${PROJ_ROOT}/code:${PYTHONPATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

CONDA_ENV="${CONDA_ENV:-eeg_analysis}"

echo "===== 1. Python 依赖（conda 环境: $CONDA_ENV）====="
conda run -n "$CONDA_ENV" python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" 2>/dev/null || {
  echo "[WARN] 请先创建环境: conda env create -f environment.yml"
}

for pkg in transformers librosa soundfile; do
  conda run -n "$CONDA_ENV" python -c "import ${pkg//-/_}" 2>/dev/null || \
    conda run -n "$CONDA_ENV" pip install -q "$pkg"
done

echo ""
echo "===== 2. 检查仓库内数据 ====="
for d in experiments cache data/stage1_audio_whisper_embeddings_v2; do
  if [ -d "$PROJ_ROOT/$d" ]; then
    echo "  OK  $d"
  else
    echo "  MISSING $d"
  fi
done

if [ -d "$PROJ_ROOT/data/eeg_corrected" ] && ls "$PROJ_ROOT/data/eeg_corrected"/*.pkl &>/dev/null; then
  echo "  OK  data/eeg_corrected (*.pkl)"
else
  echo "  MISSING data/eeg_* — 请运行: bash scripts/fetch_eeg_data.sh"
fi

echo ""
echo "===== 3. Dry-run（需已有脑电 pkl）====="
if [ -d "$PROJ_ROOT/data/eeg_original" ] && ls "$PROJ_ROOT/data/eeg_original"/*.pkl &>/dev/null; then
  EEG_ARG=(--eeg-root "$PROJ_ROOT/data/eeg_original")
elif [ -d "$PROJ_ROOT/data/eeg_corrected" ] && ls "$PROJ_ROOT/data/eeg_corrected"/*.pkl &>/dev/null; then
  EEG_ARG=(--eeg-root "$PROJ_ROOT/data/eeg_corrected")
else
  echo "跳过 dry-run（无脑电 pkl）"
  exit 0
fi

conda run -n "$CONDA_ENV" python code/stage3_speech_eeg_clip.py \
  "${EEG_ARG[@]}" \
  --speech-backend whisper --speech-source existing-whisper \
  --subjects 002 --video-ids 1 --max-folds 1 --dry-run --run-name _setup_smoke

echo ""
echo "===== 完成 ====="
