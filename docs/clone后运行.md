# 克隆到任意机器后的运行说明

## 仓库里已包含（clone 即有）

| 内容 | 约大小 | 用途 |
|------|--------|------|
| `code/` | <1MB | Stage3 训练与评估 |
| `experiments/` | 12KB | video → speaker+emotion 映射 |
| `data/stage1_audio_whisper_embeddings_v2/` | **1.1GB** | Whisper 后端（existing-whisper） |
| `cache/speech_features/` | **26MB** | Wav2Vec2 / HuBERT 帧级缓存 |
| `results_published/` | 30MB | Whisper 全量 run 的指标摘要 |
| `data/qc/`、`data/corrected_subjects.txt` | 很小 | 修正被试清单与 QC |

## 不在 Git 内（需额外获取）

| 内容 | 约大小 | 原因 |
|------|--------|------|
| `data/eeg_original/`、`data/eeg_corrected/` | 各 **35GB** | 单 pkl ~300MB，超过 GitHub 100MB/文件限制 |

### 获取脑电数据

```bash
# 方式 A：从实验室服务器 rsync（推荐）
bash scripts/fetch_eeg_data.sh \
  --from-server USER@HOST:/path/to/communication_corrected_20260519

# 方式 B：GitHub Release 分卷（维护者先 pack_eeg_for_release.sh 并上传）
bash scripts/pack_eeg_for_release.sh --variant corrected   # 在数据服务器执行一次
# 上传 data/release_staging/corrected/*.tar.gz 到 Release
# 下载到同目录后：
bash scripts/fetch_eeg_data.sh --from-release corrected
```

## 一键环境检查

```bash
conda env create -f environment.yml   # 或沿用已有 eeg_analysis
bash scripts/setup_after_clone.sh
```

## 训练示例

```bash
export PYTHONPATH="$(pwd)/code:$PYTHONPATH"
export HF_ENDPOINT=https://hf-mirror.com

conda run -n eeg_analysis python code/stage3_speech_eeg_clip.py \
  --eeg-root data/eeg_corrected \
  --speech-backend whisper \
  --speech-source existing-whisper \
  --run-name clip_whisper_corrected_full \
  --device cuda --require-cuda
```

Wav2Vec2 / HuBERT 使用仓库内 `cache/`，无需重新提特征（除非改窗口参数）。
