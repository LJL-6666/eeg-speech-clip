# 跨模态对比学习（Speech–EEG CLIP）

本目录集中存放 **Stage 3：CLIP 风格语音–脑电跨模态对比学习** 的代码、数据入口、缓存、结果与日志。与 `EEG/code` 中 TRF/encoding（Stage 1/2）分离，便于独立维护工作流。

## 核心问题

被试观看视频/听语音时，EEG 是否与语音表征存在可学习的跨模态对应？通过冻结语音编码器（Whisper / Wav2Vec2 / HuBERT）+ EEGNet + 对称 InfoNCE，在共享 128 维空间做 retrieval 评估。

## 目录结构

```
跨模态对比学习/
├── README.md
├── environment.yml           # Conda 环境参考
├── code/                     # 训练脚本（已纳入 Git）
├── experiments/              # narrative_identity 等（已纳入 Git）
├── data/
│   ├── stage1_audio_whisper_embeddings_v2/  # 1.1GB Whisper 特征（已纳入 Git）
│   ├── corrected_subjects.txt
│   ├── manifests/、qc/
│   ├── eeg_original/         # 35GB，不纳入 Git；clone 后 fetch
│   └── eeg_corrected/        # 35GB，不纳入 Git；clone 后 fetch
├── cache/                    # Wav2Vec2/HuBERT 语音缓存 26MB（已纳入 Git）
├── results_published/        # Whisper 全量结果摘要（已纳入 Git）
├── scripts/
│   ├── setup_after_clone.sh
│   ├── fetch_eeg_data.sh
│   └── pack_eeg_for_release.sh
└── docs/
```

**任意机器 clone 后**：自带代码 + Whisper 特征 + 语音 cache，装好 conda 并 `fetch` 脑电 pkl 即可跑。详见 [docs/clone后运行.md](docs/clone后运行.md)。

## 三语音基座

| 后端 | 模型 ID | 默认 speech-source |
|------|---------|-------------------|
| whisper | openai/whisper-base | existing-whisper（读已有 embedding） |
| wav2vec2 | facebook/wav2vec2-xls-r-300m | cache（帧级特征，训练前提取） |
| hubert | utter-project/mHuBERT-147-base-3rd-iter | cache |

## 数据版本

- **原版**：`data/eeg_original`，120 被试，与历史 Stage 1/2/3 一致。
- **修正版**：`data/eeg_corrected`，108 原版 + 12 人 begin-timestamp 修正（见 `data/corrected_subjects.txt`）；018/019 未修正。

## 环境与运行

```bash
cd /data/liujialing/TY/建模/EEG与语音联动/跨模态对比学习
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH="$(pwd)/code:$PYTHONPATH"

# 单后端试跑
conda run -n eeg_analysis python code/stage3_speech_eeg_clip.py \
  --speech-backend whisper --speech-source existing-whisper \
  --subjects 002,003 --max-folds 1 --dry-run

# 原版数据：当前机器上正在跑的任务见 logs/stage3_*_full.log
# bash code/run_stage3_safe.sh

# 修正数据：等内存释放后
# bash code/run_stage3_corrected_3gpu.sh
```

## 当前结果（原版数据）

| 运行名 | 状态 |
|--------|------|
| `clip_whisper_existing_full` | 已完成（52/52 fold） |
| `clip_wav2vec2_full` | 训练中 |
| `clip_hubert_full` | 训练中 |

结果 JSON/CSV 在 `results/<run_name>/`。

## 与 EEG 目录的关系

- 本仓库为 **可移植发布版**；服务器上 `EEG/` 仍可能有进行中的训练，脑电 pkl 在本机通过 `data/eeg_*` 软链接指向 `Preprocessing/output/`。
- **Stage 1/2（TRF）**：仍在 `EEG/code`，不在此仓库。

## 相关文档

- [docs/设计说明.md](docs/设计说明.md)
- [docs/数据说明.md](docs/数据说明.md)
- [code/utils/eeg_speech_clip/README.md](code/utils/eeg_speech_clip/README.md)
