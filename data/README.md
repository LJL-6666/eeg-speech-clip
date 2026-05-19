# data/ 数据入口

本目录**不复制大文件**，均为软链接，指向仓库内已有数据。

| 链接名 | 内容 |
|--------|------|
| `eeg_original` | 120 被试 communication 预处理 pkl（原版） |
| `eeg_corrected` | 12 人 timestamp 修正 + 108 人原版 |
| `stage1_audio_whisper_embeddings_v2` | Whisper full-audio embedding（Stage 3 whisper 后端） |
| `whisper_embeddings` | 同上（别名） |
| `raw_bdf_begin_timestamp_correction` | 修正试验 QC 与 isolated 输出 |

修正被试列表见 `corrected_subjects.txt`。
