# Stage 3 Speech-EEG CLIP Baseline

This module implements a conservative CLIP/CLAP-style speech-EEG contrastive
baseline.

Design choices:

- EEG uses the current preprocessed pkl signal directly; no additional low-frequency bandpass is applied.
- Per-window EEG preprocessing is limited to finite checks, optional resampling, and channel-wise z-score.
- Stimulus identity is `speaker_id + target_emotion`; paired samples use `material_key + window_idx`.
- The batch sampler prevents duplicate `pair_key` values in the same batch.
- Speech backends are frozen. Trainable parameters are EEGNet and projection heads only.
- Embeddings are L2-normalized for cosine logits; no extra L2 penalty is added.
- The loss is symmetric CLIP/CLAP-style InfoNCE with one learnable logit scale.

Quick dry run:

```bash
python code/stage3_speech_eeg_clip.py \
  --dry-run \
  --subjects 002,003 \
  --video-ids 1,2 \
  --max-windows-per-trial 2 \
  --run-name dryrun_stage3 \
  --overwrite
```

Use existing Whisper caches:

```bash
python code/stage3_speech_eeg_clip.py \
  --speech-backend whisper \
  --speech-source existing-whisper \
  --subjects 002,003,004,005 \
  --max-folds 1 \
  --epochs 5
```

Prepare a frozen speech cache for another backend:

```bash
CUDA_VISIBLE_DEVICES=0 python code/stage3_speech_eeg_clip.py \
  --speech-backend wav2vec2 \
  --speech-source cache \
  --prepare-speech-cache

CUDA_VISIBLE_DEVICES=1 python code/stage3_speech_eeg_clip.py \
  --speech-backend hubert \
  --speech-source cache \
  --prepare-speech-cache
```

Then run a cache-backed dry run or training without `--prepare-speech-cache`.
