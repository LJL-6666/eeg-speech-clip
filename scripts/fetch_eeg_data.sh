#!/usr/bin/env bash
# 获取脑电 pkl（35GB 级，不在 Git 仓库内）
#
# 用法示例：
#   # 从本实验室服务器复制（推荐）
#   bash scripts/fetch_eeg_data.sh --from-server liujialing@your-server:/data/liujialing/TY/建模/EEG与语音联动/Preprocessing/output/communication_corrected_20260519
#
#   # 从 GitHub Release 分卷包还原（需先在服务器运行 pack_eeg_for_release.sh 并上传 assets）
#   bash scripts/fetch_eeg_data.sh --from-release corrected
#
set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VARIANT="corrected"   # original | corrected
SRC=""
RELEASE_TAG="eeg-data-v1"

usage() {
  sed -n '2,12p' "$0"
  exit 1
}

while [ $# -gt 0 ]; do
  case "$1" in
    --variant) VARIANT="$2"; shift 2 ;;
    --from-server) SRC="$2"; shift 2 ;;
    --from-release) VARIANT="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "未知参数: $1"; usage ;;
  esac
done

if [ "$VARIANT" = "original" ]; then
  DEST="$PROJ_ROOT/data/eeg_original"
else
  DEST="$PROJ_ROOT/data/eeg_corrected"
fi
mkdir -p "$DEST"

if [ -n "$SRC" ]; then
  echo "rsync 从 $SRC -> $DEST"
  rsync -a --info=progress2 "${SRC%/}/" "$DEST/"
  echo "完成: $(ls "$DEST"/*.pkl 2>/dev/null | wc -l) 个 pkl"
  exit 0
fi

STAGING="$PROJ_ROOT/data/release_staging/${VARIANT}"
mkdir -p "$STAGING"

if ls "$STAGING"/eeg_${VARIANT}_part_*.tar.gz &>/dev/null; then
  echo "解压分卷到 $DEST ..."
  cat "$STAGING"/eeg_${VARIANT}_part_*.tar.gz | tar -xzf - -C "$DEST" --strip-components=0 2>/dev/null || {
    for f in "$STAGING"/eeg_${VARIANT}_part_*.tar.gz; do
      tar -xzf "$f" -C "$DEST"
    done
  }
  echo "完成: $(ls "$DEST"/*.pkl 2>/dev/null | wc -l) 个 pkl"
  exit 0
fi

cat <<EOF
[说明] 脑电 pkl 约 35GB，GitHub 无法直接存放（单文件 ~300MB > 100MB 限制）。

请选择一种方式：

1) 实验室 rsync（最快）：
   bash scripts/fetch_eeg_data.sh --from-server USER@HOST:/path/to/communication_corrected_20260519

2) GitHub Release 分卷（需维护者先打包上传）：
   - 在数据服务器: bash scripts/pack_eeg_for_release.sh --variant corrected
   - 将 data/release_staging/corrected/*.tar.gz 上传到 Release: $RELEASE_TAG
   - 下载到 $STAGING 后重新运行本脚本

3) 手动：把 120 个 *.pkl 放入 $DEST
EOF
exit 1
