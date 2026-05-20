#!/usr/bin/env bash
# 将脑电 pkl 打成 ≤1.9GB 的分卷，便于上传 GitHub Release（单 asset 上限 2GB）
set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VARIANT="corrected"
PART_MB=1800

while [ $# -gt 0 ]; do
  case "$1" in
    --variant) VARIANT="$2"; shift 2 ;;
    --part-mb) PART_MB="$2"; shift 2 ;;
    *) echo "未知: $1"; exit 1 ;;
  esac
done

if [ "$VARIANT" = "original" ]; then
  SRC="/data/liujialing/TY/建模/EEG与语音联动/Preprocessing/output/communication"
else
  SRC="/data/liujialing/TY/建模/EEG与语音联动/Preprocessing/output/communication_corrected_20260519"
fi

OUT="$PROJ_ROOT/data/release_staging/${VARIANT}"
mkdir -p "$OUT"
rm -f "$OUT"/eeg_${VARIANT}_part_*.tar.gz

echo "打包 $SRC -> $OUT (每卷约 ${PART_MB}MB)"
tar -C "$(dirname "$SRC")" -cf - "$(basename "$SRC")" \
  | split -b "${PART_MB}M" -d -a 3 - "$OUT/eeg_${VARIANT}_part_" --additional-suffix=.tar.gz

ls -lh "$OUT"/eeg_${VARIANT}_part_*.tar.gz
echo ""
echo "请将以上文件上传到 GitHub Release（tag 建议: eeg-data-v1）"
echo "他人下载后放入: $OUT 并运行 bash scripts/fetch_eeg_data.sh --from-release $VARIANT"
