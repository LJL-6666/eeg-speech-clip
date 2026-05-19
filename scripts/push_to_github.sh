#!/usr/bin/env bash
# 在服务器上一键创建私人仓库并 push（需 GitHub Personal Access Token）
#
# 用法：
#   export GITHUB_TOKEN='ghp_xxxxxxxx'   # 勿写入脚本、勿提交 git
#   bash scripts/push_to_github.sh
#
# Token 权限：至少勾选 repo（私人仓库）

set -euo pipefail

GITHUB_USER="${GITHUB_USER:-LJL-6666}"
REPO_NAME="${REPO_NAME:-eeg-speech-clip}"
PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -z "${GITHUB_TOKEN:-}" ]; then
  echo "[ERROR] 请先设置环境变量 GITHUB_TOKEN"
  echo "  在 https://github.com/settings/tokens 创建 Classic token，勾选 repo"
  echo "  然后执行: export GITHUB_TOKEN='你的token'"
  exit 1
fi

cd "$PROJ_ROOT"

echo "===== 1. 在 GitHub 创建私人仓库（若已存在则跳过）====="
HTTP_CODE=$(curl -sS -o /tmp/gh_create_repo.json -w "%{http_code}" \
  -X POST -H "Authorization: token ${GITHUB_TOKEN}" \
  -H "Accept: application/vnd.github+json" \
  "https://api.github.com/user/repos" \
  -d "{\"name\":\"${REPO_NAME}\",\"private\":true,\"description\":\"EEG-speech CLIP contrastive learning (Stage 3)\"}")

if [ "$HTTP_CODE" = "201" ]; then
  echo "已创建私人仓库: https://github.com/${GITHUB_USER}/${REPO_NAME}"
elif [ "$HTTP_CODE" = "422" ]; then
  echo "仓库可能已存在，继续 push..."
  cat /tmp/gh_create_repo.json | head -c 200; echo
else
  echo "[WARN] 创建仓库 HTTP $HTTP_CODE"
  cat /tmp/gh_create_repo.json
  echo ""
fi

echo ""
echo "===== 2. 配置 remote 并 push ====="
REPO_URL="https://${GITHUB_USER}:${GITHUB_TOKEN}@github.com/${GITHUB_USER}/${REPO_NAME}.git"
if git remote get-url origin &>/dev/null; then
  git remote set-url origin "https://github.com/${GITHUB_USER}/${REPO_NAME}.git"
else
  git remote add origin "https://github.com/${GITHUB_USER}/${REPO_NAME}.git"
fi

# 缓解服务器到 GitHub HTTPS 不稳定（GnuTLS -110 等）
git config --global http.version HTTP/1.1 2>/dev/null || true
git config --global http.postBuffer 524288000 2>/dev/null || true

PUSH_OK=0
for attempt in 1 2 3 4 5; do
  echo "push 尝试 $attempt/5 ..."
  if git push "$REPO_URL" main:main; then
    PUSH_OK=1
    break
  fi
  echo "push 失败，15 秒后重试..."
  sleep 15
done
[ "$PUSH_OK" -eq 1 ] || { echo "[ERROR] push 多次失败，见下方「网络备选方案」"; exit 1; }

git branch --set-upstream-to=origin/main main 2>/dev/null || true

# 恢复 remote 为不含 token 的 URL
git remote set-url origin "https://github.com/${GITHUB_USER}/${REPO_NAME}.git"

echo ""
echo "===== 完成 ====="
echo "私人仓库: https://github.com/${GITHUB_USER}/${REPO_NAME}"
