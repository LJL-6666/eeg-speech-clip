# GitHub 私人仓库上传说明

## 会上传什么

| 会上传 | 不上传（留在服务器） |
|--------|----------------------|
| `code/` 全部脚本与模块 | `data/eeg_*` 脑电 pkl（各约 35GB） |
| `docs/`、`README.md` | Whisper embedding、speech cache |
| `data/corrected_subjects.txt`、`data/README.md` | `results/`、`cache/`、`logs/` |
| `.gitignore` | 预训练模型权重目录 |

GitHub 单文件限制 100MB，仓库不宜存数十 GB 脑电；数据路径见 `docs/数据说明.md`，在服务器用软链接即可。

## 方式一：GitHub Desktop（推荐，与截图一致）

1. 打开 GitHub Desktop → **File → Add Local Repository**
2. 选择路径：`/data/liujialing/TY/建模/EEG与语音联动/跨模态对比学习`
3. 若提示不是 git 仓库，选 **create a repository** 或本目录已 `git init` 后直接添加
4. 填写提交说明 → **Commit to main**
5. **Publish repository**（或 Repository → Publish）
6. Owner 选 **LJL-6666**，勾选 **Keep this code private**，仓库名建议：`eeg-speech-clip` 或 `跨模态对比学习`（英文更省事）
7. 点击 Publish

## 方式二：服务器命令行 + Personal Access Token（推荐）

1. 打开 https://github.com/settings/tokens → **Generate new token (classic)**  
   - 勾选 **repo**（含私人仓库读写）  
   - 生成后复制 token（只显示一次）

2. 在服务器终端执行：

```bash
cd "/data/liujialing/TY/建模/EEG与语音联动/跨模态对比学习"
export GITHUB_TOKEN='ghp_你的token'   # 不要写入文件、不要提交 git
bash scripts/push_to_github.sh
```

脚本会自动：创建私人仓库 `LJL-6666/eeg-speech-clip`（若不存在）→ push `main`（失败会自动重试 5 次）→ 将 remote 改回不含 token 的 URL。

### 若出现 `GnuTLS recv error (-110)` 或 TLS 中断

说明服务器到 `github.com` 网络不稳定，可：

1. 多执行几次 `bash scripts/push_to_github.sh`
2. 或在本机（能稳定访问 GitHub 的电脑）克隆后 push
3. 或打包代码拷到本机 push：

```bash
cd "/data/liujialing/TY/建模/EEG与语音联动"
tar czf /tmp/eeg-speech-clip-code.tar.gz \
  --exclude='.git' 跨模态对比学习/code 跨模态对比学习/docs \
  跨模态对比学习/README.md 跨模态对比学习/.gitignore \
  跨模态对比学习/data/README.md 跨模态对比学习/data/corrected_subjects.txt
# 下载 /tmp/eeg-speech-clip-code.tar.gz 到本机后，在已 clone 的仓库里解压并 commit/push
```

**安全**：Token 不要贴在聊天或终端历史里；若已泄露，请到 GitHub → Settings → Developer settings → 撤销该 token 并重新生成。

## 克隆到新机器后恢复数据

在服务器上重新建软链接，或从 `Preprocessing/output/` 复制 pkl，参见 `docs/数据说明.md`。
