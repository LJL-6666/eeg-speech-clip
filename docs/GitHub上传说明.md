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

脚本会自动：创建私人仓库 `LJL-6666/eeg-speech-clip`（若不存在）→ push `main` → 将 remote 改回不含 token 的 URL。

## 克隆到新机器后恢复数据

在服务器上重新建软链接，或从 `Preprocessing/output/` 复制 pkl，参见 `docs/数据说明.md`。
