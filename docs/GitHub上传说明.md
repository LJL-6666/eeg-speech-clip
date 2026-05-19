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

## 方式二：命令行（需已安装 `gh` 并 `gh auth login`）

```bash
cd "/data/liujialing/TY/建模/EEG与语音联动/跨模态对比学习"
gh repo create eeg-speech-clip --private --source=. --remote=origin --push
```

## 克隆到新机器后恢复数据

在服务器上重新建软链接，或从 `Preprocessing/output/` 复制 pkl，参见 `docs/数据说明.md`。
