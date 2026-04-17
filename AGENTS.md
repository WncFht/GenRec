# AGENTS.md

## 环境
- 这台机器主要用于本地编辑、轻量级 CPU 分析，以及 notebook 编写。
- GPU 训练和大规模生成任务在远端服务器上进行，而不是在这台本机上。
- 最终实验、长时间运行的任务，以及依赖完整产物的分析，应优先假设走远端执行路径。

## 分析工作流
- 优先读取已有的远端产物，例如 `summary.json`、`details.json`、缓存日志和导出的 map，而不是在本地重新跑生成。
- 对结果分析 notebook 和脚本，默认使用 `/mnt/dolphinfs/.../GenRec` 下的远端路径。
- 尽量把分析产物集中在尽可能少的文件里，方便同步。
- 对 notebook 和分析脚本里的图表，标题、坐标轴标签和图例文本请保持英文，以避免 CJK 字体渲染告警；需要中文说明时用 markdown。

## 实验文档
- 将 `docs/` 视为可读性实验追踪和项目笔记的主目录。
- `docs/` 下顶层带日期的笔记必须使用 `YYYY-MM-DD-<slug>.md` 格式；不要使用 `article.md`、`周报.md` 或 `*.zh.md` 这类别名。
- 只要新增或重命名了顶层笔记，就同步更新 `docs/README.md`。
- 常青型工作流文档应保留稳定文件名，例如 `docs/experiment_tracking.md`、`docs/eval_wandb_sidecar.md`、`docs/runtime_paths.md` 和 `docs/sft_rl_data_pipeline.md`。
- 记录实验时，要明确引用 `hope/` 下的配置入口，以及 `results/` 下的输出/结果目录。
- 每篇实验记录都应包含：目标、记录日期、最后更新日期、数据集/切分/任务、基础模型或 checkpoint、相关 `hope/` 脚本、关键 overrides、结果路径、指标来源、最佳 checkpoint 摘要，以及后续动作。
- 对同一条实验线，优先继续扩写已有文档，而不是新建内容几乎重复的摘要。

## 编辑建议
- 除非确实能明显降低复杂度，否则避免引入额外的 helper module。
- 更新已有分析 notebook 时，优先在原 notebook 基础上继续扩展，而不是再建平行变体。
- 不要假设远端文件已经镜像到本地；在配置单元里把远端位置写清楚。
- 对仓库里的 shell 脚本，尤其是会在远端训练机上运行或同步过去的脚本，避免使用 `set -u` / `set -euo pipefail`；那个环境里未完全填充的环境变量很容易触发 nounset。优先使用 `set -eo pipefail`，并给显式默认值。

## 版本控制
- 这个仓库是 `jj-first`。日常的状态、历史、diff、commit 描述和 push 工作流都使用 `jj`。
- 相比对应的 Git 命令，优先使用 `jj st`、`jj log`、`jj diff`、`jj new` 和 `jj desc -m "..."`。
- 将 Git 视为共置仓库 plumbing 或用户显式要求时的兼容层，而不是日常主工作流。
- 要记住：共置的 `jj` 仓库从 Git 视角看可能像 detached HEAD；本地状态应以 `jj st` 和 `jj log` 为准。
- 推送前，先显式移动或创建合适的 bookmark，然后再使用 `jj git push -b <bookmark>`。

## 训练上下文
- RL 训练数据混合了多个任务。当 `task1_sid_sft`、`task4_hisTitle2sid` 和 `task5_title_desc2sid` 之间的结论可能不一致时，应分任务分析。
- 对 hint depth 或 fixed hint 相关分析，除非用户明确要求重新生成，否则将缓存好的 `analyze_rl_beam_hint.py` 输出视为事实来源。
