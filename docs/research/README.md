# GenRec Research 报告维护说明

这个目录现在承载两类内容：

- 一份以 `Instruments-grec` 为主体、并附带 `Games-grec` 附录的 LaTeX 研究稿
- 与这份研究稿强绑定的本地图表生成脚本和本地资产目录

目标是把研究稿的 `.tex`、section、脚本和最终 PNG/PDF 都收敛到
`docs/research/` 内部维护，而不是继续长期依赖 scattered dated note
下面的静态图片。

## 1. 当前范围

- 主文件：`instruments_rl_research_report.tex`
- section 目录：`sections/`
- Instruments 本地图表目录：`assets/instruments-report/`
- Games 本地图表目录：`assets/games-report/`
- 画图脚本目录：`scripts/`
- 样式挑选文档：`instruments_style_options.md`

当前总文档的边界是：

- 正文仍然以 `Instruments-grec` 为主
- `Games-grec` 作为附录收录
- 报告里实际引用到的图，原则上都应该能在 `docs/research/assets/` 下本地重建

## 2. 环境

### Python

统一使用仓库虚拟环境：

```bash
source /Users/fanghaotian/Desktop/src/GenRec/.venv/bin/activate
```

当前报告脚本依赖：

- `matplotlib`
- `pandas`

### LaTeX

当前这台机器上稳定可用的是 `xelatex`：

```bash
/Library/TeX/texbin/xelatex -interaction=nonstopmode -halt-on-error instruments_rl_research_report.tex
```

这里不要求 `latexmk`。当前目录只需要连续跑两次 `xelatex`，就足够更新目录与交叉引用。

### 文本与图表语言

- matplotlib 图里的标题、坐标轴、图例文本保持英文
- `.tex` 正文说明保持中文

这样做的原因是：

- 避免 matplotlib 的 CJK 字体渲染告警
- 保持图表风格统一
- LaTeX 正文仍然可以用中文完整表达实验结论

## 3. 目录结构

```text
docs/research/
├── README.md
├── instruments_rl_research_report.tex
├── instruments_rl_research_report.pdf
├── instruments_style_options.md
├── assets/
│   ├── instruments-report/
│   └── games-report/
├── scripts/
│   ├── build_instruments_report_figures.py
│   ├── build_games_report_figures.py
│   ├── build_instruments_style_options.py
│   └── instruments_plot_lib.py
└── sections/
    ├── 00-overview.tex
    ├── 01-rl-variant-comparison.tex
    ├── ...
    ├── 10-games-pipeline-and-sft.tex
    └── 11-games-rl-results.tex
```

## 4. 当前编译命令

### 4.1 重建全部图表

```bash
source /Users/fanghaotian/Desktop/src/GenRec/.venv/bin/activate
python /Users/fanghaotian/Desktop/src/GenRec/docs/research/scripts/build_instruments_report_figures.py
python /Users/fanghaotian/Desktop/src/GenRec/docs/research/scripts/build_games_report_figures.py
python /Users/fanghaotian/Desktop/src/GenRec/docs/research/scripts/build_instruments_style_options.py
```

### 4.2 重建 PDF

```bash
cd /Users/fanghaotian/Desktop/src/GenRec/docs/research
/Library/TeX/texbin/xelatex -interaction=nonstopmode -halt-on-error instruments_rl_research_report.tex
/Library/TeX/texbin/xelatex -interaction=nonstopmode -halt-on-error instruments_rl_research_report.tex
```

## 5. 当前样式规范

- Active palette：`tableau-10`
- Active CE profile：`CE-A`
- Instruments 样式主入口：
  `scripts/build_instruments_report_figures.py`
- Games 样式主入口：
  `scripts/build_games_report_figures.py`
- 通用 helper：
  `scripts/instruments_plot_lib.py`

当前默认约定：

- SFT 参考线使用中性灰色虚线
- 每条核心变体都有固定的颜色 + marker
- CE 子变体必须同时靠颜色和线型区分，不能只靠相近色相
- 曲线图默认优先使用 epoch 作为横轴
- 当前 report line plot 的默认点大小是 `5`

如果要调整配色或 marker：

1. 先改相应的 `build_*_report_figures.py`
2. 再改 `build_instruments_style_options.py`
3. 重新生成 style 文档和预览图

## 6. 当前图表清单

### 6.1 Instruments 图

由 `build_instruments_report_figures.py` 统一生成：

- `rl-seven-way-main-curves.png`
- `rl-best-ndcg10-vs-hr50-scatter.png`
- `dynamic_sid_only_vs_dynamic_gather_fix_curves.png`
- `fixed_sid_only_vs_fixed_taskfix_curves.png`
- `ranking_dynamic_vs_canonical_dynamic_curves.png`
- `old_fixed_vs_corrected_fixed_curves.png`
- `fixed-hint-bug-task-depth-distribution.png`
- `max1-ablation-epoch-curves.png`
- `max1-vs-fixed-epoch-curves.png`
- `ce_scaling_three_variants_curves.png`
- `ce_scaling_dynamic_first_look_curves.png`
- `single_hint_vs_fixed_family_epoch_curves.png`
- `single_hint_mixed_vs_baselines_compact_curves.png`
- `dual_task_vs_fixed_family_curves.png`
- `dual_task_vs_dynamic_family_curves.png`

配套结构化输出：

- `all_variant_checkpoint_metrics.csv`
- `all_variant_best_summary.csv`
- `instrument_variants_metadata.json`

### 6.2 Games 图

由 `build_games_report_figures.py` 统一生成：

- `games_sft_checkpoint_curves.png`
- `games_rl_epoch_curves.png`
- `games_best_ndcg10_vs_hr50_scatter.png`

配套结构化输出：

- `games_sft_checkpoint_metrics.csv`
- `games_sft_best_summary.csv`
- `games_rl_checkpoint_metrics.csv`
- `games_rl_best_summary.csv`
- `games_rl_variants_metadata.json`

## 7. 数据来源

### 7.1 Instruments

大多数 Instruments 图都直接读取本地同步回来的 checkpoint 指标：

- `results/<model_dir>/checkpoint-*/metrics.json`
- `results/Instruments-grec-sft-qwen4B-4-256-dsz0/checkpoint-495/metrics.json`

当前唯一特殊项是 old fixed bug 的 depth distribution 图，它读取：

- `docs/deepresearch/genrec_rl_study_2026-03-28/data/fixed_hint_bug_depth_distribution.csv`
- `docs/deepresearch/genrec_rl_study_2026-03-28/data/fixed_hint_bug_task_summary.csv`

### 7.2 Games

Games 图也统一读取本地 `results/`：

- `results/Games-grec-sft-qwen4B-4-256-dsz0/checkpoint-*/metrics.json`
- `results/Games-grec-grpo-rule-only-rerun-quietlog-qwen2.5-3b-qwen4B-4-256-from-sft896/checkpoint-*/metrics.json`
- `results/Games-grec-grpo-rule-only-dynamic-hint-cascade-qwen2.5-3b-qwen4B-4-256-from-sft896/checkpoint-*/metrics.json`
- `results/Games-grec-grpo-rule-only-fixedhint-taskfix-b16-sft896/checkpoint-*/metrics.json`

Games 这一侧目前不再依赖 W\&B 截图作为正式报告图源。

## 8. 以后新增 section 怎么写

### 8.1 建 section 文件

沿用当前编号规则，例如：

```text
sections/12-your-topic.tex
```

建议：

- 一个 section 文件只讲一个清晰的问题
- 不要把多个几乎独立的实验线强塞到同一节
- 能放表和图说明清楚的，就不要再把 dated note 大段原文搬进来

### 8.2 接进主文

在 `instruments_rl_research_report.tex` 里加：

```tex
\input{sections/12-your-topic}
```

如果它更适合作为附录，就放在 `\appendix` 之后。

### 8.3 图表优先放本地资产

在 `.tex` 里只写文件名，例如：

```tex
\includegraphics{games_rl_epoch_curves.png}
```

不要再把总文档绑回老的 `docs/assets/YYYY-MM-DD-.../` 目录，除非你还没有把对应脚本迁进 `docs/research/scripts/`。

### 8.4 优先复用现有脚本

如果新增 section 需要图：

- Instruments 相关，优先扩 `build_instruments_report_figures.py`
- Games 相关，优先扩 `build_games_report_figures.py`
- 通用散点图 / metric grid / best-point scatter，优先扩 `instruments_plot_lib.py`

只有在图类型完全不同、而且之后很可能复用时，才值得继续加新的 helper。

## 9. 什么时候该扩画图脚本

### 9.1 扩 `build_instruments_report_figures.py`

适用于：

- Instruments 主报告新增 figure
- 老的 Instruments dated note 图需要迁进 research 本地资产目录
- 新增 Instruments 变体，需要稳定样式分配

### 9.2 扩 `build_games_report_figures.py`

适用于：

- Games 附录新增 figure
- Games SFT / RL 又同步到了新的 checkpoint
- 需要把 Games 的 task-level 结果也纳入同一份 research 报告

### 9.3 扩 `instruments_plot_lib.py`

适用于：

- 新 helper 足够通用
- 同一类 matplotlib 代码已经在多个脚本里重复

## 10. 常见维护规则

- 不要在这套 research stack 里硬编码 W\&B 链接
- 不要把外部平台 URL 再塞回 metadata，除非报告明确需要
- Instruments 变体优先维护现有 `SPECS`
- 图一旦更新，同一轮任务里要重新编译 PDF，避免 `.tex` 和渲染产物不同步
- 新图能本地重建，就不要继续长期依赖 dated-note 里的静态 PNG

## 11. 当前已知 LaTeX 状态

当前总文档已经可以稳定用 `xelatex` 编译通过，但仍保留若干
`Overfull \hbox` / `Underfull \hbox` 警告，主要来自：

- 很长的 `\path{...}` launcher 路径
- 较窄表格列里放了长 checkpoint 字符串

这些警告目前不阻塞 PDF 产出，因此暂时容忍。后面如果要做排版清理，优先用这些方法处理：

- 缩短正文里展示的路径文本
- 调整表格列宽
- 减少不必要的长 inline literal

不要为了消除 warning 去改实验目录本身的真实名字。
