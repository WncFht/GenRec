# GenRec Research LaTeX

当前先整理了一份 `Instruments-grec` RL 研究合并稿：

- 主文件：`instruments_rl_research_report.tex`
- 子章节目录：`sections/`

建议编译命令：

```bash
cd /Users/fanghaotian/Desktop/src/GenRec/docs/research
xelatex -interaction=nonstopmode instruments_rl_research_report.tex
xelatex -interaction=nonstopmode instruments_rl_research_report.tex
```

当前稿件把这三篇 note 合并进同一个 LaTeX 文档：

- `docs/2026-04-11-genrec-instruments-rl-variant-comparison.md`
- `docs/2026-04-16-instruments-dynamic-hint-max1-ablation.md`
- `docs/2026-04-19-instruments-dual-task-single-hint-tracking.md`

结构上已经拆成“总文档 + 分章节子文件”，后续可以继续把单个实验扩写成独立子稿，再由总文档统一 `\input{}`。
