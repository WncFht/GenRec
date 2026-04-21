# GenRec Research Report Workflow

This directory currently hosts the consolidated `Instruments-grec` RL report and
its local figure pipeline. The goal is that the LaTeX document, section files,
plot scripts, and rendered assets are all maintained here rather than relying
on scattered dated-note artifacts.

## Scope

- Main report: `instruments_rl_research_report.tex`
- Section files: `sections/*.tex`
- Local figure outputs: `assets/instruments-report/`
- Figure builders: `scripts/`
- Style reference: `instruments_style_options.md`

The current report is intentionally local-first:

- All figures used by `instruments_rl_research_report.tex` should be available
  under `assets/instruments-report/`.
- Do not introduce new `\includegraphics{...}` dependencies on old
  `docs/assets/YYYY-MM-DD-.../` directories unless there is no local source yet.
- If an old dated note already has a useful figure, port its generation logic
  into `docs/research/scripts/` instead of keeping the report tied to the old
  PNG forever.

## Environment

### Python

Use the repo virtual environment:

```bash
source /Users/fanghaotian/Desktop/src/GenRec/.venv/bin/activate
```

The report builders currently rely on:

- `matplotlib`
- `pandas`

### LaTeX

The report is compiled with `xelatex` on this machine.

Known working command:

```bash
/Library/TeX/texbin/xelatex -interaction=nonstopmode -halt-on-error instruments_rl_research_report.tex
```

`latexmk` is not required. Running `xelatex` twice is enough for the table of
contents and cross-reference state used here.

### Fonts / language split

- Plot titles, axis labels, and legend text should stay in English.
- Body prose inside `.tex` sections can be Chinese.
- This avoids the common CJK warnings and keeps matplotlib output stable.

## Directory layout

```text
docs/research/
├── README.md
├── instruments_rl_research_report.tex
├── instruments_rl_research_report.pdf
├── instruments_style_options.md
├── assets/
│   └── instruments-report/
├── scripts/
│   ├── build_instruments_report_figures.py
│   ├── build_instruments_style_options.py
│   └── instruments_plot_lib.py
└── sections/
    ├── 00-overview.tex
    ├── 01-rl-variant-comparison.tex
    └── ...
```

## Build commands

### Rebuild all report figures

```bash
source /Users/fanghaotian/Desktop/src/GenRec/.venv/bin/activate
python /Users/fanghaotian/Desktop/src/GenRec/docs/research/scripts/build_instruments_report_figures.py
python /Users/fanghaotian/Desktop/src/GenRec/docs/research/scripts/build_instruments_style_options.py
```

### Rebuild the PDF

```bash
cd /Users/fanghaotian/Desktop/src/GenRec/docs/research
/Library/TeX/texbin/xelatex -interaction=nonstopmode -halt-on-error instruments_rl_research_report.tex
/Library/TeX/texbin/xelatex -interaction=nonstopmode -halt-on-error instruments_rl_research_report.tex
```

## Current plotting policy

- Active palette: `tableau-10`
- Active CE profile: `CE-A`
- Source of truth for variant style assignment:
  `scripts/build_instruments_report_figures.py`
- Shared plotting helpers:
  `scripts/instruments_plot_lib.py`

Current conventions:

- Canonical SFT reference line uses a neutral gray dashed line.
- Each tracked variant has a fixed color + marker assignment.
- CE sub-variants must differ by both color and line style; do not rely on hue
  alone.
- Report curves default to epoch on the x-axis.
- Default point size in report line plots is `5`.

If you need to inspect or revise the palette, update:

- `scripts/build_instruments_report_figures.py`
- `scripts/build_instruments_style_options.py`
- `instruments_style_options.md`

Then regenerate figures and the style preview.

## Current figure inventory

These report figures are rebuilt by `build_instruments_report_figures.py`:

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
- `single_hint_vs_fixed_family_epoch_curves.png`
- `single_hint_mixed_vs_baselines_compact_curves.png`
- `dual_task_vs_references_curves.png`

Supporting structured outputs also live beside them:

- `all_variant_checkpoint_metrics.csv`
- `all_variant_best_summary.csv`
- `instrument_variants_metadata.json`

## Data sources

Most report figures read from local synced checkpoint metrics:

- `results/<model_dir>/checkpoint-*/metrics.json`
- `results/Instruments-grec-sft-qwen4B-4-256-dsz0/checkpoint-495/metrics.json`

The fixed-hint bug distribution figure is the only current exception. It reads
from deepresearch CSV exports:

- `docs/deepresearch/genrec_rl_study_2026-03-28/data/fixed_hint_bug_depth_distribution.csv`
- `docs/deepresearch/genrec_rl_study_2026-03-28/data/fixed_hint_bug_task_summary.csv`

If those CSVs are replaced upstream, regenerate the report figure after
verifying the column names still match.

## How to add a new section

### 1. Create a new section file

Use the existing numbering pattern:

```text
sections/10-your-topic.tex
```

Guidelines:

- One section file should correspond to one experimental question or one clean
  comparison unit.
- Keep the prose in Chinese if that reads better for the report narrative.
- Keep figure captions concise.

### 2. Add it to the main LaTeX file

Update `instruments_rl_research_report.tex` and insert:

```tex
\input{sections/10-your-topic}
```

Put it where it belongs in the reading order. Do not append blindly at the end
if the section is conceptually part of an earlier family.

### 3. Generate figures locally

If the section needs plots:

- Prefer extending `scripts/build_instruments_report_figures.py`.
- Reuse `VariantSpec` entries that already exist.
- Reuse `plot_metric_grid()` or `plot_best_scatter()` before writing a custom
  plotting block.
- Only add a new custom plotting function when the chart type is genuinely
  different, like the fixed-hint bug stacked distribution figure.

### 4. Keep figure references local

Inside the `.tex` file:

- Reference only filenames, for example
  `\includegraphics{dual_task_vs_references_curves.png}`.
- The main report already points `\graphicspath` to
  `assets/instruments-report/`.

### 5. Rebuild and verify

After editing:

1. Rebuild figures.
2. Recompile the PDF twice.
3. Check the rendered PDF rather than trusting the `.tex` source.

## When to extend the plot builder

Extend `build_instruments_report_figures.py` when:

- a new section needs a curve/scatter/bar figure for the main report
- a legacy dated-note plot should be pulled into the local report asset set
- a new tracked variant needs a stable style entry

Extend `instruments_plot_lib.py` when:

- the helper is reusable across multiple report figures
- the new plot type is generic enough to avoid duplicated matplotlib code

Avoid adding extra helper files unless the logic is reused by multiple figure
families.

## Common maintenance rules

- Do not hardcode W&B links in this report stack.
- Do not add external URLs into figure metadata unless the report explicitly
  needs them.
- Prefer modifying the existing `SPECS` list over inventing a parallel variant
  registry.
- If a figure is updated, regenerate the PDF in the same task so the rendered
  output stays in sync.

## Known compile state

The report currently compiles successfully with `xelatex`, but it still emits
some `Overfull \hbox` / `Underfull \hbox` warnings caused mainly by:

- long `\path{...}` launcher paths
- long checkpoint strings in narrow table columns

Those warnings are currently tolerated because they do not block PDF generation.
If the report gets cleaned up later, fix them by shortening displayed path text
or widening the relevant table columns, not by changing the underlying paths.
