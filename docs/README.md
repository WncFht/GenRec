# Docs Map

This directory mixes long-lived operator guides with dated research notes.

## Naming Rules

- Top-level dated notes use `YYYY-MM-DD-<slug>.md`.
- Do not create aliases such as `article.md`, `周报.md`, or `*.zh.md`.
- Keep evergreen operator docs in explicit stable filenames instead of forcing fake dates onto them.

## Stable Guides

- `experiment_tracking.md`: how to document experiments with `hope/` configs and `results/` outputs.
- `eval_wandb_sidecar.md`: remote-to-local evaluation upload workflow and manifest conventions.
- `runtime_paths.md`: local runtime directory conventions derived from current shell entrypoints.
- `sft_rl_data_pipeline.md`: SFT/RL data preparation flow around `preprocess_data_sft_rl.py`.

## Research Notes

Treat dated notes as snapshots, not evergreen source-of-truth docs. Current inventory:

- `2026-03-12-genrec-hint-research.md`: initial hint literature/research note. Historical snapshot.
- `2026-03-13-genrec-hint-research-v1-fixed-16-rollout.md`: follow-up design note for the fixed-16-rollout V1 strategy. Historical snapshot.
- `2026-03-13-genrec-rl-reward-forms.md`: reward-form implementation note tied to the 2026-03-13 code state. Historical snapshot.
- `2026-03-15-genrec-fixed-hint-rule-only-status.md`: fixed-hint `rule_only` status record. Historical snapshot.
- `2026-03-16-genrec-hint-local-bundle-findings.md`: Chinese main note for the local hint bundle re-analysis. Historical snapshot.
- `2026-03-16-genrec-hint-local-bundle-findings-english.md`: English companion note for the same bundle pass. Historical snapshot.
- `2026-03-17-genrec-dynamic-fixed-hint-metrics.md`: dynamic vs fixed hint metrics/implementation note. Historical snapshot.
- `2026-03-17-genrec-rollout-node-analysis.md`: node-level rollout failure analysis. Historical snapshot.
- `2026-03-17-genrec-training-speed-optimization-notes.md`: training-speed optimization ideas without changing step counts. Historical snapshot.
- `2026-03-18-genrec-main-results-weekly.md`: detailed weekly report for the 2026-03-18 Instruments-grec result set. Historical snapshot.
- `2026-03-19-genrec-results-weekly-summary.md`: condensed summary of the 2026-03-18 weekly report. Convenience note, not an independent source of truth.
- `2026-04-02-genrec-results-since-2026-03-19-epoch-report.md`: epoch-aligned inventory and analysis of experiments visible in `results/` since the 2026-03-19 reporting boundary.
- `2026-04-11-genrec-instruments-rl-variant-comparison.md`: unified seven-way comparison note for the main `Instruments-grec` RL variants, with locally generated checkpoint-curve assets and manual image-based comparison.
- `2026-04-01-games-grec-qwen4b-4-256-full-pipeline.md`: current Games pipeline note. As of 2026-04-02, the local `results/` tree still has no `*Games*` result directories, so the "RL not started" status remains current.
- `2026-04-02-instruments-lc-rec-reproduction.md`: operator note for converting `Instruments` data, training `LC-Rec` with `Qwen2.5-3B-Instruct`, and evaluating with GenRec's unified checkpoint script.

## Generated Planning Artifacts

- `superpowers/specs/`: design specs generated during structured planning.
- `superpowers/plans/`: implementation plans and execution notes.

## Local-Only Working Areas

These directories are intentionally kept outside git and should be treated as disposable local state when no longer needed:

- `output/`
- `results/`
- `wandb/`
- `state/`
- `temp/`
- `log/`
- `data/_preprocess_input/`

Large extracted bundles under `output/local-research-bundles/` are local analysis workspaces, not source files.
