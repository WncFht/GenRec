# Docs Map

This directory mixes long-lived operator guides with dated research notes.

## Stable Guides

- `eval_wandb_sidecar.md`: remote-to-local evaluation upload workflow and manifest conventions.
- `runtime_paths.md`: local runtime directory conventions derived from current shell entrypoints.
- `sft_rl_data_pipeline.md`: SFT/RL data preparation flow around `preprocess_data_sft_rl.py`.

## Research Notes

The dated `genrec_*.md` files and `article.md` are analysis notes. Keep them as references, but treat them as snapshots instead of source-of-truth operational docs.

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
