# Runtime Paths

This note summarizes where the current shell entrypoints write local runtime state by default.

## Training

- `trl_trainer.sh`
  - foreground and background logs: `log/<run-name>.log`
  - default training outputs: `rl_outputs/<category>-qwen2.5-3b-instruct-grpo`
  - `LOG_DIR` and `LOG_FILE` can override the log location.

## Evaluation

- `evaluate.sh`
  - temporary split/intermediate files: `temp/eval-<category>`
  - metrics/results: `results/<model-subpath>`
  - `TEMP_ROOT` and `RESULTS_ROOT` can override the roots.

- `evaluate_sft_3b.sh`
  - temporary split/intermediate files: `temp/eval-<category>-sft3b-<model-parent>-<checkpoint>`
  - metrics/results: `results/<model-parent>/<checkpoint>`
  - `TEMP_ROOT` and `RESULTS_ROOT` can override the roots.

- `scripts/evaluate_all_checkpoints.sh`
  - background logs: `log/evaluate_all_checkpoints_<timestamp>.log`
  - pid marker: `<log-file>.pid`
  - `LOG_DIR` and `LOG_FILE` can override the log location.

## W&B Upload Sidecar

- `eval_wandb_sidecar.sh`
  - manager pid/log files: `log/eval_sidecar/<instance>.pid` and `.log`
  - `MANAGER_DIR` can override the directory.

- `eval_wandb_sidecar.py`
  - uploader progress state: `state/wandb_eval_uploader`
  - manifest generation reads and writes under `results/`.

## Hint Analysis

- `scripts/sync_hint_research_bundle.sh`
  - source bundle inputs include:
    - `output/jupyter-notebook/...`
    - `temp/rl_beam_hint/...`
  - local unpack root defaults to `output/local-research-bundles/`.

## Cleanup Boundary

The repo treats these as local working areas rather than source-of-truth code:

- `log/`
- `temp/`
- `state/`
- `results/`
- `output/`
- `wandb/`

That does not mean they are disposable in every session. It means scripts are allowed to write there, and git should not treat them as source files.
