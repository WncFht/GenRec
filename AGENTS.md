# AGENTS.md

## Environment
- This machine is primarily for local editing, lightweight CPU analysis, and notebook authoring.
- GPU training and large-scale generation run on the remote server, not on this local machine.
- Final experiments, long-running jobs, and analysis that depends on full artifacts should assume remote execution paths first.

## Analysis Workflow
- Prefer reading existing remote outputs such as `summary.json`, `details.json`, cached logs, and exported maps instead of re-running generation locally.
- For result analysis notebooks and scripts, default to remote paths under `/mnt/dolphinfs/.../GenRec`.
- Keep analysis artifacts concentrated in as few files as possible so they are easy to sync.
- For plots in notebooks and analysis scripts, keep titles, axis labels, and legend text in English to avoid CJK font rendering warnings; use markdown for Chinese explanations when needed.

## Experiment Documentation
- Treat `docs/` as the home for human-readable experiment tracking and project notes.
- Top-level dated notes in `docs/` must use `YYYY-MM-DD-<slug>.md`; do not use aliases such as `article.md`, `周报.md`, or `*.zh.md`.
- Keep `docs/README.md` updated whenever adding or renaming a top-level note.
- Keep evergreen workflow docs in stable filenames such as `docs/experiment_tracking.md`, `docs/eval_wandb_sidecar.md`, `docs/runtime_paths.md`, and `docs/sft_rl_data_pipeline.md`.
- When recording an experiment, explicitly cite config entrypoints under `hope/` and output/result directories under `results/`.
- Each experiment note should include the objective, record date, last updated date, dataset/split/task, base model or checkpoint, relevant `hope/` scripts, important overrides, result paths, metric sources, best-checkpoint summary, and next actions.
- Prefer extending the existing note for the same experiment line instead of creating near-duplicate summaries.

## Editing Guidance
- Avoid introducing extra helper modules unless they clearly reduce complexity.
- When updating existing analysis notebooks, prefer extending the current notebook over creating parallel variants.
- Do not assume remote files are mirrored locally; make remote locations explicit in configuration cells.
- For repo shell scripts, especially ones that run on or get synced to the remote training machine, avoid `set -u` / `set -euo pipefail`; that environment can trip nounset on partially populated envs. Prefer `set -eo pipefail` plus explicit defaults instead.

## Version Control
- This repository is `jj-first`. Use `jj` for routine status, history, diff, commit-description, and push workflows.
- Prefer `jj st`, `jj log`, `jj diff`, `jj new`, and `jj desc -m "..."` over the corresponding Git commands.
- Treat Git as a compatibility layer for colocated repo plumbing or explicit user requests, not as the primary day-to-day interface.
- Remember that colocated `jj` repos may look like detached HEADs from Git's point of view; trust `jj st` and `jj log` for the authoritative local state.
- Before pushing, move or create the appropriate bookmark explicitly, then use `jj git push -b <bookmark>`.

## Training Context
- RL train data mixes multiple tasks. Analyze task-level behavior separately when conclusions may differ across `task1_sid_sft`, `task4_hisTitle2sid`, and `task5_title_desc2sid`.
- For hint-depth or fixed-hint analysis, treat cached `analyze_rl_beam_hint.py` outputs as the source of truth unless the user explicitly asks to regenerate them.
