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

## Editing Guidance
- Avoid introducing extra helper modules unless they clearly reduce complexity.
- When updating existing analysis notebooks, prefer extending the current notebook over creating parallel variants.
- Do not assume remote files are mirrored locally; make remote locations explicit in configuration cells.

## Version Control
- This repository is `jj-first`. Use `jj` for routine status, history, diff, commit-description, and push workflows.
- Prefer `jj st`, `jj log`, `jj diff`, `jj new`, and `jj desc -m "..."` over the corresponding Git commands.
- Treat Git as a compatibility layer for colocated repo plumbing or explicit user requests, not as the primary day-to-day interface.
- Remember that colocated `jj` repos may look like detached HEADs from Git's point of view; trust `jj st` and `jj log` for the authoritative local state.
- Before pushing, move or create the appropriate bookmark explicitly, then use `jj git push -b <bookmark>`.

## Training Context
- RL train data mixes multiple tasks. Analyze task-level behavior separately when conclusions may differ across `task1_sid_sft`, `task4_hisTitle2sid`, and `task5_title_desc2sid`.
- For hint-depth or fixed-hint analysis, treat cached `analyze_rl_beam_hint.py` outputs as the source of truth unless the user explicitly asks to regenerate them.
