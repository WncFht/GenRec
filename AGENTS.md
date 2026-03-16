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

## Training Context
- RL train data mixes multiple tasks. Analyze task-level behavior separately when conclusions may differ across `task1_sid_sft`, `task4_hisTitle2sid`, and `task5_title_desc2sid`.
- For hint-depth or fixed-hint analysis, treat cached `analyze_rl_beam_hint.py` outputs as the source of truth unless the user explicitly asks to regenerate them.
