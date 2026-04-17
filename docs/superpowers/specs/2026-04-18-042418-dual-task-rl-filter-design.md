# Dual-Task RL Filter Design

## Context

The current GenRec RL pipeline supports task selection at data-preprocessing time, but the training and fixed-hint analysis entrypoints still consume the full `rl/{train,valid,test}.json` payload from a data variant.

This is why existing `sid-only` runs rely on a dedicated preprocessed RL dataset variant instead of selecting tasks at training time.

For the new run family, the desired behavior is:

- Train on `task1_sid_sft` and `task5_title_desc2sid`
- Keep eval on `task1_sid_sft` only
- Support both dynamic-hint and fixed-hint launchers
- Avoid regenerating a dedicated RL dataset variant for this task combination

## Goals

1. Add task-level filtering to the RL training entrypoint so mixed RL datasets can be reused directly.
2. Add the same task-level filtering to the fixed-hint beam-analysis/export path so exported hint maps match the training subset.
3. Preserve the current eval behavior where `task1_sid_sft` remains the only eval task.
4. Add dedicated launchers for the dual-task run family without disturbing existing launchers.

## Non-Goals

- Do not change preprocessing defaults or require a new RL data variant.
- Do not add new eval data for `task5_title_desc2sid`.
- Do not change reward definitions or hint-depth semantics.
- Do not refactor unrelated launcher structure.

## Existing Constraints

### RL data layout

`preprocess_data_sft_rl.py` already writes task labels into `extra_info.task` and mixes:

- `task1_sid_sft`
- `task4_hisTitle2sid`
- `task5_title_desc2sid`

inside RL train data. RL valid/test only contain task1 samples today.

### Trainer limitation

`trl_trainer.py` loads the full dataset split but has no parameter for including or excluding tasks. Any task selection therefore has to happen before training, by pointing the launcher at a dedicated dataset directory.

### Fixed-hint limitation

`analyze_rl_beam_hint.py` also reads the full RL train set and exports a fixed-hint depth map for every sample it sees. Because fixed-hint training consumes that exported map, training-time filtering alone is insufficient: the exported map should be built on the same task subset.

## Design

### 1. Add task filtering to `trl_trainer.py`

Introduce two optional string arguments:

- `train_task_names`
- `eval_task_names`

Format:

- comma-separated task names
- empty / unset means "use all tasks in that split"

Behavior:

- After loading `train_dataset` and `eval_dataset`, filter rows by `example["extra_info"]["task"]`
- Preserve the existing default when no filter is provided
- Print the resolved task filter and post-filter dataset sizes to logs

This allows mixed RL data to remain the default source while enabling launcher-level task selection.

### 2. Add task filtering to `analyze_rl_beam_hint.py`

Introduce one optional string argument:

- `task_names`

Format matches `trl_trainer.py`.

Behavior:

- Filter loaded train samples before any cascade analysis runs
- Reuse the filtered sample list for both summary/details generation and `export_fixed_hint_depth_map`
- Log the resolved task filter and filtered sample count

This ensures the fixed-hint depth map is generated from exactly the same task subset used by training.

### 3. Launcher defaults for the new run family

Add two new launchers derived from the existing `sid-only` scripts:

- dynamic hint dual-task launcher
- fixed hint dual-task launcher

Default task settings:

- train tasks: `task1_sid_sft,task5_title_desc2sid`
- eval tasks: `task1_sid_sft`

The fixed-hint launcher should also pass the same train-task list to `analyze_rl_beam_hint.py`.

Naming should clearly indicate the run is:

- rule-only
- dual-task (`sid + title_desc2sid`)
- dynamic or fixed

### 4. Eval behavior

No new eval data generation is required.

Because RL valid/test already only contain task1 data, explicitly passing `eval_task_names=task1_sid_sft` is both correct and future-proof. It makes the intended eval contract visible even if future data variants add more RL eval tasks.

## Error Handling

- Unknown task names should raise a clear `ValueError` listing available tasks in the split.
- Filtering down to zero rows should raise a clear `ValueError` instead of silently starting a broken run.
- Missing `extra_info.task` should raise a clear error in both trainer and hint-analysis code paths.

## Testing

Add focused tests for:

- trainer split filtering by task
- trainer rejection of unknown tasks
- trainer rejection of empty filtered splits
- hint-analysis filtering before fixed-hint export
- launcher smoke tests ensuring the new scripts forward the expected task-filter flags

## Implementation Notes

- Keep parsing logic small and local; a tiny shared helper is acceptable if it removes duplication between `trl_trainer.py` and `analyze_rl_beam_hint.py`.
- Existing launchers and behavior must remain unchanged when the new filter args are omitted.
- The new launchers should keep the current shell conventions used in `hope/`.
