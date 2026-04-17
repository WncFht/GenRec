# Dual-Task RL Filter Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow GenRec RL launchers to train on `task1_sid_sft` + `task5_title_desc2sid` directly from mixed RL data, while keeping eval on `task1_sid_sft` only and keeping fixed-hint analysis/export aligned with the same training task subset.

**Architecture:** Add small task-filter parsing and split-filtering helpers to the training and beam-hint analysis entrypoints, keep filtering opt-in via new CLI args, and add dedicated dual-task dynamic/fixed shell launchers that forward the same task selections consistently. Fixed-hint export must consume the filtered training sample list so `task+index` maps stay aligned with the subset used by training.

**Tech Stack:** Python, `datasets`, `fire`, bash launchers, `unittest`, `pytest`

---

## Chunk 1: Trainer Task Filtering

### Task 1: Add failing tests for trainer split filtering

**Files:**
- Modify: `tests/test_trl_trainer_entrypoint.py`
- Modify: `trl_trainer.py`

- [ ] **Step 1: Write failing tests for train/eval task filtering and invalid-task rejection**

Add tests that:

- call `trl_trainer.main(...)` with `train_task_names="task1_sid_sft,task5_title_desc2sid"` and `eval_task_names="task1_sid_sft"`
- assert the constructed trainer receives a filtered train dataset containing only `task1_sid_sft` and `task5_title_desc2sid`
- assert the eval dataset contains only `task1_sid_sft`
- assert an unknown task name raises `ValueError`
- assert filtering to zero examples raises `ValueError`

- [ ] **Step 2: Run targeted trainer-entrypoint tests and verify they fail**

Run:

```bash
python3 -m pytest tests/test_trl_trainer_entrypoint.py -k "task_filter or dual_task" -q
```

Expected:

- FAIL because `trl_trainer.py` does not yet accept or apply task filters

- [ ] **Step 3: Implement minimal trainer-side task parsing and split filtering**

In `trl_trainer.py`:

- add optional `train_task_names` / `eval_task_names` args
- parse comma-separated names into normalized sets
- filter loaded `train_dataset` / `eval_dataset` by `extra_info.task`
- raise clear `ValueError` on unknown tasks, missing task metadata, or empty filtered result
- log requested filters, available tasks, and post-filter split sizes

- [ ] **Step 4: Run targeted trainer-entrypoint tests and verify they pass**

Run:

```bash
python3 -m pytest tests/test_trl_trainer_entrypoint.py -k "task_filter or dual_task" -q
```

Expected:

- PASS

## Chunk 2: Fixed-Hint Analysis Task Filtering

### Task 2: Add failing tests for beam-hint task filtering

**Files:**
- Modify: `tests/test_analyze_rl_beam_hint.py`
- Modify: `analyze_rl_beam_hint.py`

- [ ] **Step 1: Write failing tests for train-sample task filtering in beam analysis**

Add tests that:

- validate task-name parsing/filtering on loaded train samples
- assert only requested tasks survive
- assert unknown task names raise `ValueError`
- assert empty filtered result raises `ValueError`
- assert missing `extra_info.task` raises `ValueError`

- [ ] **Step 2: Run targeted beam-hint tests and verify they fail**

Run:

```bash
python3 -m pytest tests/test_analyze_rl_beam_hint.py -k "task_filter" -q
```

Expected:

- FAIL because no task-filter helper exists yet

- [ ] **Step 3: Implement minimal beam-analysis task filtering**

In `analyze_rl_beam_hint.py`:

- add optional CLI arg `task_names`
- parse comma-separated task names
- filter the train sample list immediately after loading
- reuse the filtered sample list for all later summary/detail/export paths
- log requested filters and filtered sample counts
- raise clear `ValueError` on invalid task names, missing metadata, or empty result

- [ ] **Step 4: Run targeted beam-hint tests and verify they pass**

Run:

```bash
python3 -m pytest tests/test_analyze_rl_beam_hint.py -k "task_filter" -q
```

Expected:

- PASS

## Chunk 3: Dual-Task Launchers

### Task 3: Add dual-task launcher coverage first

**Files:**
- Modify: `tests/test_trl_trainer_entrypoint.py`
- Create: `hope/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-rule-only-dynamic-hint-sid-title-desc.sh`
- Create: `hope/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-rule-only-fixed-hint-sid-title-desc.sh`

- [ ] **Step 1: Write failing launcher dry-run tests**

Add tests that:

- verify the new dynamic launcher forwards `--train_task_names task1_sid_sft,task5_title_desc2sid`
- verify the new dynamic launcher forwards `--eval_task_names task1_sid_sft`
- verify the new fixed launcher forwards the same trainer flags
- verify the new fixed launcher forwards `--task_names task1_sid_sft,task5_title_desc2sid` to `analyze_rl_beam_hint.py`
- verify both launchers remain standalone scripts

- [ ] **Step 2: Run targeted launcher tests and verify they fail**

Run:

```bash
python3 -m pytest tests/test_trl_trainer_entrypoint.py -k "sid_title_desc" -q
```

Expected:

- FAIL because the new launcher scripts do not exist yet

- [ ] **Step 3: Implement the new launcher scripts**

Create two new standalone scripts derived from the current `sid-only` launchers, but:

- point at the mixed RL data variant by default
- default `TRAIN_TASK_NAMES="task1_sid_sft,task5_title_desc2sid"`
- default `EVAL_TASK_NAMES="task1_sid_sft"`
- pass trainer filter flags to `trl_trainer.py`
- pass beam-analysis `--task_names` in the fixed-hint script
- keep current shell style, dry-run output, nohup handling, and existing fixed/dynamic defaults

- [ ] **Step 4: Run targeted launcher tests and verify they pass**

Run:

```bash
python3 -m pytest tests/test_trl_trainer_entrypoint.py -k "sid_title_desc" -q
```

Expected:

- PASS

## Chunk 4: Integration Verification

### Task 4: Run the full focused verification set

**Files:**
- Modify: `trl_trainer.py`
- Modify: `analyze_rl_beam_hint.py`
- Modify: `tests/test_trl_trainer_entrypoint.py`
- Modify: `tests/test_analyze_rl_beam_hint.py`
- Create: `hope/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-rule-only-dynamic-hint-sid-title-desc.sh`
- Create: `hope/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-rule-only-fixed-hint-sid-title-desc.sh`

- [ ] **Step 1: Run focused test suite**

Run:

```bash
python3 -m pytest \
  tests/test_trl_trainer_entrypoint.py \
  tests/test_analyze_rl_beam_hint.py \
  tests/test_fixed_hint_oracle.py \
  -q
```

Expected:

- PASS

- [ ] **Step 2: Run launcher dry-run sanity checks**

Run:

```bash
bash hope/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-rule-only-dynamic-hint-sid-title-desc.sh --dry-run
bash hope/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-rule-only-fixed-hint-sid-title-desc.sh --dry-run
```

Expected:

- dry-run output shows the new task-filter flags
- fixed-hint dry-run output shows `analyze_rl_beam_hint.py` receives `--task_names task1_sid_sft,task5_title_desc2sid`

- [ ] **Step 3: Review final diff for unintended launcher or default behavior regressions**

Run:

```bash
git diff --stat
git diff -- trl_trainer.py analyze_rl_beam_hint.py tests/test_trl_trainer_entrypoint.py tests/test_analyze_rl_beam_hint.py
```

Expected:

- changes are limited to trainer filtering, beam-analysis filtering, tests, and the two new launchers

- [ ] **Step 4: Commit implementation**

Run:

```bash
git add \
  trl_trainer.py \
  analyze_rl_beam_hint.py \
  tests/test_trl_trainer_entrypoint.py \
  tests/test_analyze_rl_beam_hint.py \
  docs/superpowers/plans/2026-04-18-043200-dual-task-rl-filter.md \
  hope/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-rule-only-dynamic-hint-sid-title-desc.sh \
  hope/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-rule-only-fixed-hint-sid-title-desc.sh
git commit -m "feat: add dual-task rl launcher filters"
```

Expected:

- commit succeeds with only the intended files staged
