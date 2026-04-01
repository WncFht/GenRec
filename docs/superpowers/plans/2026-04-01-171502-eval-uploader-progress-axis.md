# Eval Uploader Progress Axis Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add normalized progress fields to the eval uploader so runs with different total checkpoint steps can still be aligned in W&B and local state.

**Architecture:** Keep `checkpoint_step` as the existing source-of-truth axis, and derive two additional axes from each model's discovered checkpoints: `checkpoint_index` and `epoch_progress`. Compute the derived values inside the uploader from the full checkpoint list already visible under `results/<model_dir>`, persist them into uploader state rows, and log them to W&B together with the existing metrics.

**Tech Stack:** Python 3.11, pytest, W&B uploader state JSON, existing manifest-driven uploader flow

---

## Chunk 1: Tests First

### Task 1: Add a regression test for derived progress fields

**Files:**
- Modify: `tests/test_eval_wandb_sidecar.py`
- Modify: `eval_wandb_sidecar.py`

- [ ] **Step 1: Write the failing test**

Add a test that builds a temporary `results/` tree with one model and uneven checkpoint steps such as `266`, `532`, `798`, plus numeric `metrics.json` payloads. Exercise the uploader once with W&B disabled and assert that the saved uploader state rows contain:

```python
{
    "checkpoint_step": 266,
    "checkpoint_index": 1,
    "epoch_progress": pytest.approx(2 / 3, rel=1e-6),
}
```

for the first row when the model has `num_train_epochs=2`.

- [ ] **Step 2: Run the targeted test to verify it fails**

Run: `pytest tests/test_eval_wandb_sidecar.py -k progress -v`

Expected: FAIL because the uploader currently stores only `checkpoint_step`/`checkpoint_name` without derived progress fields.

## Chunk 2: Minimal Implementation

### Task 2: Derive and persist normalized axes in the uploader

**Files:**
- Modify: `eval_wandb_sidecar.py`

- [ ] **Step 1: Extend uploader model metadata**

Add a model-level `num_train_epochs` field with a default value of `2` when generating and parsing the manifest.

- [ ] **Step 2: Compute derived checkpoint axes**

Introduce a small helper that receives the full discovered checkpoint list, the current checkpoint, and the model spec, then returns:

```python
{
    "checkpoint_index": <1-based rank>,
    "epoch_progress": <checkpoint_step / last_checkpoint_step * num_train_epochs>,
}
```

Use the largest discovered checkpoint step as the normalization denominator so sid-only and non-sid-only runs align by relative training progress while still respecting their actual checkpoint spacing.

- [ ] **Step 3: Attach fields to rows and W&B payloads**

Update row construction, table columns, and `run.log(...)` payloads so `checkpoint_index` and `epoch_progress` are stored in local uploader state and sent to W&B alongside the existing metrics.

- [ ] **Step 4: Keep backward compatibility**

Ensure existing state files without the new keys can still be loaded and merged without crashing.

## Chunk 3: Verification And Re-upload

### Task 3: Verify code, update docs, and re-upload the affected runs

**Files:**
- Modify: `tests/test_eval_wandb_sidecar.py`
- Modify: `eval_wandb_sidecar.py`
- Modify: `docs/eval_wandb_sidecar.md`
- Modify: `state/wandb_eval_uploader/*.json` only by deliberate deletion/rebuild for selected runs

- [ ] **Step 1: Run the focused test suite**

Run: `pytest tests/test_eval_wandb_sidecar.py -v`

Expected: PASS.

- [ ] **Step 2: Update the uploader doc**

Document that W&B now receives `checkpoint_step`, `checkpoint_index`, and `epoch_progress`, and recommend `epoch_progress` for cross-run alignment when total steps differ.

- [ ] **Step 3: Rebuild uploader state for the selected models**

Delete only the local uploader state JSON files for:

```text
Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-sid-only-sft495
Instruments-grec-grpo-rule-only-dynamic-hint-sid-only-qwen2.5-3b-qwen4B-4-256-from-sft495
Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-sft495
Instruments-grec-grpo-rule-only-dynamic-hint-cascade-qwen2.5-3b-qwen4B-4-256-from-sft495
```

so the uploader replays every checkpoint for those runs.

- [ ] **Step 4: Re-upload the selected models once**

Run:

```bash
PYTHON_BIN=python bash eval_wandb_sidecar.sh once \
  --results-root ./results \
  --manifest-path ./results/.wandb_eval_manifest.json \
  --wandb-mode online \
  --model-filter Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-sid-only-sft495 \
  --model-filter Instruments-grec-grpo-rule-only-dynamic-hint-sid-only-qwen2.5-3b-qwen4B-4-256-from-sft495 \
  --model-filter Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-sft495 \
  --model-filter Instruments-grec-grpo-rule-only-dynamic-hint-cascade-qwen2.5-3b-qwen4B-4-256-from-sft495
```

Expected: uploader logs pending checkpoints for those models and exits cleanly after replaying them.
