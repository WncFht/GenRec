# Dual Variant Eval Upload Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `eval_wandb_sidecar.py upload` write both `ckpt_step` and `epoch` W&B variants by default from the same manifest and results tree.

**Architecture:** Keep manifest generation as the single source of truth for base model metadata, then expand each base model into one or more upload variants at runtime. Each variant gets its own derived `run_id`, `run_name`, `wandb_group`, and state file, so dual uploads remain idempotent and independently resumable.

**Tech Stack:** Python 3.12, pytest, W&B public API conventions already used by the uploader, manifest + local state JSON

---

### Task 1: Add regression tests for dual-variant expansion

**Files:**
- Modify: `tests/test_eval_wandb_sidecar.py`

- [ ] **Step 1: Write a failing test for default variant parsing**

Add a test that feeds one manifest model and asserts upload-time expansion returns both `ckpt_step` and `epoch` variants when no explicit `--variants` override is passed.

- [ ] **Step 2: Write a failing test for derived epoch run naming**

Assert the epoch variant derives a new `run_id`, `run_name`, and `wandb_group="epoch"` while the ckpt-step variant keeps the original identifiers and uses `wandb_group="ckpt_step"`.

- [ ] **Step 3: Verify red**

Run: `WANDB_DISABLED=true uv run --with pytest python -m pytest tests/test_eval_wandb_sidecar.py -k variant -v`

Expected: FAIL because the uploader currently parses a single model spec per manifest row.

### Task 2: Implement variant-aware upload expansion

**Files:**
- Modify: `eval_wandb_sidecar.py`

- [ ] **Step 1: Introduce base-model parsing separate from upload-variant expansion**

Keep manifest parsing focused on base metadata, then add an expansion helper that emits one or more upload specs for `ckpt_step` and `epoch`.

- [ ] **Step 2: Make dual upload the CLI default**

Add `--variants` to `upload` with default `ckpt_step,epoch`, while still allowing callers to restrict to one variant.

- [ ] **Step 3: Derive variant-specific run ids, names, and groups**

Use the manifest row as the base `ckpt_step` spec. Derive the epoch variant by appending a stable suffix such as `-epoch` or a configurable override suffix, and assign `wandb_group` automatically if the manifest row did not already specify one for that variant.

- [ ] **Step 4: Preserve idempotence**

Ensure each variant still maps to a unique local state file through its distinct `run_id`.

### Task 3: Documentation and verification

**Files:**
- Modify: `docs/eval_wandb_sidecar.md`
- Modify: `tests/test_eval_wandb_sidecar.py`
- Modify: `eval_wandb_sidecar.py`

- [ ] **Step 1: Document the new default**

Explain that `upload` now writes both `ckpt_step` and `epoch` runs by default, and show how to restrict variants explicitly.

- [ ] **Step 2: Run focused verification**

Run:

```bash
WANDB_DISABLED=true uv run --with pytest python -m pytest tests/test_eval_wandb_sidecar.py -v
```

Expected: PASS.
