# Eval Checkpoint Watcher Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn checkpoint evaluation into a long-running serial watcher that automatically discovers stable new checkpoints and evaluates them in the background.

**Architecture:** Keep the existing shell script as the operator-facing manager and move watch-loop logic into a new Python sidecar. The sidecar owns discovery, readiness checks, deterministic queue ordering, state persistence, and single-task execution, while `evaluate_sft_3b.sh` remains the evaluation executor.

**Tech Stack:** Bash, Python 3.12, `unittest`/`pytest`, JSON state files, existing evaluation shell scripts

---

### Task 1: Add failing watcher regression tests

**Files:**
- Modify: `tests/test_evaluate_all_checkpoints.py`

- [ ] **Step 1: Add a failing test for shard-complete readiness**

Create a checkpoint with `config.json` plus `model.safetensors.index.json` that references one missing shard, and assert the sidecar does not treat it as ready. Then add the missing shard and assert readiness becomes possible only after stability confirmation.

- [ ] **Step 2: Add a failing test for global serial ordering**

Create multiple model roots and checkpoints and assert the sidecar returns tasks sorted by `model_name`, then `checkpoint_step`.

- [ ] **Step 3: Add a failing test for sticky failure behavior**

Mark one task failed in watcher state and assert later scans skip it while still returning the next eligible task.

- [ ] **Step 4: Verify red**

Run:

```bash
python -m pytest tests/test_evaluate_all_checkpoints.py -v
```

Expected: FAIL because the current implementation has no watcher-side readiness or stateful queue logic.

### Task 2: Implement the Python watcher sidecar

**Files:**
- Create: `scripts/evaluate_all_checkpoints_sidecar.py`

- [ ] **Step 1: Extract reusable discovery and eval-profile helpers from the current shell behavior**

Implement model-root collection, filter handling, checkpoint parsing, and dataset/index profile resolution in Python so watch mode and one-shot mode share the same routing rules.

- [ ] **Step 2: Add stable-and-complete checkpoint detection**

Implement top-level file fingerprinting, stable poll counting, newest-file age checks, metadata/artifact checks, and index-shard completeness checks.

- [ ] **Step 3: Add persisted watch state**

Store failed tasks, current task, observation cache, and timestamps under `state/evaluate_all_checkpoints/watch_state.json`.

- [ ] **Step 4: Add serial execution loop**

Every poll, recompute eligible tasks, pick the first sorted task, invoke `evaluate_sft_3b.sh` for exactly that checkpoint, and persist success or failure.

- [ ] **Step 5: Add CLI commands for `once` and `watch`**

Support a one-pass mode for testing and a long-running poll loop for daemonized execution.

### Task 3: Refactor the shell entrypoint into a manager

**Files:**
- Modify: `scripts/evaluate_all_checkpoints.sh`

- [ ] **Step 1: Preserve backward-compatible one-shot invocation**

Keep `bash scripts/evaluate_all_checkpoints.sh` working as a one-shot evaluation pass.

- [ ] **Step 2: Add manager commands**

Add `run`, `start`, `stop`, `status`, and `tail`, mirroring the W&B sidecar workflow and routing the watch loop through the Python sidecar.

- [ ] **Step 3: Keep log/pid management explicit**

Write logs under `log/evaluate_all_checkpoints` and use per-instance pid files for safe background management.

### Task 4: Verify end to end

**Files:**
- Modify: `tests/test_evaluate_all_checkpoints.py`
- Modify: `scripts/evaluate_all_checkpoints.sh`
- Modify: `scripts/evaluate_all_checkpoints_sidecar.py`

- [ ] **Step 1: Run focused tests**

Run:

```bash
python -m pytest tests/test_evaluate_all_checkpoints.py -v
```

Expected: PASS.

- [ ] **Step 2: Run shell syntax validation**

Run:

```bash
bash -n scripts/evaluate_all_checkpoints.sh
```

Expected: PASS.

- [ ] **Step 3: Do a dry-run watcher smoke test**

Run:

```bash
python scripts/evaluate_all_checkpoints_sidecar.py once --dry-run
```

Expected: PASS with a scan summary and no execution crash.
