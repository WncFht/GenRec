# Main Process Log Dedup Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce repeated multi-rank startup and runtime logs so the dynamic-hint training log is dominated by rank-0 user-facing output.

**Architecture:** Keep error reporting intact on all ranks, but route our own informational prints through a rank-aware helper and quiet non-main-process library loggers where safe. Touch only the local logging surface in launcher-adjacent modules and verify with focused unit tests plus syntax checks.

**Tech Stack:** Python, bash, HuggingFace TRL/Transformers, unittest.

---

### Task 1: Checkpoint the current dynamic-hint worktree state

**Files:**
- Modify: git index only

- [ ] **Step 1: Review `git status --short` to capture the current worktree scope**
- [ ] **Step 2: Stage the current worktree state that should be preserved as a checkpoint**
- [ ] **Step 3: Create a checkpoint commit before the log-dedup changes**

### Task 2: Lock main-process logging behavior in tests

**Files:**
- Modify: `tests/test_fixed_hint_runtime.py`
- Modify: `tests/test_trl_trainer_entrypoint.py`

- [ ] **Step 1: Write a failing test for rank-aware info printing helpers**
- [ ] **Step 2: Run the focused unittest command and confirm failure**
- [ ] **Step 3: Add a failing test for `trl_trainer.py` startup logging only emitting on rank 0**
- [ ] **Step 4: Re-run the focused unittest command and confirm the intended failure**

### Task 3: Implement rank-aware informational logging

**Files:**
- Modify: `util.py`
- Modify: `MIMIGenRec.py`
- Modify: `trl_trainer.py`

- [ ] **Step 1: Add a small helper for detecting/logging on the main process only**
- [ ] **Step 2: Route startup/config/trie informational prints through that helper**
- [ ] **Step 3: Quiet non-main-process library loggers where safe without suppressing errors**
- [ ] **Step 4: Re-run the focused unittest command and confirm pass**

### Task 4: Verify the log-dedup patch

**Files:**
- Modify: `util.py`
- Modify: `MIMIGenRec.py`
- Modify: `trl_trainer.py`
- Test: `tests/test_fixed_hint_runtime.py`
- Test: `tests/test_trl_trainer_entrypoint.py`

- [ ] **Step 1: Run `python3 -m unittest tests.test_fixed_hint_runtime tests.test_trl_trainer_entrypoint -v`**
- [ ] **Step 2: Run `python3 -m py_compile util.py MIMIGenRec.py trl_trainer.py fixed_hint_grpo_trainer.py`**
- [ ] **Step 3: Inspect output and only then report the actual verification status**
