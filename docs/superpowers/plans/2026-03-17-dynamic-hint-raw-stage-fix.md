# Dynamic Hint Raw Stage Fix Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the dynamic-hint multi-stage deadlock risk by keeping stage retries out of TRL's distributed `_generate()` wrapper while preserving the current first-hit cascade behavior.

**Architecture:** Keep `DynamicHintRuleOnlyGRPOTrainer` and its unresolved-subset cascade semantics, but switch each stage to TRL's raw single-turn generation path instead of repeatedly calling `_generate()`. After the cascade selects one group per original prompt, compute the final completion-token normalizer from the selected outputs and continue with the existing GRPO reward/log-prob/advantage path.

**Tech Stack:** Python, HuggingFace TRL/GRPOTrainer, pytest.

---

### Task 1: Lock the new dynamic cascade contract in tests

**Files:**
- Modify: `tests/test_fixed_hint_trainer_control_flow.py`

- [ ] **Step 1: Add a failing test showing dynamic cascade uses `_generate_single_turn(...)` instead of `_generate(...)` for stage retries**
- [ ] **Step 2: Run `python3 -m pytest tests/test_fixed_hint_trainer_control_flow.py -q` and confirm failure**
- [ ] **Step 3: Add a failing assertion that the cascade reports completion-token count from the final selected outputs rather than the first stage**
- [ ] **Step 4: Re-run `python3 -m pytest tests/test_fixed_hint_trainer_control_flow.py -q` and confirm failure for the intended reason**

### Task 2: Implement the raw-stage cascade path

**Files:**
- Modify: `fixed_hint_grpo_trainer.py`

- [ ] **Step 1: Add a small dynamic-stage helper that calls `_generate_single_turn(...)` and enforces the current text-only limitations**
- [ ] **Step 2: Update `_run_dynamic_hint_cascade(...)` to use the raw-stage helper while keeping unresolved-subset selection behavior unchanged**
- [ ] **Step 3: Compute the final completion-token normalizer from the selected outputs after cascade assembly**
- [ ] **Step 4: Re-run `python3 -m pytest tests/test_fixed_hint_trainer_control_flow.py -q` and confirm pass**

### Task 3: Verify the trainer entry points still hold

**Files:**
- Modify: `fixed_hint_grpo_trainer.py`
- Test: `tests/test_fixed_hint_trainer_control_flow.py`
- Test: `tests/test_trl_trainer_entrypoint.py`

- [ ] **Step 1: Run `python3 -m pytest tests/test_fixed_hint_trainer_control_flow.py tests/test_trl_trainer_entrypoint.py -q`**
- [ ] **Step 2: Run `python3 -m py_compile fixed_hint_grpo_trainer.py trl_trainer.py`**
- [ ] **Step 3: Inspect the output and only then report the actual verification status**
