# Dynamic Hint Fastpath Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove avoidable dynamic-hint trainer overhead without changing step count, batch shape, or cascade semantics.

**Architecture:** Keep the existing cascade selection logic intact, but carry stage-level rule rewards forward so the final reward path can skip duplicate rule parsing in `reward_mode=rule_only`. Also gate prompt/completion string gathering behind the existing `log_completions` flag so observability work is only paid for when enabled.

**Tech Stack:** Python, TRL `GRPOTrainer`, unittest-based regression tests

---

## Chunk 1: Tests

### Task 1: Dynamic cascade carries selected rule rewards

**Files:**
- Modify: `tests/test_fixed_hint_trainer_control_flow.py`
- Test: `tests/test_fixed_hint_trainer_control_flow.py`

- [ ] Step 1: Write a failing test asserting `_run_dynamic_hint_cascade(...)` returns the selected stage rule rewards per sample.
- [ ] Step 2: Run `pytest tests/test_fixed_hint_trainer_control_flow.py -k selected_rule_rewards -v` and verify it fails for the missing key/value.
- [ ] Step 3: Implement the minimal trainer change to store and return the selected stage rewards.
- [ ] Step 4: Re-run the targeted test and verify it passes.

### Task 2: Text logging is skipped when `log_completions` is disabled

**Files:**
- Modify: `tests/test_fixed_hint_trainer_control_flow.py`
- Test: `tests/test_fixed_hint_trainer_control_flow.py`

- [ ] Step 1: Write a failing test asserting the trainer does not call `gather_object` for prompt/completion text when `log_completions=False`.
- [ ] Step 2: Run `pytest tests/test_fixed_hint_trainer_control_flow.py -k log_completions -v` and verify it fails.
- [ ] Step 3: Implement the minimal logging helper and wire it into the trainer.
- [ ] Step 4: Re-run the targeted test and verify it passes.

## Chunk 2: Implementation

### Task 3: Dynamic reward fast path

**Files:**
- Modify: `fixed_hint_grpo_trainer.py`
- Test: `tests/test_fixed_hint_trainer_control_flow.py`

- [ ] Step 1: Write a failing test asserting the dynamic trainer reuses selected cascade rule rewards instead of calling the generic reward path in `rule_only`.
- [ ] Step 2: Run `pytest tests/test_fixed_hint_trainer_control_flow.py -k rule_only_fast_path -v` and verify it fails.
- [ ] Step 3: Implement the minimal `rule_only` fast path and remove the redundant post-`_calculate_rewards` gather in the dynamic trainer.
- [ ] Step 4: Run the focused trainer test file and verify all related cases pass.

### Task 4: Verification

**Files:**
- Modify: `fixed_hint_grpo_trainer.py`
- Test: `tests/test_fixed_hint_trainer_control_flow.py`

- [ ] Step 1: Run `pytest tests/test_fixed_hint_trainer_control_flow.py -v`.
- [ ] Step 2: Inspect failures or warnings and adjust only if needed.
- [ ] Step 3: Report the verified scope, any residual risks, and the commands run.
