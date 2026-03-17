# Dynamic Oracle Hint Rule-Only RL Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add runtime hint escalation to `rule_only` GRPO training so each sample rolls out without hint first, escalates to oracle hint depths 1..3 only when needed, and trains on the first successful stage or the deepest fallback stage.

**Architecture:** Extend the current hint-aware GRPO trainer with a stage-wise cascade loop. Each stage generates completions for the unresolved subset only, uses rule-hit presence as the stop signal, and then reassembles exactly one selected completion group per original sample before running the normal GRPO reward, log-prob, and advantage computation. Reuse the fixed-hint-aware constrained logits processor and the existing rule reward reconstruction path.

**Tech Stack:** Python, HuggingFace TRL/GRPOTrainer, existing constrained generation helpers, pytest/unittest, bash launcher scripts.

---

## Chunk 1: Tests And Control-Flow Contracts

### Task 1: Define cascade-selection behavior in tests

**Files:**
- Modify: `tests/test_fixed_hint_trainer_control_flow.py`
- Modify: `tests/test_fixed_hint_runtime.py`

- [ ] **Step 1: Write failing tests covering**
  - base rollout success stops escalation
  - base miss then hint-1 success escalates once
  - all stages miss and max-depth outputs are kept
  - runtime hint text is injected into reward reconstruction
- [ ] **Step 2: Run**
  - `python3 -m pytest tests/test_fixed_hint_trainer_control_flow.py tests/test_fixed_hint_runtime.py -q`
  - Expected: failing assertions for missing dynamic cascade behavior

## Chunk 2: Trainer And CLI Implementation

### Task 2: Implement dynamic cascade trainer

**Files:**
- Modify: `fixed_hint_grpo_trainer.py`

- [ ] **Step 1: Add small helper methods for**
  - building prompts with a requested hint depth
  - grouping stage rewards by sample
  - selecting final stage outputs back into original order
- [ ] **Step 2: Implement `DynamicHintRuleOnlyGRPOTrainer`**
  - run stage `0..max_hint_depth`
  - decode stage completions
  - detect `rule_hit_any`
  - stop carrying forward resolved samples
  - assemble selected outputs and run standard GRPO scoring
- [ ] **Step 3: Keep the existing fixed-hint trainer behavior unchanged**

### Task 3: Wire dynamic mode into the training entrypoint

**Files:**
- Modify: `trl_trainer.py`

- [ ] **Step 1: Add CLI args**
  - `dynamic_hint_max_depth`
  - `dynamic_hint_apply_to_eval`
- [ ] **Step 2: Validate combinations**
  - dynamic hint requires `reward_mode=rule_only`
  - dynamic hint and fixed hint map cannot both be enabled
- [ ] **Step 3: Route processor and trainer**
  - dynamic hint uses `build_fixed_hint_constrained_logits_processor`
  - instantiate `DynamicHintRuleOnlyGRPOTrainer`

## Chunk 3: Launcher

### Task 4: Add dynamic-hint training launcher

**Files:**
- Create: `hope/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-rule-only-dynamic-hint.sh`

- [ ] **Step 1: Copy the fixed-hint launcher structure**
- [ ] **Step 2: Remove offline analysis/export steps**
- [ ] **Step 3: Add overrides for**
  - `--dynamic-hint-max-depth`
  - `--dynamic-hint-apply-to-eval`
- [ ] **Step 4: Forward the new flags to `trl_trainer.py`**
- [ ] **Step 5: Add `--dry-run` output validation**

## Chunk 4: Verification

### Task 5: Run fresh verification

**Files:**
- Modify: `fixed_hint_grpo_trainer.py`
- Modify: `trl_trainer.py`
- Modify: `tests/test_fixed_hint_trainer_control_flow.py`
- Modify: `tests/test_fixed_hint_runtime.py`
- Create: `hope/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-rule-only-dynamic-hint.sh`

- [ ] **Step 1: Run**
  - `python3 -m pytest tests/test_fixed_hint_trainer_control_flow.py tests/test_fixed_hint_runtime.py -q`
- [ ] **Step 2: Run**
  - `python3 -m py_compile fixed_hint_grpo_trainer.py fixed_hint_utils.py trl_trainer.py`
- [ ] **Step 3: Run**
  - `bash -n hope/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-rule-only-dynamic-hint.sh`
- [ ] **Step 4: Run**
  - `bash hope/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-rule-only-dynamic-hint.sh --dry-run`
- [ ] **Step 5: Confirm the dry-run includes**
  - `--dynamic_hint_max_depth`
  - `--reward_mode rule_only`
  - no fixed-hint export stage

Plan complete and saved to `docs/superpowers/plans/2026-03-17-dynamic-oracle-hint-rule-only-rl.md`. Ready to execute.
