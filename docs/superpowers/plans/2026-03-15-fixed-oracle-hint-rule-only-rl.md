# Fixed Oracle Hint Rule-Only RL Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train `rule_only` RL with a fixed per-sample oracle hint depth derived from offline `beam=16` cascade analysis, while keeping mixed-depth samples in a single GRPO generation pass.

**Architecture:** Export a stable `extra_info.index -> hint_depth` map from the analyzer, load it into RL data examples, and route each mixed-depth training batch through a custom GRPO trainer that appends each sample's SID prefix hint directly to its prompt, performs a single `_generate(...)` call per step, then reconstructs full completions before computing rule reward. Reuse upstream GRPO reward gathering and advantage normalization so multi-rank synchronization stays aligned with vanilla TRL behavior.

**Tech Stack:** Python, HuggingFace TRL/GRPOTrainer, existing constrained beam search utilities, bash launcher scripts, pytest.

---

### Task 1: Lock export format with tests

**Files:**
- Modify: `tests/test_analyze_rl_beam_hint.py`
- Modify: `analyze_rl_beam_hint.py`

- [ ] **Step 1: Write failing tests for fixed hint depth map export and loading helpers**
- [ ] **Step 2: Run `python -m pytest tests/test_analyze_rl_beam_hint.py -q` and confirm failure**
- [ ] **Step 3: Implement minimal analyzer helpers for exporting/loading per-sample fixed depth maps**
- [ ] **Step 4: Re-run `python -m pytest tests/test_analyze_rl_beam_hint.py -q` and confirm pass**

### Task 2: Add fixed-hint single-generate rule-only trainer

**Files:**
- Create: `fixed_hint_grpo_trainer.py`
- Modify: `tests/test_analyze_rl_beam_hint.py`
- Modify: `trl_trainer.py`

- [ ] **Step 1: Write failing unit tests for single-call mixed-depth generation, prompt hint injection, and full completion reconstruction**
- [ ] **Step 2: Run targeted pytest and confirm failure**
- [ ] **Step 3: Implement `FixedHintRuleOnlyGRPOTrainer` with a single mixed-depth `_generate(...)` pass**
- [ ] **Step 4: Re-run targeted pytest and confirm pass**

### Task 3: Wire CLI and launcher

**Files:**
- Modify: `trl_trainer.py`
- Modify: `hope/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-rule-only.sh`

- [ ] **Step 1: Add CLI args for fixed hint map path, cap, and unsolved depth**
- [ ] **Step 2: Forward shell args to `trl_trainer.py`**
- [ ] **Step 3: Add a dry-run validation command for the launcher**
- [ ] **Step 4: Run launcher `--dry-run` and confirm the new args appear**

### Task 4: Verify end-to-end integrity

**Files:**
- Modify: `analyze_rl_beam_hint.py`
- Modify: `fixed_hint_grpo_trainer.py`
- Modify: `trl_trainer.py`
- Modify: `hope/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-rule-only.sh`
- Test: `tests/test_analyze_rl_beam_hint.py`

- [ ] **Step 1: Run `python -m pytest tests/test_analyze_rl_beam_hint.py -q`**
- [ ] **Step 2: Run `python -m py_compile analyze_rl_beam_hint.py fixed_hint_grpo_trainer.py trl_trainer.py fixed_hint_utils.py`**
- [ ] **Step 3: Run `bash -n hope/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-rule-only.sh`**
- [ ] **Step 4: Run the rule-only launcher in `--dry-run` mode with a fixed hint map path**
