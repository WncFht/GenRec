# Dynamic Oracle Hint Rule-Only RL Design

## Context

Current `fixed_hint` RL injects an offline `extra_info.index -> oracle_hint_depth` map into the dataset, then appends the corresponding oracle SID prefix hint to each prompt before a single GRPO rollout. That is useful for an oracle-scaffold upper bound, but it cannot answer the online question:

> Under the current policy, how much hint strength is actually needed before a sample can produce at least one rule hit?

The requested behavior is online cascade generation during RL:

1. Roll out `num_generations=16` with no hint.
2. If none of the 16 completions hit the rule, append hint depth 1 and roll out again.
3. If still no hit, try hint depth 2.
4. If still no hit, try hint depth 3.
5. Stop at the first successful stage, or keep the last stage if all fail.

## Goals

- Keep GRPO training semantics: each original sample contributes one final group of `num_generations` completions.
- Make hint strength depend on the current model rollout, not on offline preprocessing.
- Reuse existing rule reward and constrained generation code where possible.
- Preserve a training launcher interface similar to the existing fixed-hint shell script.

## Non-Goals

- Supporting dynamic hint cascade for reward modes other than `rule_only`.
- Learning the hint policy itself.
- Replacing the offline analyzer. The analyzer remains the source of truth for post-hoc cascade evaluation.

## Options Considered

### Option A: Keep offline fixed hint depth

Use the current `fixed_hint_depth_map` path and do not change runtime behavior.

- Pros: simplest, already implemented, stable.
- Cons: does not solve the requested online hint-strength decision problem.

### Option B: One mixed `_generate()` pass with per-sample hint depth

Compute each sample's current hint depth inside the trainer, then generate all samples together in a single `_generate()` call.

- Pros: fewer generation calls.
- Cons: awkward control flow for staged stopping, harder to reason about, and current repository notes already identify mixed-depth runtime generation as the most fragile part.

### Option C: Stage-wise runtime cascade

Run generation in stages: `base`, `hint_1`, `hint_2`, `hint_3`. At each stage, only unresolved samples continue.

- Pros: matches `analyze_rl_beam_hint.py`, easy to inspect, easy to log, and has a clean stopping rule.
- Cons: more generate calls when many samples are hard.

## Chosen Approach

Use Option C.

For each batch item, the trainer will keep exactly one selected stage:

- `hint_depth=0` if base rollout already contains a rule hit.
- Otherwise the first hint depth in `[1, 2, 3]` whose rollout contains a rule hit.
- Otherwise `hint_depth=3`.

The final GRPO reward and advantage computation will use only the selected stage's completions.

## Detailed Behavior

### 1. Stage execution

For a local batch of `N` prompts:

- Stage `0`:
  - build prompts without hint text
  - generate `N * num_generations` completions
  - decode completions and check per-sample `rule_hit_any`
- Stage `d in [1, max_hint_depth]`:
  - only carry forward samples that failed stage `d-1`
  - append the first `d` SID tokens from `reward_model.ground_truth`
  - generate a fresh group of completions for that unresolved subset
  - stop carrying a sample forward once its stage has any rule hit

### 2. Final batch assembly

After all stages:

- reorder selected stage outputs back to the original sample order
- keep exactly `num_generations` completions per original sample
- pad `prompt_ids` and `completion_ids` across the full batch
- compute log-probs and advantages exactly once on the selected outputs

### 3. Reward reconstruction

The existing `rule_reward` already reconstructs a full completion when `oracle_hint_text` is present in reward kwargs:

- selected stage hint text is injected into the reward inputs at runtime
- reward code sees `hint_text + generated_suffix`
- no reward formula rewrite is needed

### 4. Constrained generation

Dynamic hint stages should use `FixedHintConstrainedLogitsProcessor`, not the old count-based processor, because prompt text now legitimately contains hint tokens after the generation prefix. This keeps trie continuation aligned with the current prompt suffix plus injected hint.

### 5. Logging

Per batch, log at least:

- mean selected hint depth
- fraction of samples resolved at base stage
- fraction of samples first resolved at hint depths 1, 2, 3
- fraction of samples still unresolved after max hint depth

This will let later analysis compare online effective hint depth against offline cascade findings.

## Code Changes

### `fixed_hint_grpo_trainer.py`

- Keep the existing fixed-hint trainer.
- Add a new dynamic cascade trainer for `rule_only`.
- Factor shared prompt/hint helpers into local utility methods where needed.

### `fixed_hint_utils.py`

- Reuse `build_hint_text`.
- Add small runtime helpers only if they reduce complexity; avoid creating a parallel helper module unless necessary.

### `trl_trainer.py`

- Add CLI args for dynamic hint enablement and max depth.
- Route dynamic hint mode to the new trainer.
- Force the fixed-hint-compatible logits processor when dynamic hint mode is enabled.
- Reject incompatible combinations such as `reward_mode != rule_only`.

### `hope/...-rl-rule-only-dynamic-hint.sh`

- Mirror the existing fixed-hint launcher shape.
- Remove the offline export step.
- Forward dynamic hint args to `trl_trainer.py`.

## Risks

### More generation work

Hard samples can trigger up to 4 generation stages. This is expected. The tradeoff is acceptable because the method intentionally measures how much scaffold the current policy needs.

### Mixed prompt lengths across selected outputs

This is safe as long as selected `prompt_ids` and `completion_ids` are padded after final assembly, which matches the current fixed-hint trainer pattern.

### Ambiguity in stopping criterion for non-rule rewards

Avoided by explicitly limiting dynamic hint mode to `reward_mode=rule_only`.

## Validation Strategy

1. Add unit tests for stage selection:
   - base-stage success stops escalation
   - repeated failures escalate to deeper hints
   - all-fail case keeps max-depth outputs
2. Add tests for prompt construction and reward reconstruction with runtime hint text.
3. Add CLI tests or dry-run validation for the new launcher arguments.
4. Run targeted pytest plus `py_compile` and shell syntax checks.

## Worktree Note

The superpowers workflow usually prefers a fresh git worktree before plan execution. In this repository, no project-local worktree directory exists and none is ignored. Creating one would require `.gitignore` and branch-management changes, while the user explicitly requested deliverables at the current workspace paths. For this task, implementation proceeds in-place.
