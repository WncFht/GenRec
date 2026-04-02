# Eval Checkpoint Watcher Design

## Context

Current [`scripts/evaluate_all_checkpoints.sh`](/Users/fanghaotian/Desktop/src/GenRec/scripts/evaluate_all_checkpoints.sh) is a one-shot scanner:

- it discovers model roots under `SFT_ROOT` and `RL_ROOT`
- it collects missing `results/<model>/checkpoint-*/metrics.json`
- it invokes [`evaluate_sft_3b.sh`](/Users/fanghaotian/Desktop/src/GenRec/evaluate_sft_3b.sh) once per model root
- it exits when the current pass finishes

That is useful for batch backfills, but it does not satisfy the remote-training workflow where new model directories and new `checkpoint-*` directories appear over time and should be evaluated automatically in the background.

## Goals

- Keep a long-running watcher on the remote training machine.
- Discover both:
  - newly created model directories under `SFT_ROOT` / `RL_ROOT`
  - newly created `checkpoint-*` directories inside existing model directories
- Run evaluation strictly serially: one checkpoint task at a time.
- Use a deterministic global order: `model_name` ascending, then `checkpoint_step` ascending.
- Only queue checkpoints after they are judged stable and complete enough for evaluation.
- Persist watcher state so restarts do not lose:
  - failure records
  - readiness observations
  - the current in-progress task
- Continue past failures without automatic retry.
- Preserve a simple one-shot mode for the existing backfill workflow.

## Non-Goals

- Multi-GPU task scheduling across multiple concurrent evaluations.
- Automatic retry for failed checkpoints.
- Training-side marker file requirements.
- Replacing the evaluation logic in [`evaluate_sft_3b.sh`](/Users/fanghaotian/Desktop/src/GenRec/evaluate_sft_3b.sh).

## Options Considered

### Option A: Turn the current bash script into an infinite `while true` loop

- Pros: smallest surface-area change.
- Cons: hard to test, awkward state persistence, fragile stability heuristics, and difficult process management.

### Option B: Add a Python sidecar and keep the bash script as a process manager

- Pros: watcher logic becomes testable, state can be structured JSON, process management can mirror the existing W&B sidecar pattern, and one-shot behavior can remain intact.
- Cons: introduces one new script.

### Option C: Build a filesystem-event watcher

- Pros: reacts quickly to changes.
- Cons: more OS-specific behavior, harder to reason about on remote training boxes, and still needs a persistence layer for failures and partial observations.

## Chosen Approach

Use Option B.

The existing shell entrypoint becomes a manager with commands such as:

- `once`: run the legacy one-shot scan
- `run`: foreground watch loop
- `start`: background watch loop
- `stop`
- `status`
- `tail`

The long-running logic moves into a Python sidecar that:

- scans candidate model roots every poll interval
- builds a global checkpoint task list
- applies stability checks before a task is eligible
- executes exactly one task at a time
- writes JSON state after every meaningful transition

## Detailed Behavior

### 1. Global task model

The watcher treats each `checkpoint-*` as one task, not one whole model root.

Task identity:

- `root_kind`: `sft` or `rl`
- `model_root`
- `model_name`
- `checkpoint_name`
- `checkpoint_step`

Why per-checkpoint:

- serial execution becomes explicit
- failure handling is precise
- ordering across models is deterministic
- a model with many pending checkpoints does not need special batching logic inside the watcher

### 2. Stable-and-complete readiness gate

A checkpoint is eligible only when all of the following are true:

1. The checkpoint directory exists and its name parses as `checkpoint-<step>`.
2. The checkpoint does not already have `results/<model>/<checkpoint>/metrics.json`, unless `FORCE_REEVAL=1`.
3. The checkpoint is not marked failed in watcher state.
4. The checkpoint passes the completeness check:
   - at least one model metadata file exists:
     - `config.json` or `adapter_config.json`
   - at least one model weight artifact is usable:
     - `model.safetensors`
     - `model.safetensors.index.json` with all referenced shard files present
     - `pytorch_model.bin`
     - `pytorch_model.bin.index.json` with all referenced shard files present
     - `adapter_model.safetensors`
     - `adapter_model.bin`
5. The checkpoint passes the stability check:
   - the sidecar fingerprints the top-level checkpoint files using file name, size, and mtime
   - the fingerprint must remain unchanged for `STABLE_CONFIRMATION_POLLS` consecutive scans
   - the newest top-level file mtime must be at least `STABLE_AGE_SECONDS` in the past

This is intentionally conservative. A directory that is still being written should keep changing fingerprint or have a too-recent newest file mtime, so it will not enter the queue prematurely.

### 3. Queue ordering

Every poll, the sidecar computes all eligible tasks and sorts them by:

1. `model_name`
2. `checkpoint_step`
3. `checkpoint_name` as a final tie-breaker

The watcher then picks the first task only. After it finishes, the next poll recomputes the queue from filesystem truth plus persisted failure state.

### 4. Execution

For each selected task, the sidecar resolves the same eval profile logic currently implemented in [`scripts/evaluate_all_checkpoints.sh`](/Users/fanghaotian/Desktop/src/GenRec/scripts/evaluate_all_checkpoints.sh):

- category inference
- dataset/index path mapping
- cb-width-aware variant selection

Execution remains delegated to [`evaluate_sft_3b.sh`](/Users/fanghaotian/Desktop/src/GenRec/evaluate_sft_3b.sh) with:

- `CKPT_LIST=<checkpoint_name>`
- `CATEGORY`
- `TEST_DATA_PATH`
- `INDEX_PATH`
- `CUDA_LIST`
- `PYTHON_BIN`

That preserves current evaluation behavior and keeps the sidecar focused on scheduling rather than model inference details.

### 5. Failure semantics

If one checkpoint evaluation command exits non-zero:

- record it in state under a stable task key
- store timestamp, command, and exit code
- mark the task as failed
- continue to the next eligible task on later polls

The watcher never auto-retries a failed checkpoint. A human can recover by:

- deleting the failed entry from the state file, or
- using a future explicit reset helper if needed

### 6. State layout

Default state directory:

- `state/evaluate_all_checkpoints`

State file responsibilities:

- `watch_state.json`
  - observation cache per checkpoint
  - failed task records
  - current task metadata
  - last scan / last execution timestamps

Observation entries contain the last fingerprint, stable poll count, and newest file mtime used for stability decisions.

### 7. Manager process layout

Default manager directory:

- `log/evaluate_all_checkpoints`

Manager files:

- `<instance>.pid`
- `<instance>.log`

This mirrors the existing [`eval_wandb_sidecar.sh`](/Users/fanghaotian/Desktop/src/GenRec/eval_wandb_sidecar.sh) interface so the remote workflow stays consistent.

## Files To Change

- Modify [`scripts/evaluate_all_checkpoints.sh`](/Users/fanghaotian/Desktop/src/GenRec/scripts/evaluate_all_checkpoints.sh)
- Add `scripts/evaluate_all_checkpoints_sidecar.py`
- Modify [`tests/test_evaluate_all_checkpoints.py`](/Users/fanghaotian/Desktop/src/GenRec/tests/test_evaluate_all_checkpoints.py)

## Risks

### Readiness heuristic too strict

Some checkpoint formats may be valid but not match the first version of the completeness heuristic. This is acceptable as long as the heuristic is explicit and test-covered; expanding recognized artifacts later is safer than eagerly running on half-written checkpoints.

### Readiness heuristic too loose

If the sidecar admits a checkpoint before shard writes finish, evaluation can fail. The fingerprint-plus-age gate reduces this risk materially compared with checking only directory existence.

### State drift after manual edits

Because failed checkpoints are intentionally sticky, manual operator edits to the state file may be needed after environment failures. This is acceptable for the first version.

## Validation Strategy

1. Add regression tests for global task ordering.
2. Add regression tests for stable-and-complete readiness.
3. Add regression tests for index-sharded weight completeness.
4. Add regression tests proving failed checkpoints are skipped on later scans.
5. Run focused pytest on the evaluation watcher tests.
6. Run shell syntax validation on the manager script.
