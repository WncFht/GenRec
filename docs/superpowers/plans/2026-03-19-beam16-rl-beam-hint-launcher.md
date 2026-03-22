# Beam16 RL Beam Hint Launcher Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the RL beam-hint analyzer launcher default to `beam=16` only while preserving explicit `--beam-sizes` overrides.

**Architecture:** Keep the change local to the launcher script and its dry-run contract. Add a regression test that executes the shell script in `--dry-run` mode and asserts the default forwarded beam size is `16`, plus a second test that confirms manual overrides still work.

**Tech Stack:** Bash launcher, Python `unittest` shell invocation tests

---

## Chunk 1: Launcher Default And Regression Test

### Task 1: Lock the desired launcher behavior with tests

**Files:**
- Modify: `tests/test_trl_trainer_entrypoint.py`
- Test: `tests/test_trl_trainer_entrypoint.py`

- [ ] **Step 1: Write the failing test**

```python
def test_analyze_beam_hint_shell_dry_run_defaults_to_beam_16_only(self):
    result = subprocess.run(
        ["bash", str(ANALYZE_BEAM_HINT_SCRIPT), "--dry-run"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    self.assertEqual(result.returncode, 0, msg=result.stderr)
    self.assertIn("--beam-sizes 16", result.stdout)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest -q tests/test_trl_trainer_entrypoint.py -k analyze_beam_hint`
Expected: FAIL because the current script still forwards `--beam-sizes 8,16`

- [ ] **Step 3: Write minimal implementation**

```bash
BEAM_SIZES="${BEAM_SIZES:-16}"
```

Also update the usage text from `Default: 8,16` to `Default: 16`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest -q tests/test_trl_trainer_entrypoint.py -k analyze_beam_hint`
Expected: PASS

- [ ] **Step 5: Run direct launcher dry-run verification**

Run: `bash hope/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-analyze-rl-beam-hint.sh --dry-run`
Expected: printed command contains `--beam-sizes 16`
