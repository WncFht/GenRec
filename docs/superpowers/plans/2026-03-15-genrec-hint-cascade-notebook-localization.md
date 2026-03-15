# GenRec Hint Cascade Notebook Localization Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `genrec-hint-cascade-analysis.ipynb` into a Chinese, tutorial-style analysis notebook that explains the cascade evaluation flow and key metrics for readers who have not run `analyze_rl_beam_hint.py`.

**Architecture:** Keep the notebook's existing analysis pipeline and data dependencies intact, but add Chinese narrative cells ahead of the analysis sections and localize the human-facing display strings inside the code cells. Preserve internal field names used for computation while translating rendered tables, plot titles, and summary text.

**Tech Stack:** Jupyter notebook JSON, Python notebook cells, pandas/matplotlib display logic

---

## Chunk 1: Notebook Structure And Reader Guidance

### Task 1: Add teaching-oriented Chinese front matter

**Files:**
- Modify: `/Users/fanghaotian/Desktop/src/GenRec/genrec-hint-cascade-analysis.ipynb`

- [ ] Replace the opening markdown cells with Chinese content that explains:
  - what `analyze_rl_beam_hint.py` produces
  - what `base -> hint_1 -> hint_2 -> hint_3` means
  - what `beam` and `hint_depth` mean
  - how to read the later tables and plots

- [ ] Insert one dedicated metric glossary markdown cell covering:
  - exact hit / rule hit
  - stage-local recovery vs cumulative recovery
  - prefix match length / prefix reward
  - `full` vs `suffix-only`
  - rollout-group pattern and all-zero group rate

## Chunk 2: Notebook Localization

### Task 2: Translate notebook display content

**Files:**
- Modify: `/Users/fanghaotian/Desktop/src/GenRec/genrec-hint-cascade-analysis.ipynb`

- [ ] Translate markdown section titles and explanatory bullets into Chinese.
- [ ] Translate plot titles, axis labels, and narrative summary strings into Chinese.
- [ ] Keep internal computation column names stable where needed, but rename displayed DataFrame columns into Chinese before rendering.

## Chunk 3: Output Consistency

### Task 3: Avoid stale English outputs

**Files:**
- Modify: `/Users/fanghaotian/Desktop/src/GenRec/genrec-hint-cascade-analysis.ipynb`

- [ ] Remove stale execution outputs that would contradict the updated Chinese notebook source if the notebook cannot be re-executed locally.
- [ ] Leave code runnable so the user can regenerate outputs later in their own environment.

## Chunk 4: Verification

### Task 4: Validate notebook JSON and inspect diff

**Files:**
- Modify: `/Users/fanghaotian/Desktop/src/GenRec/genrec-hint-cascade-analysis.ipynb`

- [ ] Run a JSON parse check on the notebook.
- [ ] Inspect the notebook cell structure to confirm the new Chinese guidance cells exist and key display strings were updated.
