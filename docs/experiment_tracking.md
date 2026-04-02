# Experiment Tracking Guide

Use this guide when recording new experiments in `docs/`.

## Placement

- Put dated experiment and analysis notes at the top level of `docs/` with the format `YYYY-MM-DD-<slug>.md`.
- Reuse the existing dated note if you are continuing the same experiment line; do not create near-duplicate aliases.
- Update `docs/README.md` whenever you add or rename a top-level note.

## Source Of Truth

- Configuration entrypoints live under `hope/`.
- Local evaluation and training outputs live under `results/`.
- Small supporting assets that make a note self-contained can go under `docs/assets/<note-slug>/`.
- If results mainly exist on the remote machine, make the remote path explicit instead of assuming it is mirrored locally.

## Required Fields

Every experiment note should explicitly record:

- `Record date` and `Last updated`
- Goal or hypothesis
- Dataset, split, and task family
- Base model and base checkpoint
- Config script paths under `hope/`
- Important config overrides or deviations from the default script
- Result directories under `results/`
- Metric source files such as `metrics.json`, `summary.json`, `details.json`, or exported CSV/JSON tables
- Best checkpoint summary and the rule used to select it
- Key observations, regressions, and next actions

## Recommended Structure

```markdown
# <Title>

- Record date:
- Last updated:
- Goal:
- Current status:

## 1. Config

- Dataset / split:
- Base model / checkpoint:
- Hope scripts:
- Important overrides:

## 2. Result Paths

- Main result dir:
- Compared result dirs:
- Metric sources:

## 3. Key Metrics

| Variant | Best checkpoint | NDCG@10 | HR@10 | NDCG@50 | HR@50 | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |

## 4. Conclusions

- ...

## 5. Next Actions

- ...
```

## Writing Rules

- Prefer one document per experiment line or milestone instead of one document per command.
- Keep the narrative short, but make paths and selection criteria concrete.
- If a note becomes stale, mark it as a historical snapshot in `docs/README.md` instead of rewriting history.
- When you quote a result, make it traceable to a path under `results/` or an asset checked into `docs/assets/`.
