# GenRec Hint Local Bundle Findings (2026-03-16)

This note records a local re-analysis of the bundled `Instruments-grec` hint artifacts under:

- `/Users/fanghaotian/Desktop/src/GenRec/output/local-research-bundles/instruments_grec_hint_research_bundle`

The goal was not to repeat the existing notebook narrative, but to test concrete hypotheses about what `hint depth` is actually tracking.

## Scope

Data used:

- bundled `train.json`
- bundled `id2sid.json`
- bundled `summary/details.json`
- bundled `instruments_grec_beam16_hint_difficulty_table.csv`
- bundled `Instruments.item.json` and `Instruments.inter.json`

Reusable code added for this pass:

- [explore_local_hint_bundle.py](/Users/fanghaotian/Desktop/src/GenRec/scripts/hint_research/explore_local_hint_bundle.py)

Generated outputs:

- `/Users/fanghaotian/Desktop/src/GenRec/output/local-research-bundles/instruments_grec_hint_research_bundle/analysis/hint_transition_summary.json`
- `/Users/fanghaotian/Desktop/src/GenRec/output/local-research-bundles/instruments_grec_hint_research_bundle/analysis/hint_transition_feature_thresholds.csv`
- `/Users/fanghaotian/Desktop/src/GenRec/output/local-research-bundles/instruments_grec_hint_research_bundle/analysis/hint_transition_top_groups.csv`
- `/Users/fanghaotian/Desktop/src/GenRec/output/local-research-bundles/instruments_grec_hint_research_bundle/analysis/shared_transition_hotspots.csv`
- `/Users/fanghaotian/Desktop/src/GenRec/output/local-research-bundles/instruments_grec_hint_research_bundle/analysis/transition_task_gap_groups.csv`
- `/Users/fanghaotian/Desktop/src/GenRec/output/local-research-bundles/instruments_grec_hint_research_bundle/analysis/hint_transition_residual_cases.csv`
- `/Users/fanghaotian/Desktop/src/GenRec/output/local-research-bundles/instruments_grec_hint_research_bundle/analysis/dominant_residual_parent_items.csv`
- `/Users/fanghaotian/Desktop/src/GenRec/output/local-research-bundles/instruments_grec_hint_research_bundle/analysis/dominant_residual_parent_leaf_stats.csv`
- `/Users/fanghaotian/Desktop/src/GenRec/output/local-research-bundles/instruments_grec_hint_research_bundle/analysis/dominant_residual_parent_title_terms.csv`

## Method Correction

Two analysis pitfalls in the existing notebook needed correction before drawing conclusions:

1. `required_*` features mix different decision depths.
   If `effective_hint_depth=1`, the decisive token is depth 1; if `effective_hint_depth=2`, the decisive token is depth 2. Comparing these rows directly can create artificial separation.

2. The existing `deep-hint token lift` table is not directly interpretable.
   In practice it compares different token levels together, e.g. many `hint_1` rows contribute `<a_*>`, while many deep-hint rows contribute `<b_*>` or `<c_*>`. A `share_lift=1.0` result there is not evidence of "no concentration"; it is mostly a grouping mismatch.

For this re-analysis I switched to stage-aligned transitions:

- `base -> need_hint` uses depth-1 features
- `hint_1 -> need_hint_2+` uses depth-2 features
- `hint_2 -> need_hint_3+` uses depth-3 features
- `hint_3 -> unsolved` uses depth-4 features

This aligns each prediction target with the token that would actually be revealed by the next hint.

## Hypothesis 1

Task difficulty is not fully explained by local tree structure.

Result: supported.

Raw task gap:

- `sid`: `need_hint_rate = 71.75%`, `hint2_plus_rate = 32.70%`
- `hisTitle2sid`: `need_hint_rate = 91.31%`, `hint2_plus_rate = 61.24%`

After matching `sid` and `hisTitle2sid` on stage-aligned local structure bins:

- `base -> need_hint`
  - raw: `sid 0.717`, `hisTitle2sid 0.913`
  - standardized: `sid 0.788`, `hisTitle2sid 0.957`
- `hint_1 -> need_hint_2+`
  - raw: `sid 0.456`, `hisTitle2sid 0.671`
  - standardized: `sid 0.460`, `hisTitle2sid 0.711`
- `hint_2 -> need_hint_3+`
  - raw: `sid 0.0106`, `hisTitle2sid 0.0127`
  - standardized: `sid 0.0223`, `hisTitle2sid 0.0287`

Interpretation:

- local branch structure matters, but it does not explain away the task gap
- `hisTitle2sid` remains harder even after matching on local frequency/share/subtree bins
- the textual input channel itself likely contributes independent ambiguity

## Hypothesis 2

The meaning of deeper hint changes by stage.

Result: supported.

### `base -> need_hint`

Best single feature:

- `task_parent_share_d1`, balanced accuracy `0.601`

Other useful but weaker signals:

- `global_count_d1`, balanced accuracy `0.591`
- `subtree_d1`, balanced accuracy `0.575`

Sequence features are weak here:

- best sequence metric only reaches balanced accuracy about `0.526`

Interpretation:

- failing without hint is partly a first-token routing problem
- the strongest signal is not sequence position
- the first token is harder when it is less dominant under the task-conditioned root distribution

### `hint_1 -> need_hint_2+`

Best single features:

- `task_parent_share_d2`, balanced accuracy `0.620`
- `sibling_d2`, balanced accuracy `0.592`
- `parent_entropy_d2`, balanced accuracy `0.590`

Interpretation:

- once the first token is revealed, the next difficulty is mainly whether the second token is locally dominant under that revealed prefix
- this is where the problem starts to look like local codebook competition, not general sequence difficulty

### `hint_2 -> need_hint_3+`

Best single features:

- `sibling_d3`, balanced accuracy `0.838`
- `task_parent_share_d3`, balanced accuracy `0.761`
- `child_share_d3`, balanced accuracy `0.748`
- `subtree_d3`, balanced accuracy `0.726`

Interpretation:

- needing `hint_3` is a sharply localized depth-3 branching problem
- it is strongly associated with large sibling sets and weak local dominance
- by this stage, task-level sequence effects are no longer the main story

### `hint_3 -> unsolved`

Best single features:

- `child_share_d4 <= 0.0476`, balanced accuracy `0.987`
- `sibling_d4 >= 21`, balanced accuracy `0.987`
- `parent_maxshare_d4 <= 0.0476`, balanced accuracy `0.987`

Interpretation:

- the final residual set is almost entirely a "last leaf among many near-equal siblings" problem
- this is not broad model weakness; it is concentrated leaf-level ambiguity

## Hypothesis 3

Residual hard cases are concentrated, not diffuse.

Result: strongly supported.

Key finding:

- all `11` final unsolved cases share the same depth-4 parent prefix: `<a_65><b_80><c_183>`

That parent contains `21` leaf items.

Within that parent:

- `<d_100>` appears `6` times and is unsolved every time
- `<d_229>` appears `1` time and is unsolved
- `<d_88>`, `<d_137>`, `<d_239>`, `<d_1>` are the other residual leaves

The exported [dominant_residual_parent_items.csv](/Users/fanghaotian/Desktop/src/GenRec/output/local-research-bundles/instruments_grec_hint_research_bundle/analysis/dominant_residual_parent_items.csv) shows that this is not a random bucket. It is a very tight `Walker & Williams` guitar-strap family.

This means the final residual set is not random long-tail noise. It is a single pathological branch.

## Concentrated Groups Before Residual Failure

The strongest over-concentrated groups before deeper hint are also branch-specific:

### `hint_1 -> need_hint_2+`

High-risk `(parent_d2, token_d2)` groups include:

- `(<a_157>, <b_8>)`, rate `1.000`
- `(<a_157>, <b_249>)`, rate `1.000`
- `(<a_198>, <b_153>)`, rate `1.000`
- `(<a_157>, <b_40>)`, rate `1.000`
- `(<a_241>, <b_56>)`, rate `1.000`

### `hint_2 -> need_hint_3+`

High-risk `(parent_d3, token_d3)` groups include:

- `(<a_253><b_240>, <c_59>)`, rate `0.679`
- `(<a_253><b_240>, <c_49>)`, rate `0.652`
- `(<a_241><b_145>, <c_84>)`, rate `0.562`
- `(<a_125><b_20>, <c_222>)`, rate `0.533`
- `(<a_65><b_80>, <c_210>)`, rate `0.346`
- `(<a_65><b_80>, <c_183>)`, rate `0.250`

This supports a branch-cascade view:

- some root children are more likely to require a second hint
- some depth-3 prefixes are much more likely to require a third hint
- one depth-4 parent prefix dominates the final unsolved set

## Shared Hotspots vs Task-Specific Gaps

The new exports separate two different questions that were mixed together before:

1. which branches are jointly hard for both `sid` and `hisTitle2sid`
2. which branches are disproportionately harder for `hisTitle2sid`

### Shared hotspots

The new [shared_transition_hotspots.csv](/Users/fanghaotian/Desktop/src/GenRec/output/local-research-bundles/instruments_grec_hint_research_bundle/analysis/shared_transition_hotspots.csv) shows that many difficult branches are genuinely shared across tasks.

At `base -> need_hint`, shared high-risk root tokens include:

- `<a_105>`
- `<a_32>`
- `<a_173>`
- `<a_19>`
- `<a_198>`
- `<a_65>`
- `<a_234>`
- `<a_157>`

At `hint_1 -> need_hint_2+`, the strongest shared local hotspots include:

- `(<a_253>, <b_240>)`
- `(<a_241>, <b_197>)`
- `(<a_157>, <b_153>)`
- `(<a_194>, <b_247>)`
- `(<a_194>, <b_58>)`
- `(<a_65>, <b_2>)`

At `hint_2 -> need_hint_3+`, the strongest shared hotspot is:

- `(<a_241><b_197>, <c_210>)`

with very large lift in both tasks:

- `sid`: lift about `12.43`
- `hisTitle2sid`: lift about `15.70`

Interpretation:

- the two tasks do not fail on disjoint SID regions
- they repeatedly collide on a shared hard-tree backbone
- this supports the view that local SID-branch competition is a common mechanism, not just a task-specific artifact

### Task-specific gap groups

The new [transition_task_gap_groups.csv](/Users/fanghaotian/Desktop/src/GenRec/output/local-research-bundles/instruments_grec_hint_research_bundle/analysis/transition_task_gap_groups.csv) isolates where `hisTitle2sid` is much worse than `sid` even inside the same local group.

For `base -> need_hint`, the biggest gaps appear at the root:

- `<a_77>`: gap about `+0.394`
- `<a_241>`: gap about `+0.321`
- `<a_59>`: gap about `+0.312`
- `<a_253>`: gap about `+0.292`
- `<a_125>`: gap about `+0.292`

For `hint_1 -> need_hint_2+`, the strongest local gaps are sharper and more branch-specific:

- `(<a_65>, <b_194>)`: gap about `+0.629`
- `(<a_125>, <b_212>)`: gap about `+0.589`
- `(<a_192>, <b_246>)`: gap about `+0.567`
- `(<a_77>, <b_254>)`: gap about `+0.548`
- `(<a_241>, <b_170>)`: gap about `+0.536`

Interpretation:

- the global task gap is not just "hisTitle2sid is harder everywhere"
- there are specific local branches where title-history inputs degrade much more sharply than SID-history inputs
- a realistic policy would likely need both:
  - a shared branch-risk prior
  - a task-specific escalation prior

## Dominant Residual Branch Anatomy

The earlier pass already showed that the final residual set collapses onto one parent:

- `<a_65><b_80><c_183>`

The new [dominant_residual_parent_leaf_stats.csv](/Users/fanghaotian/Desktop/src/GenRec/output/local-research-bundles/instruments_grec_hint_research_bundle/analysis/dominant_residual_parent_leaf_stats.csv) shows that the problem is even narrower than "one bad branch".

The dominant failure leaf is:

- `<d_100>`
  - `total_case_count = 6`
  - `unsolved_case_count = 6`
  - unsolved rate within leaf = `1.0`

Other residual leaves are much weaker:

- `<d_229>`: `1 / 4`
- `<d_239>`: `1 / 4`
- `<d_1>`: `1 / 7`
- `<d_137>`: `1 / 7`
- `<d_88>`: `1 / 9`

So the pathology is:

- one parent prefix
- dominated by one especially catastrophic leaf

The title export still shows that this is a very tight `Walker & Williams` guitar-strap family, and the new [dominant_residual_parent_title_terms.csv](/Users/fanghaotian/Desktop/src/GenRec/output/local-research-bundles/instruments_grec_hint_research_bundle/analysis/dominant_residual_parent_title_terms.csv) reinforces that the residual leaves differ mostly by fine-grained variant descriptors:

- `red`
- `natural`
- `bullnose`
- `cabernet`
- `cognac`
- `suede`

Interpretation:

- this is not generic long-tail noise
- it is a near-duplicate family where the final distinction has collapsed into leaf-level product variants
- the residual set should therefore be treated as leaf-disambiguation pathology, not broad recommendation weakness

### Important refinement

The new [dominant_residual_parent_leaf_distinctiveness.csv](/Users/fanghaotian/Desktop/src/GenRec/output/local-research-bundles/instruments_grec_hint_research_bundle/analysis/dominant_residual_parent_leaf_distinctiveness.csv) adds an important correction to that story:

- the catastrophic leaf `<d_100>` is **not** the most lexically generic item under the parent
- in fact it has:
  - `unique_term_count = 3`
  - `mean_jaccard_to_siblings = 0.0`
  - unique terms such as `bullnose`, `gloss`, `semi`

By contrast, several solved siblings are at least as similar or more similar to other leaves in the same family.

It is also **not** simply the rarest leaf in the branch:

- `<d_100>` has `global_count_d4 = 360`
- but several solved siblings are lower-frequency:
  - `<d_59>`: `240`
  - `<d_88>`: `253`
  - `<d_192>`: `211`

So neither of these simple stories is sufficient on its own:

- "the model failed because the title is too generic"
- "the model failed because the leaf is too rare"

Interpretation:

- the failure is not well explained by title-token indistinguishability alone
- the failure is not well explained by global rarity alone
- the SID tree has grouped this family into a very tight branch, but the final catastrophic leaf is not simply the least descriptive title
- this points back to a representation/indexing issue:
  - the branch may be semantically over-collapsed in item space
  - the history signal may not provide enough information to isolate one specific variant even when the title itself is distinctive

So the strongest reading is now:

- **family-level collapse creates the branch**
- **one specific leaf becomes the dominant failure mode inside that branch**
- but the leaf itself is not just a textually generic duplicate

## Similar Pathologies Elsewhere

The new [parent_leaf_pathology_summary.csv](/Users/fanghaotian/Desktop/src/GenRec/output/local-research-bundles/instruments_grec_hint_research_bundle/analysis/parent_leaf_pathology_summary.csv) and [repeated_pathology_candidates.csv](/Users/fanghaotian/Desktop/src/GenRec/output/local-research-bundles/instruments_grec_hint_research_bundle/analysis/repeated_pathology_candidates.csv) show that there are other parents with structurally similar warning signs, but none of them reach the same residual severity.

Examples near the top of the candidate list:

- `<a_253><b_240><c_49>`:
  - `2` leaves
  - `15` hint3+ cases concentrated on one leaf
  - very high within-parent title similarity
  - but **no final unsolved cases**

- `<a_194><b_58><c_78>` and `<a_194><b_58><c_248>`:
  - cable / microphone accessory families
  - high within-parent lexical overlap
  - one leaf absorbs most late-stage difficulty
  - but still no final residual collapse

This matters because it means:

- there is a broader class of "potential pathology" branches
- but `<a_65><b_80><c_183>` is uniquely bad in actually producing unsolved samples

So not every near-duplicate family is catastrophic.
The truly dangerous pattern seems to be:

1. a tightly collapsed family branch
2. late-stage concentration onto one leaf
3. that leaf also appearing often enough in train/eval trajectories to repeatedly fail

## Practical Takeaways

1. `hint depth` is not a single scalar notion of "difficulty".
   It mixes at least two mechanisms:
   - task/input ambiguity
   - local branch competition in the SID tree

2. `hisTitle2sid` is genuinely harder than `sid`.
   This remains true after controlling for local tree structure, so the gap is not just codebook exposure bias.

3. Once a sample has already needed `hint_2`, the story becomes highly structural.
   Large sibling count, low local child share, and weak parent dominance are the main signals.

4. The residual set is small enough to inspect as branch pathology.
   It should not be treated as ordinary hard negatives.

5. `title_desc2sid` looks bimodal rather than simply easy.
   It has the lowest `base -> need_hint` rate (`28.19%`), but conditional on already missing at base it has the highest continuation rates:
   - `hint_1 -> need_hint_2+ = 81.42%`
   - `hint_2 -> need_hint_3+ = 3.39%`
   - `hint_3 -> unsolved = 4.44%`

   A reasonable reading is:
   - when title/description is informative enough, the task is easy immediately
   - when it is ambiguous, the remaining cases are unusually concentrated in hard local branches

## What This Changes About the Story

After the second pass, the picture is now sharper:

1. there is a shared hard-tree backbone
   - both `sid` and `hisTitle2sid` repeatedly fail on many of the same prefixes

2. there is also task-specific amplification
   - some local branches are disproportionately worse for `hisTitle2sid`

3. the final residual set is narrower than "one bad branch"
   - it is dominated by one parent prefix and, inside it, effectively one especially bad leaf

4. this suggests a two-layer intervention strategy
   - a global / task-level hint-depth prior
   - a branch- and possibly leaf-aware escalation policy for a tiny pathological subset

## What I Would Change Next

Priority order:

1. patch [genrec-hint-cascade-analysis-3.ipynb](/Users/fanghaotian/Desktop/src/GenRec/scripts/hint_research/genrec-hint-cascade-analysis-3.ipynb) or [genrec-hint-cascade-analysis-2.ipynb](/Users/fanghaotian/Desktop/src/GenRec/scripts/hint_research/genrec-hint-cascade-analysis-2.ipynb) to surface the new transition-aligned outputs directly
2. inspect why `<d_100>` is uniquely catastrophic inside `<a_65><b_80><c_183>`, instead of treating all residual leaves as equivalent
3. look for repeated copies of the same pathology elsewhere:
   - large near-duplicate family
   - one dominant unsolved leaf
4. if this is turned into training policy, separate:
   - task-level hint prior
   - branch-level adaptive hint escalation

## Status

This pass is exploratory but reproducible.

The current bundle-level analysis is now scripted, and the new outputs can be regenerated with:

```bash
/Users/fanghaotian/Desktop/src/GenRec/.venv/bin/python \
  /Users/fanghaotian/Desktop/src/GenRec/scripts/hint_research/explore_local_hint_bundle.py
```
