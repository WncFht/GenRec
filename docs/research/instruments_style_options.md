# Instruments Plot Style Options

- Generated at: `2026-04-22T00:54:19+08:00`
- Preview image: [`assets/instruments-report/style_option_preview.png`](assets/instruments-report/style_option_preview.png)

## 1. Current inventory

- Current variant colors: `14` entries, all unique.
- Current plot markers: `D`, `P`, `^`, `o`, `s`
- Current line styles: `-`, `--`, `-.`
- Extra non-variant colors: SFT reference gray plus LaTeX link colors.

### Current variant styles

| Key | Label | Color | Marker | Line |
| --- | --- | --- | --- | --- |
| rule_only | RL rule-only | #1b7f79 | o | - |
| dynamic_sid_only | RL dynamic sid-only | #4e79a7 | s | - |
| dynamic_gather_fix | RL dynamic gather-fix | #12355b | o | - |
| ranking_dynamic | RL ranking dynamic | #7b8fa1 | ^ | -- |
| fixed_taskfix | RL fixed taskfix | #d88c3a | o | - |
| fixed_taskfix_sid_only | RL fixed taskfix sid-only | #e4572e | s | - |
| fixed_old | RL fixed old | #7d5a99 | D | -. |
| max1 | RL dynamic max1 | #0a9396 | P | - |
| hintce_batch_mean | RL fixed CE batch-mean | #8c6a43 | o | - |
| hintce_token_mean | RL fixed CE token-mean | #c77d11 | s | - |
| hintce_coef_005 | RL fixed CE coef=0.005 | #bc3908 | ^ | - |
| single_hint_mixed | RL single-hint mixed | #c1128d | P | - |
| dynamic_dual_task | RL dynamic dual-task | #3d5a80 | o | - |
| fixed_dual_task | RL fixed dual-task | #8d99ae | s | - |

### Current support colors

| Usage | Color |
| --- | --- |
| SFT reference line | #6b7280 |
| Report link color | blue!45!black |
| Report URL color | blue!55!black |

## 2. What is making CE hard to read now

- `hintce`, `hintce-2`, `hintce-3` are all warm colors, so hue separation is weak.
- Their current colors are `#8c6a43`, `#c77d11`, `#bc3908`.
- The three CE lines are all solid lines, so line shape does not help.
- Current marker size in plotting code is `4`, so `o / s / ^` alone is not enough once curves overlap.
- Future style policy should reserve colors first for families, then use line and marker to separate family-internal ablations.

## 3. Candidate 10-color palettes

### `tableau-10`

| Slot | Hex |
| --- | --- |
| C1 | #4E79A7 |
| C2 | #F28E2B |
| C3 | #E15759 |
| C4 | #76B7B2 |
| C5 | #59A14F |
| C6 | #EDC948 |
| C7 | #B07AA1 |
| C8 | #FF9DA7 |
| C9 | #9C755F |
| C10 | #BAB0AB |

### `tol-bright-10`

| Slot | Hex |
| --- | --- |
| C1 | #4477AA |
| C2 | #EE6677 |
| C3 | #228833 |
| C4 | #CCBB44 |
| C5 | #66CCEE |
| C6 | #AA3377 |
| C7 | #BBBBBB |
| C8 | #000000 |
| C9 | #EE7733 |
| C10 | #0077BB |

### `paper-10`

| Slot | Hex |
| --- | --- |
| C1 | #0F4C81 |
| C2 | #D55E00 |
| C3 | #009E73 |
| C4 | #CC79A7 |
| C5 | #7A8B99 |
| C6 | #E1BE6A |
| C7 | #40B0A6 |
| C8 | #A34A28 |
| C9 | #3B528B |
| C10 | #5C5C5C |

## 4. Reusable line options

| ID | Matplotlib | Use |
| --- | --- | --- |
| L1 | - | solid / canonical baseline |
| L2 | -- | dashed / same-family ablation |
| L3 | -. | dash-dot / second ablation |
| L4 | : | dotted / light auxiliary |
| L5 | (0, (5, 1.5)) | long-dash / extra separation |
| L6 | (0, (3, 1, 1, 1)) | dash-dot-dot / dense family |

## 5. Reusable marker options

| ID | Marker | Shape |
| --- | --- | --- |
| M1 | o | circle |
| M2 | s | square |
| M3 | ^ | triangle up |
| M4 | D | diamond |
| M5 | P | filled plus |
| M6 | X | x-filled |
| M7 | v | triangle down |
| M8 | < | triangle left |
| M9 | > | triangle right |
| M10 | * | star |

## 6. CE-specific redesign options

| Option | Direction | Assignment | Why |
| --- | --- | --- | --- |
| CE-A | High contrast | `fixed taskfix` = `#F28E2B` + `o` + `-`<br>`hintce` = `#4E79A7` + `s` + `--`<br>`hintce-2` = `#59A14F` + `^` + `-.`<br>`hintce-3` = `#E15759` + `D` + `:` | 最容易区分；不强调同一家族颜色。 |
| CE-B | Warm family but separated | `fixed taskfix` = `#C46A2E` + `o` + `-`<br>`hintce` = `#8C564B` + `s` + `--`<br>`hintce-2` = `#E09F3E` + `^` + `-.`<br>`hintce-3` = `#9C2F2F` + `D` + `:` | 保留 fixed/CE 的暖色亲缘，同时把亮度和线型拉开。 |
| CE-C | Color-sparing | `fixed taskfix` = `#E15759` + `o` + `-`<br>`hintce` = `#E15759` + `s` + `--`<br>`hintce-2` = `#E15759` + `^` + `-.`<br>`hintce-3` = `#E15759` + `D` + `:` | 少占颜色槽位，主要靠线型和 marker 区分。 |

## 7. Suggested decision rule

- If you want the easiest-to-read paper figures, pick `tableau-10` or `tol-bright-10` plus `CE-A`.
- If you want to preserve “fixed family is warm-colored” semantics, pick `paper-10` plus `CE-B`.
- If you want all CE variants to read as one subfamily and save colors for other sections, pick any 10-color palette plus `CE-C`.

## 8. Quick reply format

You can reply with one short line such as:

- `palette = tableau-10, CE = CE-A`
- `palette = tol-bright-10, CE = CE-B`
- `palette = paper-10, CE = CE-C, markers use M1-M10`
