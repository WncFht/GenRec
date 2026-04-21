#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from build_instruments_report_figures import ASSET_DIR, SPECS


DOC_PATH = Path(__file__).resolve().parents[1] / "instruments_style_options.md"
PREVIEW_PATH = ASSET_DIR / "style_option_preview.png"
SFT_REFERENCE_COLOR = "#6b7280"
REPORT_LINK_COLOR = "blue!45!black"
REPORT_URL_COLOR = "blue!55!black"
ACTIVE_PALETTE = "tableau-10"
ACTIVE_CE_PROFILE = "CE-A"

PALETTES = {
    "tableau-10": [
        "#4E79A7",
        "#F28E2B",
        "#E15759",
        "#76B7B2",
        "#59A14F",
        "#EDC948",
        "#B07AA1",
        "#FF9DA7",
        "#9C755F",
        "#BAB0AB",
    ],
    "tol-bright-10": [
        "#4477AA",
        "#EE6677",
        "#228833",
        "#CCBB44",
        "#66CCEE",
        "#AA3377",
        "#BBBBBB",
        "#000000",
        "#EE7733",
        "#0077BB",
    ],
    "paper-10": [
        "#0F4C81",
        "#D55E00",
        "#009E73",
        "#CC79A7",
        "#7A8B99",
        "#E1BE6A",
        "#40B0A6",
        "#A34A28",
        "#3B528B",
        "#5C5C5C",
    ],
}

LINESTYLE_OPTIONS = [
    ("L1", "-", "solid / canonical baseline"),
    ("L2", "--", "dashed / same-family ablation"),
    ("L3", "-.", "dash-dot / second ablation"),
    ("L4", ":", "dotted / light auxiliary"),
    ("L5", "(0, (5, 1.5))", "long-dash / extra separation"),
    ("L6", "(0, (3, 1, 1, 1))", "dash-dot-dot / dense family"),
]

MARKER_OPTIONS = [
    ("M1", "o", "circle"),
    ("M2", "s", "square"),
    ("M3", "^", "triangle up"),
    ("M4", "D", "diamond"),
    ("M5", "P", "filled plus"),
    ("M6", "X", "x-filled"),
    ("M7", "v", "triangle down"),
    ("M8", "<", "triangle left"),
    ("M9", ">", "triangle right"),
    ("M10", "*", "star"),
]

CE_OPTIONS = [
    {
        "id": "CE-A",
        "title": "High contrast",
        "why": "最容易区分；不强调同一家族颜色。",
        "styles": [
            ("fixed taskfix", "#F28E2B", "o", "-"),
            ("hintce", "#4E79A7", "s", "--"),
            ("hintce-2", "#59A14F", "^", "-."),
            ("hintce-3", "#E15759", "D", ":"),
        ],
    },
    {
        "id": "CE-B",
        "title": "Warm family but separated",
        "why": "保留 fixed/CE 的暖色亲缘，同时把亮度和线型拉开。",
        "styles": [
            ("fixed taskfix", "#C46A2E", "o", "-"),
            ("hintce", "#8C564B", "s", "--"),
            ("hintce-2", "#E09F3E", "^", "-."),
            ("hintce-3", "#9C2F2F", "D", ":"),
        ],
    },
    {
        "id": "CE-C",
        "title": "Color-sparing",
        "why": "少占颜色槽位，主要靠线型和 marker 区分。",
        "styles": [
            ("fixed taskfix", "#E15759", "o", "-"),
            ("hintce", "#E15759", "s", "--"),
            ("hintce-2", "#E15759", "^", "-."),
            ("hintce-3", "#E15759", "D", ":"),
        ],
    },
]


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _build_inventory_tables() -> tuple[str, str]:
    current_rows = []
    for spec in SPECS:
        current_rows.append(
            [
                spec.key,
                spec.label,
                spec.color,
                spec.marker,
                spec.linestyle,
            ]
        )

    support_rows = [
        ["SFT reference line", SFT_REFERENCE_COLOR],
        ["Report link color", REPORT_LINK_COLOR],
        ["Report URL color", REPORT_URL_COLOR],
    ]

    return (
        _markdown_table(["Key", "Label", "Color", "Marker", "Line"], current_rows),
        _markdown_table(["Usage", "Color"], support_rows),
    )


def _build_palette_tables() -> str:
    parts = []
    for palette_name, colors in PALETTES.items():
        rows = [[f"C{i + 1}", color] for i, color in enumerate(colors)]
        parts.append(f"### `{palette_name}`\n\n{_markdown_table(['Slot', 'Hex'], rows)}")
    return "\n\n".join(parts)


def _build_style_tables() -> tuple[str, str]:
    line_rows = [[style_id, code, note] for style_id, code, note in LINESTYLE_OPTIONS]
    marker_rows = [[marker_id, code, note] for marker_id, code, note in MARKER_OPTIONS]
    return (
        _markdown_table(["ID", "Matplotlib", "Use"], line_rows),
        _markdown_table(["ID", "Marker", "Shape"], marker_rows),
    )


def _build_ce_table() -> str:
    rows = []
    for option in CE_OPTIONS:
        style_text = []
        for label, color, marker, linestyle in option["styles"]:
            style_text.append(f"`{label}` = `{color}` + `{marker}` + `{linestyle}`")
        rows.append([option["id"], option["title"], "<br>".join(style_text), option["why"]])
    return _markdown_table(["Option", "Direction", "Assignment", "Why"], rows)


def write_markdown() -> None:
    current_table, support_table = _build_inventory_tables()
    palette_tables = _build_palette_tables()
    line_table, marker_table = _build_style_tables()
    ce_table = _build_ce_table()

    current_colors = len(SPECS)
    unique_markers = sorted({spec.marker for spec in SPECS})
    unique_linestyles = sorted({spec.linestyle for spec in SPECS})
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    marker_list = ", ".join(f"`{marker}`" for marker in unique_markers)
    linestyle_list = ", ".join(f"`{linestyle}`" for linestyle in unique_linestyles)

    text = f"""# Instruments Plot Style Options

- Generated at: `{timestamp}`
- Preview image: [`assets/instruments-report/style_option_preview.png`](assets/instruments-report/style_option_preview.png)
- Active selection: `{ACTIVE_PALETTE}` + `{ACTIVE_CE_PROFILE}`

## 1. Current inventory

- Current variant colors: `{current_colors}` entries, all unique.
- Current plot markers: {marker_list}
- Current line styles: {linestyle_list}
- Extra non-variant colors: SFT reference gray plus LaTeX link colors.

### Current variant styles

{current_table}

### Current support colors

{support_table}

## 2. What is making CE hard to read now

- `hintce`, `hintce-2`, `hintce-3` are all warm colors, so hue separation is weak.
- Their current colors are `#8c6a43`, `#c77d11`, `#bc3908`.
- The three CE lines are all solid lines, so line shape does not help.
- Current marker size in plotting code is `4`, so `o / s / ^` alone is not enough once curves overlap.
- Future style policy should reserve colors first for families, then use line and marker to separate family-internal ablations.

## 3. Candidate 10-color palettes

{palette_tables}

## 4. Reusable line options

{line_table}

## 5. Reusable marker options

{marker_table}

## 6. CE-specific redesign options

{ce_table}

## 7. Suggested decision rule

- If you want the easiest-to-read paper figures, pick `tableau-10` or `tol-bright-10` plus `CE-A`.
- If you want to preserve “fixed family is warm-colored” semantics, pick `paper-10` plus `CE-B`.
- If you want all CE variants to read as one subfamily and save colors for other sections, pick any 10-color palette plus `CE-C`.

## 8. Quick reply format

You can reply with one short line such as:

- `palette = tableau-10, CE = CE-A`
- `palette = tol-bright-10, CE = CE-B`
- `palette = paper-10, CE = CE-C, markers use M1-M10`
"""
    DOC_PATH.write_text(text)


def _render_palette_row(ax: plt.Axes, title: str, colors: list[str]) -> None:
    ax.set_title(title, fontsize=10, loc="left")
    for idx, color in enumerate(colors):
        ax.add_patch(plt.Rectangle((idx, 0), 0.95, 0.45, color=color))
        ax.text(idx + 0.475, -0.08, f"C{idx + 1}", ha="center", va="top", fontsize=8)
        ax.text(idx + 0.475, 0.52, color, ha="center", va="bottom", fontsize=7)
    ax.set_xlim(0, len(colors))
    ax.set_ylim(-0.25, 0.9)
    ax.axis("off")


def _render_current_row(ax: plt.Axes) -> None:
    ax.set_title("Current variant colors", fontsize=10, loc="left")
    for idx, spec in enumerate(SPECS):
        ax.plot(
            [idx, idx + 0.7],
            [0.5, 0.5],
            color=spec.color,
            marker=spec.marker,
            linestyle=spec.linestyle,
            linewidth=2.2,
            markersize=6,
        )
        ax.text(idx + 0.35, 0.82, spec.key, ha="center", va="bottom", rotation=45, fontsize=7)
        ax.text(idx + 0.35, 0.08, spec.color, ha="center", va="top", fontsize=6)
    ax.set_xlim(-0.2, len(SPECS))
    ax.set_ylim(0, 1.05)
    ax.axis("off")


def _render_style_row(ax: plt.Axes) -> None:
    ax.set_title("Line + marker combos", fontsize=10, loc="left")
    x = [0, 1, 2, 3]
    for idx, (style_id, linestyle, _) in enumerate(LINESTYLE_OPTIONS[:5]):
        marker = MARKER_OPTIONS[idx][1]
        y = [idx] * len(x)
        ax.plot(
            x,
            y,
            color="#12355B",
            marker=marker,
            linestyle=linestyle if not linestyle.startswith("(") else (0, (5, 1.5)) if style_id == "L5" else (0, (3, 1, 1, 1)),
            linewidth=2.2,
            markersize=6,
            label=f"{style_id} + {MARKER_OPTIONS[idx][0]}",
        )
    ax.set_xlim(-0.1, 3.1)
    ax.set_ylim(-0.6, 4.6)
    ax.set_yticks(range(5))
    ax.set_yticklabels([f"{LINESTYLE_OPTIONS[i][0]} + {MARKER_OPTIONS[i][0]}" for i in range(5)], fontsize=8)
    ax.set_xticks([])
    ax.grid(axis="x", alpha=0.15)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _render_ce_row(ax: plt.Axes, option: dict[str, object]) -> None:
    ax.set_title(f"{option['id']}  {option['title']}", fontsize=10, loc="left")
    x = [0, 1, 2, 3]
    for idx, (label, color, marker, linestyle) in enumerate(option["styles"]):
        y = [idx + v for v in [0.05, 0.12, -0.02, 0.08]]
        ax.plot(x, y, color=color, marker=marker, linestyle=linestyle, linewidth=2.2, markersize=6, label=label)
    ax.set_xlim(-0.1, 3.1)
    ax.set_ylim(-0.2, 3.8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(alpha=0.15)
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper left")
    for spine in ax.spines.values():
        spine.set_visible(False)


def write_preview() -> None:
    PREVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(6, 1, height_ratios=[1.4, 1.1, 1.1, 1.1, 1.4, 1.4], hspace=0.85)

    _render_current_row(fig.add_subplot(gs[0]))
    _render_palette_row(fig.add_subplot(gs[1]), "Palette: tableau-10", PALETTES["tableau-10"])
    _render_palette_row(fig.add_subplot(gs[2]), "Palette: tol-bright-10", PALETTES["tol-bright-10"])
    _render_palette_row(fig.add_subplot(gs[3]), "Palette: paper-10", PALETTES["paper-10"])
    _render_style_row(fig.add_subplot(gs[4]))

    ce_gs = gs[5].subgridspec(1, 3, wspace=0.3)
    for idx, option in enumerate(CE_OPTIONS):
        _render_ce_row(fig.add_subplot(ce_gs[0, idx]), option)

    fig.suptitle("Instruments Plot Style Options", fontsize=14, y=0.995)
    fig.savefig(PREVIEW_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    write_markdown()
    write_preview()
    print(f"doc_path={DOC_PATH}")
    print(f"preview_path={PREVIEW_PATH}")


if __name__ == "__main__":
    main()
