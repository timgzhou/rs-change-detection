"""Visualize lp_olmoearth_pastis.csv: one subplot per feature set, test_miou on y,
points colored by head_mode, marker shape by eval_kind (LP vs KNN), shared y-axis.

mAnyUp rows share head_mode="manyup" but differ by HR target (parsed from the checkpoint
name) and projector use; those points are labeled with their HR target so the sweep is legible.

    python visualize_lp_results.py                      # -> lp_olmoearth_pastis.png
    python visualize_lp_results.py --csv other.csv --out other.png
"""
import argparse
import math
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Categorical palette: the dataviz-skill reference theme, fixed slot order (the ordering is the
# CVD-safety mechanism -- max min-adjacent-ΔE -- not cosmetic). Color follows the head_mode
# entity, assigned in order, never cycled. 8 LP/anyup heads fill the 8 slots; manyup is a
# separate FAMILY (a learned upsampler, not a probe head) so it gets a neutral dark ink and is
# distinguished by its HR-target label + marker, keeping the 8-slot categorical set intact.
HEAD_COLORS = {
    "lp_pa2pa_bu":  "#2a78d6",   # blue
    "lp_pa2px":     "#1baf7a",   # aqua
    "lp_pa2px_ens": "#eb6834",   # orange  (new: temporal ensemble of pa2px)
    "anyup":        "#eda100",   # yellow
    "anyup_t1":     "#008300",   # green
    "anyup_t2":     "#4a3aa7",   # violet
    "anyup_t1_ens": "#e34948",   # red
    "anyup_t2_ens": "#e87ba4",   # magenta
}
MANYUP_COLOR = "#3b3a37"         # manyup family: neutral dark ink (its own marker + HR label)
FALLBACK = "#8a8a86"

# eval_kind -> marker (secondary encoding; identity is head_mode-color, eval is SHAPE not color).
EVAL_MARKER = {"lp": "o", "knn": "^"}


def head_color(head: str) -> str:
    if head == "manyup":
        return MANYUP_COLOR
    return HEAD_COLORS.get(head, FALLBACK)


def parse_ps_tile(feature: str):
    """(ps, tile) as ints from a feature name like oe_base_s2_ps4_tile32 -> (4, 32).
    Returns (None, None) if unparseable (those features fall outside the ps x tile grid)."""
    ps = re.search(r"_ps(\d+)_", feature)
    tile = re.search(r"_tile(\d+)", feature)
    return (int(ps.group(1)) if ps else None, int(tile.group(1)) if tile else None)


def manyup_hr_target(ckpt: str) -> str:
    """Parse the HR target config from a checkpoint name like
    manyup_<lr>_to_oe_base_s2_ps1_tile64_ep9.pth -> 'ps1_tile64'."""
    if not isinstance(ckpt, str) or not ckpt:
        return ""
    m = re.search(r"_to_oe_base_s2_(ps\d+_tile\d+)", ckpt)
    return m.group(1) if m else ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="lp_olmoearth_pastis.csv")
    ap.add_argument("--out", default="lp_olmoearth_pastis.png")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    # Old rows predate eval_kind -> treat missing as LP (they were all LP runs).
    df["eval_kind"] = df.get("eval_kind", "lp").fillna("lp")
    if "manyup_ckpt" not in df:
        df["manyup_ckpt"] = ""

    # Only S2 features (drop the dual-modality s2s1 runs -- not part of this comparison).
    df = df[~df["features"].str.contains("s2s1")]
    features = sorted(df["features"].unique())
    # Legend/x order: fixed palette order, then manyup, then any unknown extras -- restricted to
    # what's actually present.
    present = set(df["head_mode"].unique())
    heads = ([h for h in HEAD_COLORS if h in present]
             + (["manyup"] if "manyup" in present else [])
             + sorted(present - set(HEAD_COLORS) - {"manyup"}))

    # Grid keyed by (patch_size, tile_size): rows = ps ascending, cols = tile ascending. Not every
    # (ps,tile) was extracted, so many cells are empty -> blanked. cell_features maps a (ps,tile)
    # to the feature name(s) there (usually one; s2 vs s2s1 can collide at the same ps/tile).
    cell_features = {}
    for feat in features:
        ps, tile = parse_ps_tile(feat)
        if ps is None or tile is None:
            print(f"skipping (no ps/tile): {feat}")
            continue
        cell_features.setdefault((ps, tile), []).append(feat)
    ps_vals = sorted({ps for ps, _ in cell_features})
    tile_vals = sorted({tile for _, tile in cell_features})
    nrows, ncols = len(ps_vals), len(tile_vals)

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols + 1.6, 3.4 * nrows),
                             sharey=True, sharex=True, squeeze=False)

    head_x = {h: i for i, h in enumerate(heads)}   # fixed column per head across ALL panels
    for ri, ps in enumerate(ps_vals):
        for ci, tile in enumerate(tile_vals):
            ax = axes[ri][ci]
            feats_here = cell_features.get((ps, tile))
            if not feats_here:
                ax.set_visible(False)              # (ps,tile) not extracted -> blank cell
                continue
            sub = df[df["features"].isin(feats_here)]
            for h in heads:
                x = head_x[h]                      # numeric x: same column in every panel
                for ek, marker in EVAL_MARKER.items():
                    pts = sub[(sub["head_mode"] == h) & (sub["eval_kind"] == ek)]
                    if pts.empty:
                        continue
                    ax.scatter([x] * len(pts), pts["test_miou"],
                               color=head_color(h), s=56, zorder=3, marker=marker,
                               edgecolor="white", linewidth=0.6)
                    if h == "manyup":
                        for j, (_, r) in enumerate(pts.iterrows()):
                            hr = manyup_hr_target(r["manyup_ckpt"])
                            proj = "" if pd.isna(r.get("manyup_use_proj")) else \
                                ("+proj" if r.get("manyup_use_proj") in (True, "True") else "-proj")
                            ax.annotate(f"{hr}{(' '+proj) if proj else ''}",
                                        (x, r["test_miou"]), fontsize=6, color=MANYUP_COLOR,
                                        xytext=(7, 9 * (j - (len(pts) - 1) / 2)),
                                        textcoords="offset points", va="center", ha="left")
            ax.set_xticks(range(len(heads)))
            ax.set_xticklabels(heads, rotation=45, ha="right", fontsize=6)
            ax.set_xlim(-0.6, len(heads) - 0.4)
            ax.grid(axis="y", color="#e6e6e3", linewidth=0.8, zorder=0)
            ax.set_axisbelow(True)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)

    # y-label + tick VALUES on the first visible cell of each row. sharey=True suppresses tick
    # labels on all but the reference axis, so re-enable them here (each row's leftmost filled
    # cell isn't necessarily column 0) -- every row then shows its own mIoU scale numbers.
    for ri in range(nrows):
        for ci in range(ncols):
            if (ps_vals[ri], tile_vals[ci]) in cell_features:
                axes[ri][ci].set_ylabel("test mIoU", fontsize=8)
                axes[ri][ci].tick_params(labelleft=True)   # show y tick numbers on this cell
                break

    # Two legends: identity (head_mode -> color) and eval_kind (-> marker shape). Identity is
    # never color-alone -- the x-axis also names the head; shape carries the LP/KNN split.
    head_handles = [plt.Line2D([0], [0], marker="o", linestyle="", markersize=7,
                               markerfacecolor=head_color(h), markeredgecolor="white", label=h)
                    for h in heads]
    eval_handles = [plt.Line2D([0], [0], marker=m, linestyle="", markersize=7,
                               markerfacecolor="#52514e", markeredgecolor="white", label=ek.upper())
                    for ek, m in EVAL_MARKER.items() if ek in set(df["eval_kind"])]
    leg1 = fig.legend(handles=head_handles, title="head_mode", loc="center left",
                      bbox_to_anchor=(1.0, 0.62), fontsize=8, frameon=False)
    fig.add_artist(leg1)
    fig.legend(handles=eval_handles, title="eval", loc="center left",
               bbox_to_anchor=(1.0, 0.28), fontsize=8, frameon=False)

    fig.suptitle("LP / KNN head test mIoU  (rows = patch size, cols = tile size)", fontsize=11)
    # Reserve left+top margin for the grid headers, lay out, THEN place headers at final coords.
    fig.tight_layout(rect=(0.03, 0, 0.99, 0.95))
    # Grid headers on the FIGURE so they share one line regardless of which cells are blank:
    #   tile titles -> TOP-aligned along the top edge, centered on each column;
    #   ps titles   -> LEFT-aligned along the left edge, centered on each row.
    for ci, tile in enumerate(tile_vals):
        pos = axes[0][ci].get_position()          # column x-span (grid columns are regular)
        fig.text((pos.x0 + pos.x1) / 2, 0.965, f"tile{tile}", ha="center", va="top",
                 fontsize=11, fontweight="bold")
    for ri, ps in enumerate(ps_vals):
        pos = axes[ri][0].get_position()          # row y-span
        fig.text(0.01, (pos.y0 + pos.y1) / 2, f"ps{ps}", ha="left", va="center",
                 fontsize=11, fontweight="bold", rotation=90)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}  (grid {nrows}ps x {ncols}tile, "
          f"{len(cell_features)} cells filled, {len(heads)} head modes, "
          f"eval kinds: {sorted(set(df['eval_kind']))})")


if __name__ == "__main__":
    main()
