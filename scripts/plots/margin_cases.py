"""Cost of design flexibility: the cost-optimum against builds with a delivery margin.

python -m scripts.plots.margin_cases

Reads outputs/optimised_build/results.json (0% margin, the cost-optimum) and
outputs/optimised_build/margin_cases/margin_<pct>pct/results.json for every
margin that has been run (scripts.builds.optimisation --delivery-margin). The
technical Monte Carlo of each build -- outputs/monte_carlo/technical for the
cost-optimum, outputs/monte_carlo/margin_<pct>pct/technical for a margin case
(scripts.uncertainty.monte_carlo --margin <pct>) -- shows what the headroom
buys: the share of the target the fixed build still delivers when its
technical inputs move, each draw's verified shortfall taken off the target.

Writes optimised_build/design_analysis/margin_cost.png (delivered cost and
footprint of each build, Fig 20), margin_cases/margin_reliability.png
(technical Monte Carlo delivery of each build, Fig 16) and
margin_cases/margin_summary.json.
"""

import json
import re

import numpy as np

from scripts.common.cli import init_plotting
from scripts.common.outputs import MARGIN_DIR, MC_DIR, OPT_DIR, design_dir, margin_mc_dir

init_plotting()

import matplotlib.pyplot as plt  # noqa: E402
from scripts.common.plot_common import (INK, THESIS_FONT_PT, THESIS_TITLE_PT,  # noqa: E402
                                  THESIS_WIDTH_IN, margin_colour, port_label, save_fig, tint)


def _builds():
    builds = {0.0: OPT_DIR}
    for path in sorted(MARGIN_DIR.glob("margin_*pct")):
        match = re.fullmatch(r"margin_([\d.]+)pct", path.name)
        if match and (path / "results.json").exists():
            builds[float(match.group(1))] = path
    out = {}
    for pct, path in sorted(builds.items()):
        root = MC_DIR if pct == 0 else margin_mc_dir(pct)
        mc = root / "technical" / "optimised_build" / "samples.json"
        out[pct] = dict(result=json.loads((path / "results.json").read_text()),
                        mc=mc if mc.exists() else None)
    return out


def _delivered_share(samples_path, target_tpa):
    """Share of the target each technical draw delivers; a verified draw delivers all of it."""
    draws = json.loads(samples_path.read_text())
    shortfall = np.array([d.get("hydrogen_shortfall_kg", np.nan) for d in draws], dtype=float)
    if np.isnan(shortfall).any():
        raise ValueError(f"{samples_path}: unquantified shortfalls; the delivered share is undefined")
    return 1 - shortfall / (target_tpa * 1000)


def main():
    builds = _builds()
    summary = {}
    for pct, b in builds.items():
        r = b["result"]
        row = dict(delivered_cost_aud_per_kg_h2=r["delivered_cost_aud_per_kg_h2"],
                   delivered_ei_kg_co2_per_kg_h2=r["delivered_ei_kg_co2_per_kg_h2"],
                   sites={p: s for p, s in r["site_h2_frac"].items() if s > 0},
                   n_pem=r["n_pem"], n_synth=r["n_synth"])
        if b["mc"]:
            share = _delivered_share(b["mc"], r["target_h2_tpa"])
            row |= dict(technical_mc_draws=len(share),
                        technical_full_delivery_share=float(np.mean(share >= 1 - 1e-9)),
                        technical_delivered_p10=float(np.percentile(share, 10)),
                        technical_delivered_p50=float(np.percentile(share, 50)))
        summary[f"{pct:g}%"] = row
    MARGIN_DIR.mkdir(parents=True, exist_ok=True)
    (MARGIN_DIR / "margin_summary.json").write_text(json.dumps(summary, indent=2))

    _plot_margin_cost(summary)
    _plot_margin_reliability(summary)
    for pct, row in summary.items():
        sites = ", ".join(f"{port_label(p)} {s:.0%}" for p, s in sorted(row["sites"].items(), key=lambda kv: -kv[1]))
        reach = (f"; full target in {row['technical_full_delivery_share']:.0%} of technical draws"
                 if "technical_full_delivery_share" in row else "")
        print(f"{pct:>4}: {row['delivered_cost_aud_per_kg_h2']:.3f} AUD/kg{reach}; {sites}")


def _rows(summary):
    rows = []
    for key, row in summary.items():
        pct = float(key.rstrip("%"))
        rows.append((pct, row))
    return sorted(rows)


def _tick(pct, row):
    hubs = len(row["sites"])
    return f"{pct:g}% margin\n{hubs} hubs\n{row['n_pem'] / 10:.1f} GW PEM\n{row['n_synth']} synthesis trains"


def _plot_margin_cost(summary):
    """Delivered cost of each margin build, with its footprint on the tick label."""
    rows = _rows(summary)
    fs = THESIS_FONT_PT
    fig, ax = plt.subplots(figsize=(THESIS_WIDTH_IN, 3.4), layout="constrained")
    x = np.arange(len(rows))
    cost = np.array([row["delivered_cost_aud_per_kg_h2"] for _, row in rows])
    for i, (pct, _) in enumerate(rows):
        colour = margin_colour(pct)
        ax.bar(i, cost[i], 0.55, color=tint(colour), edgecolor=colour, linewidth=1.5, zorder=2)
        ax.text(i, cost[i] + 0.25, f"{cost[i]:.2f}", ha="center", va="bottom",
                fontsize=fs, fontweight="bold", color=INK)
    ax.set_xticks(x, [_tick(pct, row) for pct, row in rows], fontsize=fs)
    ax.set_ylabel("AUD / kg H2", fontsize=fs)
    ax.set_ylim(0, cost.max() * 1.12)
    ax.tick_params(axis="y", labelsize=fs)
    ax.grid(axis="x", visible=False)
    ax.set_title("Delivered Cost of Design Margin Builds", fontsize=THESIS_TITLE_PT)
    design_dir(OPT_DIR).mkdir(parents=True, exist_ok=True)
    save_fig(fig, design_dir(OPT_DIR) / "margin_cost.png", tight=False)


def _plot_margin_reliability(summary):
    """(a) share of technical draws meeting the full target, (b) median and P10 delivery."""
    rows = [(pct, row) for pct, row in _rows(summary) if "technical_full_delivery_share" in row]
    if not rows:
        print("No technical Monte Carlo runs found; margin_reliability.png not drawn")
        return
    fs = THESIS_FONT_PT
    fig, (a, b) = plt.subplots(1, 2, figsize=(THESIS_WIDTH_IN, 2.9), layout="constrained")
    x = np.arange(len(rows))
    ticks = [f"{pct:g}%" for pct, _ in rows]
    full = np.array([row["technical_full_delivery_share"] for _, row in rows]) * 100
    p50 = np.array([row["technical_delivered_p50"] for _, row in rows]) * 100
    p10 = np.array([row["technical_delivered_p10"] for _, row in rows]) * 100
    for i, (pct, _) in enumerate(rows):
        colour = margin_colour(pct)
        a.bar(i, full[i], 0.55, color=tint(colour), edgecolor=colour, linewidth=1.5, zorder=2)
        a.text(i, full[i] + 1.5, f"{full[i]:.0f}%", ha="center", va="bottom", fontsize=fs,
               fontweight="bold", color=INK)
        b.plot([i, i], [p10[i], p50[i]], color="#9a9992", linewidth=1.5, zorder=1)
    colours = [margin_colour(pct) for pct, _ in rows]
    b.scatter(x, p50, s=28, color=colours, zorder=3)
    b.scatter(x, p10, s=28, facecolor="white", edgecolor=colours, linewidth=1.5, zorder=3)
    # Legend keys in ink: the marker colour follows the margin, the fill names the statistic.
    b.scatter([], [], s=28, color=INK, label="Median")
    b.scatter([], [], s=28, facecolor="white", edgecolor=INK, linewidth=1.5, label="P10")
    a.set_ylim(0, max(60, full.max() + 12))
    a.set_ylabel("Draws delivering the full target [%]", fontsize=fs)
    a.set_title("(a) Draws meeting the full target", loc="left", fontsize=fs)
    b.set_ylim(np.floor(p10.min()) - 1.5, 100.5)
    b.set_ylabel("Share of target delivered [%]", fontsize=fs)
    b.set_title("(b) Delivery across all draws", loc="left", fontsize=fs)
    b.legend(frameon=False, fontsize=fs, loc="lower right")
    for ax in (a, b):
        ax.set_xticks(x, ticks, fontsize=fs)
        ax.set_xlabel("Design margin", fontsize=fs)
        ax.tick_params(axis="y", labelsize=fs)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Technical Reliability of Design Margin Builds", fontsize=THESIS_TITLE_PT,
                 fontweight="bold")
    save_fig(fig, MARGIN_DIR / "margin_reliability.png", tight=False)


if __name__ == "__main__":
    main()
