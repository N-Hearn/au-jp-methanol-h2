"""Pathway comparison bars and delivered-cost pies shared by the e-MeOH builds."""

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from base_case.run_base_case import run_blue
from scripts.common.plot_common import (BASE_CASE, FLOW_COLORS, INK, REFERENCE_CASE, THESIS_FONT_PT,
                                        THESIS_TITLE_PT, THESIS_WIDTH_IN, plot_cost_pie,
                                        plot_cost_pie_slide, save_fig, tint)

MEOH_LABEL = "e-MeOH (hourly feasible)"


def _require_feasible(meoh):
    if not meoh.hourly_feasible:
        raise ValueError("Cannot plot an infeasible result as delivered")


def plot_pathway_comparison(grey, meoh, outdir, builds=None):
    """Grey reference case, blue base case and the e-MeOH pathway side by side.

    builds, a list of (label, result, colour), replaces the single meoh bar --
    the optimised builds compare the cost-optimum with the proposed build.
    """
    blue = run_blue()
    pathways = {
        f"{REFERENCE_CASE[0]}\n{REFERENCE_CASE[1]}": (grey, REFERENCE_CASE[2]),
        f"{BASE_CASE[0]}\n{BASE_CASE[1]}": (blue, BASE_CASE[2]),
    }
    for label, result, colour in builds or [(MEOH_LABEL, meoh, FLOW_COLORS["h2"])]:
        _require_feasible(result)
        pathways[label] = (result, colour)
    colours = [colour for _, colour in pathways.values()]
    x = range(len(pathways))
    # The thesis page width and type sizes, with the cost pies' outlined fills.
    # Up to five bars a panel leave no room for their names at THESIS_FONT_PT,
    # so one legend below both panels names them instead of tick labels.
    # Up to four pathways on one legend row; five (the margin cases) take two.
    ncol = len(pathways) if len(pathways) <= 4 else 3
    rows = -(-len(pathways) // ncol)
    fig, axes = plt.subplots(1, 2, figsize=(THESIS_WIDTH_IN, 2.9 + 0.35 * rows),
                             layout="constrained")
    fig.suptitle("Hydrogen Pathway Comparison", fontsize=THESIS_TITLE_PT, fontweight="bold")
    metrics = [
        ("Delivered cost", "AUD / kg H2", "delivered_cost_aud_per_kg_h2"),
        ("Emissions intensity", "kg CO2e / kg H2", "delivered_ei_kg_co2_per_kg_h2"),
    ]
    for ax, (metric, ylabel, attr) in zip(axes, metrics):
        values = [getattr(r, attr) for r, _ in pathways.values()]
        if attr == "delivered_ei_kg_co2_per_kg_h2":
            # Gross is drawn hatched and lighter behind the solid net bar. Grey and
            # blue emit their CO2, so net equals gross; the e-MeOH builds' net is zero.
            net = [values[0], values[1]] + [0.0] * (len(values) - 2)
            gross = ax.bar(x, values, width=0.5, color=[tint(c, 0.85) for c in colours],
                           edgecolor=colours, linewidth=1.5, hatch="////")
            # Where they differ both values sit above the gross bar: a net label at
            # the foot of a zero-height bar would overprint the hatching.
            labels = [f"Net {n:.2f}\nGross {g:.2f}" if g != n else f"{g:.2f}"
                      for g, n in zip(values, net)]
            bars = ax.bar(x, net, width=0.5, color=[tint(c) for c in colours],
                          edgecolor=colours, linewidth=1.5)
            ax.legend([Patch(facecolor=tint(INK, 0.85), edgecolor=INK, hatch="////"),
                       Patch(facecolor=tint(INK), edgecolor=INK)], ["Gross", "Net"],
                      loc="upper right", frameon=False, fontsize=THESIS_FONT_PT, handlelength=1.5)
        else:
            gross = bars = ax.bar(x, values, width=0.5, color=[tint(c) for c in colours],
                                  edgecolor=colours, linewidth=1.5)
            labels = [f"{v:.2f}" for v in values]
        ax.bar_label(gross, labels=labels, padding=2, fontweight="bold",
                     fontsize=THESIS_FONT_PT, color=INK)
        ax.set_title(metric, fontsize=THESIS_FONT_PT)
        ax.set_ylabel(ylabel, fontsize=THESIS_FONT_PT)
        ax.set_xticks(x, [])
        ax.set_ylim(0, max(values) * 1.15)
        ax.tick_params(labelsize=THESIS_FONT_PT)
        ax.grid(axis="x", visible=False)
    fig.legend(bars, list(pathways), loc="outside lower center", ncol=ncol,
               frameon=False, fontsize=THESIS_FONT_PT, handlelength=1.5, columnspacing=1.2)
    save_fig(fig, outdir / "pathway_comparison.png", tight=False)


def plot_cost_breakdown(meoh, outdir, slide=False, thesis=False):
    """Detailed pie (cost_breakdown.png), or with slide=True the slide pie
    (cost_breakdown_slide.png); thesis=True with it draws the slide pie's thesis
    copy, cost_breakdown_thesis.png."""
    costs = {k: v for k, v in meoh.cost_breakdown_aud_per_kg_h2.items() if k != "Total"}
    if slide:
        name = "cost_breakdown_thesis.png" if thesis else "cost_breakdown_slide.png"
        plot_cost_pie_slide(costs, outdir / name, thesis=thesis)
        return
    _require_feasible(meoh)
    plot_cost_pie(costs, f"{MEOH_LABEL}: delivered cost breakdown", outdir / "cost_breakdown.png")
