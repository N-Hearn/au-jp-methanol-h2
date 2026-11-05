"""Per-site renewable and storage allocation bar chart.

Shows for each port: wind and solar capacity (labelled with PEM GW and
n_synth) on a large top panel, with battery energy and H2 storage below. Accepts a SupplyChainResult directly
or runs the default SupplyChain (single-site) if called standalone.

Usage:
    python -m scripts.plots.site_allocation
    python -m scripts.plots.site_allocation --optimise
    python -m scripts.plots.site_allocation --outdir outputs
    python -m scripts.plots.site_allocation --replot outputs/optimised_build
"""

import argparse
from pathlib import Path

from scripts.common.cli import init_plotting

init_plotting()

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from meoh_pathway.parameters import PORT_SITES
from meoh_pathway.supply_chain import SupplyChain, SupplyChainResult
from scripts.common.plot_common import (BATTERY_COLOUR, H2_STORE_COLOUR, INK, SOLAR_COLOUR,
                                  THESIS_FONT_PT, THESIS_TITLE_PT, THESIS_WIDTH_IN, WIND_COLOUR,
                                  port_label as _port_label, save_fig, tint)
MUTED = "#888"  # "Not used" notes


def plot_site_allocation(r: SupplyChainResult, outpath: Path = None) -> None:
    # Show every port; sites the optimiser has left at zero renewables are
    # kept in place with a "Not used" annotation so the mix stays visible.
    ports = list(PORT_SITES)
    used = {
        p
        for p in ports
        if r.site_wind_mw.get(p, 0.0) + r.site_solar_mw.get(p, 0.0) > 0.0
    }
    if not used:
        print("No active sites found in result.")
        return

    labels = [_port_label(p) for p in ports]
    wind_gw = np.array([r.site_wind_mw.get(p, 0.0) / 1e3 for p in ports])
    solar_gw = np.array([r.site_solar_mw.get(p, 0.0) / 1e3 for p in ports])
    h2_mkg = np.array([r.site_h2_storage_kg.get(p, 0.0) / 1e6 for p in ports])
    batt_gwh = np.array([r.site_battery_mwh.get(p, 0.0) / 1e3 for p in ports])
    pem_gw = [(r.site_pem_large_modules.get(p, 0)*r.pem_large_module_mw
               + r.site_pem_small_modules.get(p, 0)*r.pem_small_module_mw)/1e3 for p in ports]
    n_synth = [r.site_n_synth.get(p, 0) for p in ports]

    x = np.arange(len(ports))
    w = 0.38
    gap = 0.05  # between a site's wind and solar bars, so their outlines do not touch

    # The thesis page width and type sizes, with the cost pies' outlined fills.
    fig = plt.figure(figsize=(THESIS_WIDTH_IN, 6.4), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.7, 1])
    ax_re = fig.add_subplot(gs[0, :])
    ax_batt = fig.add_subplot(gs[1, 0])
    ax_h2 = fig.add_subplot(gs[1, 1])
    fig.suptitle("Per-site allocation", fontsize=THESIS_TITLE_PT, fontweight="bold")

    def _bar(ax, x, heights, width, colour, **kwargs):
        # No outline on an empty bar: it would draw a coloured line along the axis.
        return ax.bar(x, heights, width, color=tint(colour), edgecolor=colour,
                      linewidth=[1.5 if h > 0 else 0 for h in heights], **kwargs)

    def _style(ax, ylabel, title, rotation=30):
        ax.set_title(title, loc="left", fontsize=THESIS_FONT_PT)
        ax.set_ylabel(ylabel, fontsize=THESIS_FONT_PT)
        ax.set_xticks(x)
        ax.set_xticklabels(
            labels, rotation=rotation, ha="right", rotation_mode="anchor"
        )
        ax.tick_params(labelsize=THESIS_FONT_PT)
        ax.grid(axis="x", visible=False)
        ax.spines[["top", "right"]].set_visible(False)
        ax.margins(y=0.18)

    def _mark_off(ax):
        yspan = ax.get_ylim()[1] - ax.get_ylim()[0]
        for xi, p in enumerate(ports):
            if p in used:
                continue
            ax.text(xi, 0.02 * yspan, "Not used", rotation=90, ha="center",
                    va="bottom", fontsize=THESIS_FONT_PT, color=MUTED)

    def _value_labels(ax, bars, vals, fmt):
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.annotate(fmt.format(val), (bar.get_x() + bar.get_width()/2, val),
                            xytext=(0, 2), textcoords="offset points",
                            ha="center", va="bottom", fontsize=THESIS_FONT_PT, color=INK)

    # --- top: plants and renewable capacity ---
    ax = ax_re
    wind_bars = _bar(ax, x - w/2, wind_gw, w - gap, WIND_COLOUR)
    solar_bars = _bar(ax, x + w/2, solar_gw, w - gap, SOLAR_COLOUR)
    _style(ax, "GW", "Plants and renewable capacity")
    # Own swatches: the bars' would copy the first port's, unoutlined if it is empty.
    ax.legend([Patch(facecolor=tint(c), edgecolor=c, linewidth=1.5)
               for c in (WIND_COLOUR, SOLAR_COLOUR)], ["Wind", "Solar"],
              fontsize=THESIS_FONT_PT, frameon=False, loc="upper left")
    _value_labels(ax, wind_bars, wind_gw, "{:.1f}")
    _value_labels(ax, solar_bars, solar_gw, "{:.1f}")
    ax.autoscale_view()
    # Headroom for the three-line plant boxes above the tallest bar.
    ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
    yspan = ax.get_ylim()[1]
    for xi, p in enumerate(ports):
        if p not in used:
            continue
        # Stacked short lines, so neighbouring sites' boxes fit side by side.
        lbl = f"PEM\n{pem_gw[xi]:,.1f} GW"
        if n_synth[xi] > 0:
            lbl += f"\nsynth x{n_synth[xi]}"
        ax.text(xi, max(wind_gw[xi], solar_gw[xi]) + 0.07*yspan, lbl, ha="center",
                va="bottom", fontsize=THESIS_FONT_PT, color=INK, multialignment="center",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#ccc", lw=0.6))
    _mark_off(ax)

    # --- bottom-left: battery ---
    ax = ax_batt
    bars = _bar(ax, x, batt_gwh, 0.6, BATTERY_COLOUR)
    _style(ax, "GWh", "Battery energy capacity", rotation=45)
    _value_labels(ax, bars, batt_gwh, "{:.1f}")
    if not any(batt_gwh > 0):
        ax.text(0.5, 0.5, "Battery not used", transform=ax.transAxes, ha="center",
                va="center", fontsize=THESIS_FONT_PT, color=MUTED)
    _mark_off(ax)

    # --- bottom-right: H2 storage ---
    ax = ax_h2
    bars = _bar(ax, x, h2_mkg, 0.6, H2_STORE_COLOUR)
    _style(ax, "M kg", "H2 storage capacity", rotation=45)
    _value_labels(ax, bars, h2_mkg, "{:.0f}")
    if not any(h2_mkg > 0):
        ax.text(0.5, 0.5, "H2 storage not used", transform=ax.transAxes, ha="center",
                va="center", fontsize=THESIS_FONT_PT, color=MUTED)
    _mark_off(ax)

    if outpath:
        outpath.parent.mkdir(parents=True, exist_ok=True)
        save_fig(fig, outpath, tight=False)
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Per-site allocation plot.")
    parser.add_argument(
        "--optimise", action="store_true", help="Run the DE optimiser first (slow)."
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Save figure to this directory instead of displaying.",
    )
    parser.add_argument(
        "--replot",
        nargs="+",
        default=None,
        help="Redraw site_allocation.png from saved results.json in these directories.",
    )
    args = parser.parse_args()

    if args.replot:
        from scripts.common.outputs import read_result

        for d in map(Path, args.replot):
            plot_site_allocation(read_result(d), outpath=d / "site_allocation.png")
            print(f"Replotted {d / 'site_allocation.png'}")
        return

    if args.optimise:
        from meoh_pathway.optimisation import optimise

        print("Running optimiser (this may take several minutes)...")
        opt = optimise()
        r = opt.full_result
    else:
        r = SupplyChain().run()

    outpath = Path(args.outdir) / "site_allocation.png" if args.outdir else None
    plot_site_allocation(r, outpath=outpath)


if __name__ == "__main__":
    main()
