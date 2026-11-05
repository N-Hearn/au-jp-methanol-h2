"""Cracker heat supply options (Fig 18): delivered cost and EI of each configuration.

python -m scripts.design_analysis.heat_configs                     # the optimised build
python -m scripts.design_analysis.heat_configs --case typical      # the Typical build
python -m scripts.design_analysis.heat_configs --replot            # redraw from saved JSON

Each option changes only the cracker heat and power inputs on the build's
verified reference: allocation weights and synthesis trains are retained,
process fleets resize, and the methanol- and H2-fired options scale the
renewable/storage footprint until the build stays hourly feasible. Every
resized build must pass a frozen-equipment replay.

Writes <build>/design_analysis/heat_config_comparison.{json,png},
heat_config_metadata.json and, per option, heat_builds/<option>/ (results,
flow diagram and site allocation of the resized build).
"""

import json
from dataclasses import asdict
from pathlib import Path

from scripts.common.cli import build_parser, init_plotting

init_plotting()

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch, Polygon, Rectangle  # noqa: E402
import numpy as np  # noqa: E402

from meoh_pathway.monte_carlo import optimum_deployment  # noqa: E402
from meoh_pathway.optimisation import model_fingerprint  # noqa: E402
from meoh_pathway.sensitivity import HeatConfigResult, ParamSet, run_heat_configs  # noqa: E402
from scripts.common.outputs import CASE_DIRS, OPT_DIR, design_dir  # noqa: E402
from scripts.common.plot_common import (FLOW_COLORS, INK, NEUTRAL_LIGHT, THESIS_FONT_PT,  # noqa: E402
                                        THESIS_WIDTH_IN, save_fig, tint)

MODE_LABELS = {"typical": "Typical build (hourly)", "optimised": "saved candidate (hourly)"}


def load_build(optimum_path, allow_stale=False):
    """(Typical ParamSet, deployment) of a saved optimum.json, decoded by the optimiser."""
    return optimum_deployment(Path(optimum_path), ParamSet.base(), allow_stale=allow_stale)


# Heat configuration figure categories, shared by its cost and EI panels so a
# colour means the same thing in both: (legend label, colour, cost_components
# key, HeatConfigResult EI attribute), None where a panel has no such part.
# Colours follow the flow diagrams (methanol blue, H2 green, electricity gold,
# CO2 orange, vented CO2 red, cracker teal); the rest of the chain is neutral.
HEAT_CATEGORIES = (
    ("Methanol burned for heat", FLOW_COLORS["meoh"], "methanol_heat", "fired_heat_ei"),
    ("H2 burned for heat", FLOW_COLORS["h2"], "hydrogen_heat", None),
    ("Electricity for heat", FLOW_COLORS["energy"], "electric_heat", "electric_heat_ei"),
    ("Other cracker electricity", "#8c6d1f", "process_power", "parasitic_power_ei"),
    ("CO2 capture + compression", FLOW_COLORS["co2"], "capture", "capture_power_ei"),
    ("CO2 liquefaction", "#f5b880", None, "liquefaction_power_ei"),
    ("Cracker plant + O&M", "#01aeae", "other_cracker", None),
    ("Reaction CO2 not captured", FLOW_COLORS["vent"], None, "reaction_vent_ei"),
)
REST_OF_CHAIN = ("Rest of supply chain", "rest_of_chain", "other_ei")


def print_heat_config_table(configs):
    base = configs[0]
    print(
        f"{'Configuration':<36}{'AUD/kg':>9}{'dCost':>8}{'EI':>7}{'dEI':>8}"
        f"{'react':>7}{'fired':>7}{'eheat':>7}{'power':>7}{'capt':>7}{'liquid':>7}{'other':>7}{'H2fuel':>8}"
    )
    for c in configs:
        if c.error:
            print(f"{c.name:<36}  failed: {c.error}")
            continue
        print(
            f"{c.name:<36}{c.delivered_cost_aud_per_kg_h2:>9.3f}"
            f"{c.delivered_cost_aud_per_kg_h2 - base.delivered_cost_aud_per_kg_h2:>+8.3f}"
            f"{c.delivered_ei_kg_co2_per_kg_h2:>7.3f}"
            f"{c.delivered_ei_kg_co2_per_kg_h2 - base.delivered_ei_kg_co2_per_kg_h2:>+8.3f}"
            f"{c.reaction_vent_ei:>7.3f}{c.fired_heat_ei:>7.3f}{c.electric_heat_ei:>7.3f}"
            f"{c.parasitic_power_ei:>7.3f}{c.capture_power_ei:>7.3f}{c.liquefaction_power_ei:>7.3f}"
            f"{c.other_ei:>7.3f}{c.heat_h2_kg_per_kg_h2:>8.3f}"
        )


# The cost panel's axis shows 0, breaks, then resumes here, so the cracker and
# heat parts (a few AUD/kg on a ~17 AUD/kg total) are wide enough to read.
HEAT_COST_AXIS_RESUME = 14  # AUD/kg H2
HEAT_COST_STUB_RATIO = 0.1  # width of the 0 stub relative to the main cost axis


BREAK_SLOPE = 2.5  # rise over run of the axis slashes and the bar cuts


def _break_marks(stub, main, y, height=0.6):
    """Mark a broken axis: pairs of slashes on the bottom axis line, one at each
    edge of the gap between its stub and main part, as in matplotlib's broken-
    axis demo, and every bar cut by the same pair of slanted lines, each through
    the gap edge at the bar's mid-height. The white band through a bar is then
    exactly as wide as the gap between the slashes.

    Each axes clips its bars square at its edge, so at each edge the half of
    the cut on the far side is filled in (grey, unclipped) and the half on the
    near side painted out (white). Call once the axes limits and the figure
    layout are final: the cut is set in inches."""
    kw = dict(marker=[(-1, -BREAK_SLOPE), (1, BREAK_SLOPE)], markersize=7, linestyle="none",
              color=INK, mec=INK, mew=0.9, clip_on=False, zorder=6)
    stub.plot([1], [0], transform=stub.transAxes, **kw)
    main.plot([0], [0], transform=main.transAxes, **kw)
    fig = stub.figure
    to_in = fig.dpi_scale_trans.inverted()
    x_left = to_in.transform(stub.transAxes.transform((1, 0)))[0]
    x_right = to_in.transform(main.transAxes.transform((0, 0)))[0]
    overlap = 0.01  # inches past the clipped edge, so no seam shows along it
    patch = dict(transform=fig.dpi_scale_trans, edgecolor="none", clip_on=False, zorder=5)
    for yi in y:
        lo, hi = (to_in.transform(stub.transData.transform((0, v)))[1]
                  for v in (yi - height / 2, yi + height / 2))
        mid, half_run = (lo + hi) / 2, (hi - lo) / 2 / BREAK_SLOPE
        for x, fill_above in ((x_left, True), (x_right, False)):
            # The cut line runs from (x - half_run, lo) through (x, mid) to
            # (x + half_run, hi). Left of it is bar, right of it gap at the
            # stub's edge; the reverse at the main part's.
            above = [(x - overlap, mid - overlap * BREAK_SLOPE), (x - overlap, hi),
                     (x + half_run, hi)]
            below = [(x - half_run, lo), (x + overlap, lo),
                     (x + overlap, mid + overlap * BREAK_SLOPE)]
            fig.add_artist(Polygon(above, closed=True,
                                   facecolor=NEUTRAL_LIGHT if fill_above else "white", **patch))
            fig.add_artist(Polygon(below, closed=True,
                                   facecolor="white" if fill_above else NEUTRAL_LIGHT, **patch))


def _outlined_barh(ax, y, widths, left, colour, outlined=True, height=0.6):
    """One series of horizontal bars in the thesis style: a tinted fill with a
    1.5 pt edge in the full colour, drawn 3 pt wide and clipped to its own bar
    so neighbours never paint over one another, as in the cost pies. Widths may
    be negative. An unoutlined series is a plain fill."""
    bars = ax.barh(y, widths, left=left, height=height,
                   color=tint(colour) if outlined else colour, edgecolor="none")
    if outlined:
        for bar, width in zip(bars, widths):
            if width == 0:
                continue
            edge = Rectangle(bar.get_xy(), bar.get_width(), bar.get_height(), facecolor="none",
                             edgecolor=colour, linewidth=3, zorder=bar.zorder + 1)
            ax.add_patch(edge)
            edge.set_clip_path(bar)


def _stacked_barh(ax, y, components, height=0.6):
    """Stacked horizontal bars of _outlined_barh segments. components:
    (colour, values, outlined) per segment, left to right."""
    left = np.zeros(len(y))
    for colour, values, outlined in components:
        _outlined_barh(ax, y, values, left, colour, outlined, height)
        left += values


def heat_config_plot(configs, mode_label, out_path):
    """Delivered cost and EI of each cracker heat configuration.

    Each bar starts with the rest of the supply chain in neutral grey, then the
    cracker and heat parts in HEAT_CATEGORIES colours, which mean the same in
    both panels; one legend below names them. Totals carry their change from
    the first (reference) configuration. Drawn at THESIS_WIDTH_IN with the house
    type sizes and saved uncropped, so the text is THESIS_FONT_PT when placed
    17 cm wide.
    """
    ok = [c for c in configs if not c.error]
    if not ok:
        Path(out_path).unlink(missing_ok=True)
        return
    base = ok[0]
    names = {
        "electric_zero_carbon_hp": "Zero-carbon power + heat pump",
        "electric_zero_carbon": "Zero-carbon power",
        "methanol_fired": "Methanol heater",
        "h2_fired": "Hydrogen heater",
        "electric_grid_jp": "Japan grid",
        "electric_grid_jp_hp": "Japan grid + heat pump",
        "electric_grid_jp_2030": "Japan 2030 grid",
        "electric_grid_jp_2030_hp": "Japan 2030 grid + heat pump",
    }
    # A fired heater burns product, so the chain makes more H2 upstream for the
    # same delivery: that extra on a second line.
    labels = [
        names.get(c.name, c.name)
        + (f"\n(+{c.upstream_h2_mtpa - base.upstream_h2_mtpa:.2f} Mt/year extra H2)"
           if c.name in ("methanol_fired", "h2_fired") else "")
        for c in ok
    ]
    labels[0] += " (ref.)"
    y = np.arange(len(ok))[::-1]
    # Fixed margins in inches, not constrained layout, which collapses the narrow
    # stub: room left for the configuration names, below for the shared legend.
    height = 0.3 * len(ok) + 2.1
    fig = plt.figure(figsize=(THESIS_WIDTH_IN, height))
    outer = fig.add_gridspec(1, 2, left=2.05 / THESIS_WIDTH_IN, right=0.97, top=1 - 0.3 / height,
                             bottom=1.15 / height, wspace=0.1)
    # (a) is a broken axis: a narrow stub from 0, a gap, then the main axis from
    # HEAT_COST_AXIS_RESUME. Both draw the whole bars; each clips to its range.
    cost_gs = outer[0].subgridspec(1, 2, width_ratios=[HEAT_COST_STUB_RATIO, 1], wspace=0.06)
    ax_stub = fig.add_subplot(cost_gs[0])
    ax_cost = fig.add_subplot(cost_gs[1], sharey=ax_stub)
    ax_ei = fig.add_subplot(outer[1], sharey=ax_stub)
    rest_label, rest_cost, rest_ei = REST_OF_CHAIN
    panels = (
        (ax_cost, rest_cost, 2, "Delivered cost [AUD/kg H2]", "delivered_cost_aud_per_kg_h2",
         HEAT_COST_AXIS_RESUME, 1.5),
        (ax_ei, rest_ei, 3, "Delivered EI [kg CO2/kg H2]", "delivered_ei_kg_co2_per_kg_h2", 0, 1.3),
    )
    for ax, rest, field, xlabel, attr, start, headroom in panels:
        get = (lambda c, k: c.cost_components[k]) if ax is ax_cost else getattr
        components = [(NEUTRAL_LIGHT, np.array([get(c, rest) for c in ok]), False)]
        components += [(colour, np.array([get(c, cat[field]) for c in ok]), True)
                       for cat in HEAT_CATEGORIES for colour in [cat[1]] if cat[field]]
        _stacked_barh(ax, y, components)
        if ax is ax_cost:
            # The stub spans only the start of the rest-of-chain segment; the
            # outlined segments' edges clip to their bars, not the axes, so
            # drawn here they would spill across the figure.
            _stacked_barh(ax_stub, y, components[:1])
        totals = [getattr(c, attr) for c in ok]
        for yi, total in zip(y, totals):
            delta = total - totals[0]
            text = f" {total:.2f}" + (f" ({delta:+.2f})" if yi != y[0] else "")
            ax.text(total, yi, text, va="center", fontsize=THESIS_FONT_PT, color=INK)
        ax.set_xlabel(xlabel, fontsize=THESIS_FONT_PT)
        ax.set_xlim(start, start + (max(totals) - start) * headroom)
        ax.tick_params(labelsize=THESIS_FONT_PT)
        ax.tick_params(axis="y", left=False, labelleft=False)
        ax.grid(axis="y", visible=False)
    # The stub at the main axis's scale, showing only its 0 tick.
    main_span = ax_cost.get_xlim()[1] - ax_cost.get_xlim()[0]
    ax_stub.set_xlim(0, main_span * HEAT_COST_STUB_RATIO)
    ax_stub.set_xticks([0])
    ax_stub.set_yticks(y, labels)
    ax_stub.tick_params(labelsize=THESIS_FONT_PT)
    ax_stub.grid(axis="y", visible=False)
    ax_cost.spines["left"].set_visible(False)
    # The theme's gridlines redrawn by hand, leaving out the one at the resume
    # value, which would run through the break.
    ax_cost.grid(axis="x", visible=False)
    for t in ax_cost.get_xticks():
        if HEAT_COST_AXIS_RESUME < t < ax_cost.get_xlim()[1]:
            ax_cost.axvline(t, color=plt.rcParams["grid.color"], zorder=0.5,
                            linewidth=plt.rcParams["grid.linewidth"],
                            linestyle=plt.rcParams["grid.linestyle"],
                            alpha=plt.rcParams["grid.alpha"])
    _break_marks(ax_stub, ax_cost, y)
    ax_stub.set_title("(a) Delivered cost", loc="left", fontsize=THESIS_FONT_PT,
                      fontweight="bold")
    ax_ei.set_title("(b) Delivered emissions intensity", loc="left", fontsize=THESIS_FONT_PT,
                    fontweight="bold")
    handles = [Patch(facecolor=NEUTRAL_LIGHT, edgecolor="none")]
    handles += [Patch(facecolor=tint(colour), edgecolor=colour, linewidth=1.5)
                for _, colour, _, _ in HEAT_CATEGORIES]
    entries = [rest_label] + [label for label, _, _, _ in HEAT_CATEGORIES]
    fig.legend(handles, entries, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=3, frameon=False,
               fontsize=THESIS_FONT_PT, labelspacing=0.35, columnspacing=1.5)
    # No title or footnotes: the thesis caption carries them (heat_config_notes).
    save_fig(fig, out_path, tight=False)


def heat_config_notes(configs):
    """Caption material for the heat configuration figure."""
    ok = [c for c in configs if not c.error]
    if not ok:
        return []
    base = ok[0]
    notes = [f"Heater label brackets show extra PEM H2 relative to the reference; net delivery stays {base.delivered_h2_mtpa:.3f} Mt/year."]
    notes.append("Fuel production is allocated from LCOMeOH, not added; H2 fuel uses its share of gross cracker H2. Shared shipping remains in rest of chain.")
    notes += [f"{c.name}: renewable/storage capacities {c.capacity_scale:.3f}x reference"
              + ("; exceeds workbook search bounds" if c.capacity_bounds_exceeded else "")
              for c in ok if c.capacity_scale != 1.0]
    notes += [f"Unavailable: {c.name}: {c.error}" for c in configs if c.error]
    notes.append("Reaction CO2 still forms and requires capture; every build verifies hourly inventories.")
    return notes


def _load_heat_configs(path):
    """Reconstruct HeatConfigResult list from a saved heat_config_comparison.json, for --replot."""
    return [HeatConfigResult(**d) for d in json.loads(Path(path).read_text())]


def write_heat_outputs(configs, mode_label, outdir, case, base, build_results=None):
    print_heat_config_table(configs)
    heat_config_plot(configs, mode_label, outdir / "heat_config_comparison.png")
    notes = heat_config_notes(configs)
    print("\n".join(notes))
    (outdir / "heat_config_comparison.json").write_text(
        json.dumps([asdict(c) for c in configs], indent=2), encoding="ascii")
    (outdir / "heat_config_metadata.json").write_text(json.dumps(dict(
        model_fingerprint=model_fingerprint(), case=case, target_h2_tpa=base.target.target_h2_tpa,
        cracker_typical=asdict(base.cracker), all_solved=all(not c.error for c in configs),
        basis="Reference allocation retained; process fleets resized; fired renewable builds proportionally scaled for hourly feasibility; frozen replay. No independent heat-case optimisation.",
        failures={c.name:c.error for c in configs if c.error}, caption_notes=notes), indent=2))
    if build_results is not None:
        from scripts.common.flow_diagrams import plot_meoh_flow
        from scripts.common.outputs import write_result
        from scripts.plots.site_allocation import plot_site_allocation
        for c in configs:
            directory = outdir / "heat_builds" / c.name
            if c.error:
                directory.mkdir(parents=True, exist_ok=True)
                for name in ("results.json", "deployment.json", "meoh_flow_diagram.png", "site_allocation.png"):
                    (directory / name).unlink(missing_ok=True)
                (directory / "run_metadata.json").write_text(json.dumps(dict(
                    case=f"{case}_{c.name}", model_fingerprint=model_fingerprint(), error=c.error,
                    hourly_feasible=False, annual_balanced=False), indent=2))
                continue
            result = build_results[c.name]
            write_result(result, directory, f"{case}_{c.name}")
            (directory / "deployment.json").write_text(json.dumps(c.deployment, indent=2, allow_nan=False))
            note = (f"{c.name}: MeOH {c.production_scale:.3f}x, upstream H2 {c.upstream_h2_scale:.3f}x reference; "
                    f"renewables/storage {c.capacity_scale:.3f}x")
            if c.capacity_bounds_exceeded:
                note += " (outside workbook search bounds)"
            plot_meoh_flow(result, outdir=directory, build_note=note)
            plot_site_allocation(result, outpath=directory / "site_allocation.png")



def run_heat_comparison(base, deployment_overrides, mode_label, outdir, case):
    build_results = {}
    configs = run_heat_configs(
        base=base,
        deployment_overrides=deployment_overrides,
        find_fired_feasible=True,
        build_results=build_results,
    )
    write_heat_outputs(configs, mode_label, outdir, case, base, build_results)


def main():
    parser = build_parser(__doc__, plot=False, replot=True)
    parser.add_argument("--case", choices=tuple(CASE_DIRS), default="optimised")
    parser.add_argument("--optimum", type=Path, default=OPT_DIR / "optimum.json",
                        help="Saved optimum.json of the optimised build (--case optimised)")
    parser.add_argument("--allow-stale", action="store_true",
                        help="Load an optimum saved under an older model fingerprint")
    args = parser.parse_args()
    build_dir = args.optimum.parent if args.case == "optimised" else CASE_DIRS[args.case]
    outdir = design_dir(build_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    mode_label = MODE_LABELS[args.case]
    base, deployment = (load_build(args.optimum, args.allow_stale) if args.case == "optimised"
                        else (ParamSet.base(), None))
    if args.replot:
        saved = outdir / "heat_config_comparison.json"
        if not saved.exists():
            parser.error(f"no saved {saved}; run without --replot first")
        write_heat_outputs(_load_heat_configs(saved), mode_label, outdir, args.case, base)
        return
    print("Cracker heat configurations ...")
    run_heat_comparison(base, deployment, mode_label, outdir, args.case)


if __name__ == "__main__":
    main()
