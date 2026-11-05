"""CO2 capture design options at the Japanese MSR plant, on the optimised build.

Usage:
    python -m scripts.design_analysis.capture_configs             # run options, write JSON + figure
    python -m scripts.design_analysis.capture_configs --replot    # redraw from the saved JSON

Outputs go to outputs/optimised_build/design_analysis/capture_config_*.
The workbook, the reference case and all other outputs are left untouched:
each option only overrides cracker capture fields on a copy of the Typical
inputs (see meoh_pathway/capture_configs.py).
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from scripts.common.cli import init_plotting

init_plotting()

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from meoh_pathway.capture_configs import CAPTURE_CONFIGS, CaptureConfigResult, run_capture_configs
from meoh_pathway.optimisation import model_fingerprint
from scripts.common.outputs import OPT_DIR, design_dir
from scripts.common.plot_common import (FLOW_COLORS, INK, NEUTRAL_LIGHT, THESIS_FONT_PT, THESIS_WIDTH_IN,
                                  save_fig, tint)
from scripts.design_analysis.heat_configs import HEAT_COST_STUB_RATIO, _break_marks, _stacked_barh, load_build

OUT_DIR = design_dir(OPT_DIR)
COST_AXIS_RESUME = 14  # AUD/kg H2, as the heat configuration figure

NAMES = {
    "pre_psa_90": "Pre-combustion PSA, 90 %",
    "pre_psa_95": "Pre-combustion PSA, 95 %",
    "pre_mea_90": "Pre-combustion MEA, 90 %",
    "post_amine_90": "Post-combustion amine, 90 %",
    "no_capture": "No capture (all CO2 as make-up)",
}
# (label, colour, cost component key, EI attribute)
CAPTURE_CATEGORIES = (
    ("CO2 capture + compression", FLOW_COLORS["co2"], "capture", None),
    ("Solvent regeneration heat", "#8c6d1f", "regeneration", None),
    ("Electricity for reformer heat", FLOW_COLORS["energy"], "reformer_heat", None),
    ("CO2 liquefaction + storage", "#f5b880", "co2_return", None),
    ("Make-up CO2 (DAC)", FLOW_COLORS["store"], "makeup", None),
    ("Reformer CO2 not captured", FLOW_COLORS["vent"], None, "reaction_vent_ei"),
)
REST = ("Rest of supply chain", "rest_of_chain", "other_ei")


def print_table(configs):
    print(f"{'Option':<16}{'AUD/kg':>9}{'dCost':>8}{'EI':>7}{'netEI':>7}{'capt Mt':>9}{'makeup Mt':>10}"
          f"{'capt':>7}{'regen':>7}{'heat':>7}{'return':>7}{'makeup':>7}")
    base = configs[0]
    for c in configs:
        if c.error:
            print(f"{c.name:<16}  failed: {c.error}")
            continue
        k = c.cost_components
        print(f"{c.name:<16}{c.delivered_cost_aud_per_kg_h2:9.3f}"
              f"{c.delivered_cost_aud_per_kg_h2 - base.delivered_cost_aud_per_kg_h2:+8.3f}"
              f"{c.delivered_ei_kg_co2_per_kg_h2:7.2f}{c.net_ei_kg_co2_per_kg_h2:7.2f}"
              f"{c.captured_co2_mtpa:9.2f}{c.makeup_co2_mtpa:10.2f}{k['capture']:7.3f}{k['regeneration']:7.3f}"
              f"{k['reformer_heat']:7.3f}{k['co2_return']:7.3f}{k['makeup']:7.3f}")


def capture_config_plot(configs, out_path):
    """Delivered cost (broken axis) and gross EI of each capture option, in the
    style of the heat configuration figure."""
    ok = [c for c in configs if not c.error]
    if not ok:
        return
    labels = [NAMES.get(c.name, c.name) for c in ok]
    labels[0] += " (ref.)"
    y = np.arange(len(ok))[::-1]
    height = 0.3 * len(ok) + 2.1
    fig = plt.figure(figsize=(THESIS_WIDTH_IN, height))
    outer = fig.add_gridspec(1, 2, left=2.05 / THESIS_WIDTH_IN, right=0.97, top=1 - 0.3 / height,
                             bottom=1.15 / height, wspace=0.1)
    cost_gs = outer[0].subgridspec(1, 2, width_ratios=[HEAT_COST_STUB_RATIO, 1], wspace=0.06)
    ax_stub = fig.add_subplot(cost_gs[0])
    ax_cost = fig.add_subplot(cost_gs[1], sharey=ax_stub)
    ax_ei = fig.add_subplot(outer[1], sharey=ax_stub)
    rest_label, rest_cost, rest_ei = REST
    panels = (
        (ax_cost, rest_cost, 2, "Delivered cost [AUD/kg H2]", "delivered_cost_aud_per_kg_h2",
         COST_AXIS_RESUME, 1.6),
        (ax_ei, rest_ei, 3, "Gross delivered EI [kg CO2/kg H2]", "delivered_ei_kg_co2_per_kg_h2", 0, 1.45),
    )
    for ax, rest, field, xlabel, attr, start, headroom in panels:
        get = (lambda c, k: c.cost_components[k]) if ax is ax_cost else getattr
        components = [(NEUTRAL_LIGHT, np.array([get(c, rest) for c in ok]), False)]
        components += [(cat[1], np.array([get(c, cat[field]) for c in ok]), True)
                       for cat in CAPTURE_CATEGORIES if cat[field]]
        _stacked_barh(ax, y, components)
        if ax is ax_cost:
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
    ax_cost.set_xticks(np.arange(COST_AXIS_RESUME, ax_cost.get_xlim()[1], 2))
    main_span = ax_cost.get_xlim()[1] - ax_cost.get_xlim()[0]
    ax_stub.set_xlim(0, main_span * HEAT_COST_STUB_RATIO)
    ax_stub.set_xticks([0])
    ax_stub.set_yticks(y, labels)
    ax_stub.tick_params(labelsize=THESIS_FONT_PT)
    ax_stub.grid(axis="y", visible=False)
    ax_cost.spines["left"].set_visible(False)
    ax_cost.grid(axis="x", visible=False)
    for t in ax_cost.get_xticks():
        if COST_AXIS_RESUME < t < ax_cost.get_xlim()[1]:
            ax_cost.axvline(t, color=plt.rcParams["grid.color"], zorder=0.5,
                            linewidth=plt.rcParams["grid.linewidth"],
                            linestyle=plt.rcParams["grid.linestyle"],
                            alpha=plt.rcParams["grid.alpha"])
    _break_marks(ax_stub, ax_cost, y)
    ax_stub.set_title("(a) Delivered cost", loc="left", fontsize=THESIS_FONT_PT, fontweight="bold")
    ax_ei.set_title("(b) Gross emissions intensity", loc="left", fontsize=THESIS_FONT_PT,
                    fontweight="bold")
    handles = [Patch(facecolor=NEUTRAL_LIGHT, edgecolor="none")]
    handles += [Patch(facecolor=tint(colour), edgecolor=colour, linewidth=1.5)
                for _, colour, _, _ in CAPTURE_CATEGORIES]
    entries = [rest_label] + [label for label, _, _, _ in CAPTURE_CATEGORIES]
    fig.legend(handles, entries, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=3, frameon=False,
               fontsize=THESIS_FONT_PT, labelspacing=0.35, columnspacing=1.5)
    save_fig(fig, out_path, tight=False)


def capture_config_notes(configs):
    ok = [c for c in configs if not c.error]
    notes = ["Gross EI shown; with non-fossil make-up CO2 every option nets to about zero "
             f"(net EI {', '.join(f'{c.name} {c.net_ei_kg_co2_per_kg_h2:+.2f}' for c in ok)})."]
    notes.append("Reference deployment, allocation and synthesis trains retained; process fleets resized; frozen replay.")
    notes += [f"Unavailable: {c.name}: {c.error}" for c in configs if c.error]
    return notes


def write_outputs(configs, outdir, base=None):
    outdir.mkdir(parents=True, exist_ok=True)
    print_table(configs)
    capture_config_plot(configs, outdir / "capture_config_comparison.png")
    notes = capture_config_notes(configs)
    print("\n".join(notes))
    (outdir / "capture_config_comparison.json").write_text(
        json.dumps([asdict(c) for c in configs], indent=2), encoding="ascii")
    meta = dict(model_fingerprint=model_fingerprint(),
                options={n: dict(note=t, overrides=ch) for n, t, ch in CAPTURE_CONFIGS},
                all_solved=all(not c.error for c in configs), caption_notes=notes,
                basis="Optimised build; only cracker capture fields overridden; process fleets resized; frozen replay.")
    (outdir / "capture_config_metadata.json").write_text(json.dumps(meta, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--optimum", default=str(OPT_DIR / "optimum.json"))
    ap.add_argument("--replot", action="store_true", help="Redraw from the saved JSON only")
    args = ap.parse_args()
    if args.replot:
        data = json.loads((OUT_DIR / "capture_config_comparison.json").read_text())
        configs = [CaptureConfigResult(**d) for d in data]
        print_table(configs)
        capture_config_plot(configs, OUT_DIR / "capture_config_comparison.png")
        return
    base, overrides = load_build(args.optimum)
    print("CO2 capture design options ...")
    configs = run_capture_configs(base=base, deployment_overrides=overrides)
    write_outputs(configs, design_dir(Path(args.optimum).parent), base)


if __name__ == "__main__":
    main()
