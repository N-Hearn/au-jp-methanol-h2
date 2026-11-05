"""Delivered cost and emissions intensity by supply-chain stage, for the thesis.

python -m scripts.plots.stage_breakdown

Al Ghafri et al. (2023) style: one stacked bar per pathway, the stages in
supply-chain order and coloured as the cost pies (COST_STAGES), with (a) the
delivered cost and (b) the operational emissions intensity on matching stages.
Reads saved outputs only -- outputs/reference_case/grey.json,
outputs/base_case/lng_ccs.json, outputs/optimised_build/results.json and the
proposed (10% margin) build -- so nothing is re-run and no output is changed.

Writes outputs/optimised_build/stage_breakdown.png and stage_breakdown.json
(the stage values behind the figure, for the thesis table).
"""

import json

import numpy as np

from scripts.common.cli import init_plotting
from scripts.common.outputs import OUTPUT_DIR, OPT_DIR, PROPOSED_DIR

init_plotting()

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scripts.common.plot_common import (COST_STAGES, INK, THESIS_FONT_PT,  # noqa: E402
                                  THESIS_TITLE_PT, THESIS_WIDTH_IN, save_fig, tint)

# Japan's hydrogen supply cost (CIF) targets, Basic Hydrogen Strategy (2023):
# 30 yen/Nm3 (about 334 yen/kg) by 2030 and 20 yen/Nm3 (about 222 yen/kg) by 2050.
# Plotting conversion only, not a model input: 2026 year-to-date average of about
# 111.4 JPY per AUD.
JPY_PER_AUD = 111.4
CIF_TARGETS_JPY_PER_KG = {"2030": 334.0, "2050": 222.0}
EI_THRESHOLD = 3.4  # kg CO2e/kg H2, Japan's low-carbon hydrogen threshold

STAGES = {name: (colour, keys) for name, colour, keys in COST_STAGES}
# Stacking order, bottom to top: the supply chain from energy supply in Australia
# to the CO2 chain in Japan. Stages a pathway does not have are skipped.
ORDER = ("Delivered LNG feed", "Electricity", "PEM electrolysis", "Battery + H2 storage",
         "H2 pipelines + compressors", "Methanol synthesis, shipping + storage",
         "ATR production", "MeOH cracker", "CO2 capture, return + storage", "CO2 recycle + make-up")
LABELS = {"MeOH cracker": "Methanol reformer (MSR)", "ATR production": "ATR (Japan)",
          "Electricity": "Renewables + transmission", "Delivered LNG feed": "Delivered LNG",
          "Methanol synthesis, shipping + storage": "MeOH synthesis, shipping + storage",
          "H2 pipelines + compressors": "H2 pipelines + compressors",
          "CO2 capture, return + storage": "CO2 capture, return + storage (blue)"}


def _read(path):
    return json.loads(path.read_text())


def _lng_cases():
    grey = _read(OUTPUT_DIR / "reference_case" / "grey.json")["result"]
    blue = _read(OUTPUT_DIR / "base_case" / "lng_ccs.json")
    g_cost = {"Delivered LNG feed": grey["feed_cost_aud_per_kg_h2"],
              "ATR production": grey["prod_cost_ex_feed_aud_per_kg_h2"]
              + grey["atr_electricity_cost_aud_per_kg_h2"]}
    g_ei = {"Delivered LNG feed": grey["upstream_ei_kg_co2_per_kg_h2"],
            "ATR production": grey["plant_gate_ei_kg_co2_per_kg_h2"]
            + grey["atr_electricity_ei_kg_co2_per_kg_h2"]}
    b_cost = dict(g_cost)
    b_cost["CO2 capture, return + storage"] = (
        blue["capture_cost_aud_per_kg_h2"] + blue["co2_liquefaction_cost_aud_per_kg_h2"]
        + blue["co2_shipping_cost_aud_per_kg_h2"] + blue["co2_storage_cost_aud_per_kg_h2"])
    bg = blue["grey"]
    b_ei = {"Delivered LNG feed": bg["upstream_ei_kg_co2_per_kg_h2"],
            "ATR production": blue["vented_co2_kg_per_kg_h2"] + bg["atr_electricity_ei_kg_co2_per_kg_h2"]
            + blue["capture_power_ei_kg_co2_per_kg_h2"],
            "CO2 capture, return + storage": blue["co2_liquefaction_ei_kg_co2_per_kg_h2"]
            + blue["co2_shipping_ei_kg_co2_per_kg_h2"]}
    for name, cost, ei, total_c, total_e in (
            ("grey", g_cost, g_ei, grey["delivered_cost_aud_per_kg_h2"], grey["delivered_ei_kg_co2_per_kg_h2"]),
            ("blue", b_cost, b_ei, blue["delivered_cost_aud_per_kg_h2"], blue["delivered_ei_kg_co2_per_kg_h2"])):
        if not (np.isclose(sum(cost.values()), total_c) and np.isclose(sum(ei.values()), total_e)):
            raise ValueError(f"{name} stages do not reconcile with the saved totals")
    return (g_cost, g_ei), (b_cost, b_ei)


def _meoh_case(result):
    breakdown = result["cost_breakdown_aud_per_kg_h2"]
    cost = {}
    for stage in ORDER:
        keys = STAGES[stage][1]
        value = sum(breakdown.get(k, 0.0) for k in keys)
        if value:
            cost[stage] = value
    # Emissions by the stage they arise in: ship fuel and the synthesis purge in
    # the methanol stage, uncaptured CO2, heat fuel and Japanese power at the
    # reformer, purchased Australian power with the electricity and CO2 lost on
    # the return leg with the CO2 recycle.
    ei = {"Electricity": result["pem_electricity_ei"] + result["synth_electricity_ei"],
          "Methanol synthesis, shipping + storage": result["ship_fuel_ei"] + result["synth_process_ei"]
          + result["meoh_loss_ei"],
          "MeOH cracker": result["cracker_vent_ei"] + result["cracker_electricity_ei"] + result["cracker_heat_ei"],
          "CO2 recycle + make-up": result["co2_capture_ei"]}
    ei = {k: v for k, v in ei.items() if v}
    if not (np.isclose(sum(cost.values()), result["delivered_cost_aud_per_kg_h2"])
            and np.isclose(sum(ei.values()), result["delivered_ei_kg_co2_per_kg_h2"])):
        raise ValueError("e-MeOH stages do not reconcile with the saved totals")
    return cost, ei


def _stack(ax, cases, which, fs):
    for i, case in enumerate(cases):
        values = case[which]
        bottom = 0.0
        for stage in ORDER:
            v = values.get(stage, 0.0)
            if v <= 0:
                continue
            colour = STAGES[stage][0]
            ax.bar(i, v, 0.6, bottom=bottom, color=tint(colour), edgecolor=colour, linewidth=1.0, zorder=2)
            bottom += v
        ax.text(i, bottom, f"{bottom:.2f}", ha="center", va="bottom", fontsize=fs,
                fontweight="normal", color=INK, zorder=3)


def main():
    (g_cost, g_ei), (b_cost, b_ei) = _lng_cases()
    opt = _read(OPT_DIR / "results.json")
    cases = [dict(label="Grey H2", cost=g_cost, ei=g_ei),
             dict(label="Blue H2", cost=b_cost, ei=b_ei)]
    c, e = _meoh_case(opt)
    cases.append(dict(label="e-MeOH\n0% margin", cost=c, ei=e))
    if (PROPOSED_DIR / "results.json").exists():
        c, e = _meoh_case(_read(PROPOSED_DIR / "results.json"))
        cases.append(dict(label="e-MeOH\n10% margin", cost=c, ei=e))

    fs = THESIS_FONT_PT
    fig, (a, b) = plt.subplots(1, 2, figsize=(THESIS_WIDTH_IN, 4.6), layout="constrained",
                              width_ratios=(1.2, 1))
    _stack(a, cases, "cost", fs)
    _stack(b, cases, "ei", fs)
    targets = {y: v / JPY_PER_AUD for y, v in CIF_TARGETS_JPY_PER_KG.items()}
    # Cost target lines stop past the last bar, with their labels in the space
    # to the right; the threshold label sits just above its line.
    n = len(cases)
    a.set_xlim(-0.5, n + 0.3)
    a.hlines(targets["2030"], -0.5, n - 0.5, color="black", linewidth=1, zorder=4)
    a.hlines(targets["2050"], -0.5, n - 0.5, color="black", linewidth=1, linestyle="--", zorder=4)
    b.axhline(EI_THRESHOLD, color="black", linewidth=1, linestyle=":", zorder=4)
    target_label = dict(ha="left", fontsize=fs - 1, color=INK, linespacing=1.1)
    a.text(n - 0.4, targets["2030"], f"2030 target\n{targets['2030']:.1f} AUD/kg", va="bottom", **target_label)
    a.text(n - 0.4, targets["2050"], f"2050 target\n{targets['2050']:.1f} AUD/kg", va="top", **target_label)
    b.text(n - 0.45, EI_THRESHOLD + 0.15, f"Low-carbon threshold\n{EI_THRESHOLD} kg CO2e/kg H2",
           ha="right", linespacing=1.1, va="bottom", fontsize=fs - 1, color=INK)
    costs = [sum(cs["cost"].values()) for cs in cases]
    eis = [sum(cs["ei"].values()) for cs in cases]
    a.set_ylim(0, max(costs) * 1.1)
    b.set_ylim(0, max(eis) * 1.1)
    a.set_title("(a) Delivered cost", loc="left", fontsize=fs)
    b.set_title("(b) Operational emissions intensity", loc="left", fontsize=fs)
    a.set_ylabel("AUD / kg H2", fontsize=fs)
    b.set_ylabel("kg CO2e / kg H2", fontsize=fs)
    for ax in (a, b):
        ax.set_xticks(range(len(cases)), [cs["label"] for cs in cases], fontsize=fs)
        ax.tick_params(axis="y", labelsize=fs)
        ax.grid(axis="x", visible=False)
    present = [s for s in ORDER if any(s in cs["cost"] or s in cs["ei"] for cs in cases)]
    by_colour = {}
    for st in present:  # stages sharing a standard colour share one legend entry
        by_colour.setdefault(STAGES[st][0], []).append(LABELS.get(st, st))
    handles = [Patch(facecolor=tint(c), edgecolor=c, label=" / ".join(n)) for c, n in by_colour.items()]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False, fontsize=fs - 1)
    fig.suptitle("Delivered Cost and Emissions by Supply Chain Stage", fontsize=THESIS_TITLE_PT,
                 fontweight="bold")
    save_fig(fig, OPT_DIR / "stage_breakdown.png", tight=False)

    table = {cs["label"].replace("\n", " "): dict(
        cost_aud_per_kg_h2={LABELS.get(k, k): round(v, 3) for k, v in cs["cost"].items()},
        ei_kg_co2e_per_kg_h2={LABELS.get(k, k): round(v, 3) for k, v in cs["ei"].items()},
        total_cost=round(sum(cs["cost"].values()), 3), total_ei=round(sum(cs["ei"].values()), 3))
        for cs in cases}
    table["targets"] = dict(jpy_per_aud=JPY_PER_AUD, cif_aud_per_kg={k: round(v, 2) for k, v in targets.items()},
                            ei_threshold=EI_THRESHOLD)
    (OPT_DIR / "stage_breakdown.json").write_text(json.dumps(table, indent=2))
    print(json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
