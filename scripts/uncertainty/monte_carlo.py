"""Economic cost and technical delivery uncertainty for the frozen builds (Figs 14-16).

python -m scripts.uncertainty.monte_carlo --case all --workers 8              # both modes, all cases
python -m scripts.uncertainty.monte_carlo --mode economic --case optimum
python -m scripts.uncertainty.monte_carlo --replot --mode technical --case all
python -m scripts.uncertainty.monte_carlo --margin 10 --mode technical --workers 8

Cases: grey (reference), grey_ccs (blue base case), optimum (the cost-optimum
build) and proposed (the 10% delivery-margin build). --margin <pct> runs the
technical draws on the <pct>% delivery-margin build instead, written to
outputs/monte_carlo/margin_<pct>pct/ (Fig 16).

--replot regenerates plots from a previous run's saved samples.json/run_metadata.json
without resampling (no dispatch, no optimum.json needed) -- use it to iterate on
plot appearance or after fixing a plotting bug.

economic draws prices and CAPEX on the verified operating schedule of the
frozen build, so feasibility is inherited and every draw has a cost.

technical draws the performance inputs against the as-built fleet: every
equipment count stays frozen and only the continuous ratings (transmission,
trunk/regional flow, pipeline compressor, terminal storage) re-rate and recost
to what each draw needs. Its headline output is how often the build still
delivers, and how far it falls short when it does not.
"""

from dataclasses import asdict
import json
from pathlib import Path
import re

from scripts.common.cli import init_plotting

init_plotting()

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from meoh_pathway.monte_carlo import (
    BUILD_CASES, CASES, design_point, COST, EI, UNMET_H2, McResult, cost_drivers, grouped_drivers,
    run_monte_carlo, summarise,
)
from meoh_pathway.uncertainty import LNG_CASES, MODES, UncertainInput
from meoh_pathway.optimisation import load_optimum
from meoh_pathway.parameters import PORT_SITES
from scripts.common.outputs import MARGIN_DIR, MC_DIR, OPT_DIR, PROPOSED_DIR, margin_mc_dir
from scripts.common.plot_common import (
    ACCENT, BASE_CASE, DIVERGING, NEUTRAL, REFERENCE_CASE, COST_COMPONENTS, FLOW_COLORS, save_fig,
    port_label,
)

CASE_DIRS = {"grey": "lng", "grey_ccs": "lng_ccs", "optimum": "optimised_build",
             "proposed": "proposed_build"}
PROPOSED_OPTIMUM = PROPOSED_DIR / "optimum.json"
CASE_LABELS = {
    "grey": f"{REFERENCE_CASE[0]}, {REFERENCE_CASE[1]}",
    "grey_ccs": f"{BASE_CASE[0]}, {BASE_CASE[1]}",
    "optimum": "e-MeOH (cost-optimum build)",
    "proposed": "e-MeOH (proposed build)",  # set_build_labels adds its margin
}
# Same convention as the pathway comparison: the reference/base case colours for
# the two LNG benchmarks, the flow-diagram H2 greens for the two MeOH builds.
CASE_COLORS = {"grey": REFERENCE_CASE[2], "grey_ccs": BASE_CASE[2],
               "optimum": FLOW_COLORS["h2"],
               # The PEM-capital green: close to the optimised build's green, as
               # the build it comes from, but still clearly distinct from it.
               "proposed": COST_COMPONENTS["pem_capital"][1]}


def _saved_margin(optimum_path):
    try:
        return json.loads(Path(optimum_path).read_text())["search_settings"].get("delivery_margin", 0)
    except (OSError, ValueError, KeyError, TypeError):
        return None


def set_build_labels(optimum_path, proposed_path):
    """Name each saved build by the delivery margin its search carried.

    --optimum pointed at a margin build (a margin_cases run) must not be
    labelled the cost-optimum, which is the zero-margin build.
    """
    margin = _saved_margin(proposed_path)
    if margin is not None:
        CASE_LABELS["proposed"] = (f"e-MeOH (proposed build, {margin:.0%} delivery margin)" if margin
                                   else "e-MeOH (proposed build)")
    margin = _saved_margin(optimum_path)
    if margin:
        CASE_LABELS["optimum"] = f"e-MeOH (build with {margin:.0%} delivery margin)"

# Known binding-constraint names from supply_chain.py's _fixed_annual_limit
# (dict keys), the hard pre-screens (cracker/synthesis/terminal storage) and
# the final allocation search, as (message substring, label, meaning), in
# longest-first order so a substring match picks the most specific label.
# failure_reason embeds a per-draw percentage ("... (83.41% of target)") that
# would otherwise make almost every reason its own value_counts() bucket.
_CONSTRAINTS = (
    ("operational allocation search found no verified hourly schedule",
     "No hourly schedule after reallocation",
     "Every built site passes its annual ceilings, but no split of the target across "
     "them could be verified hour by hour."),
    ("installed cracker fleet cannot meet required throughput",
     "Cracker fleet too small",
     "The Japan cracker units cannot crack enough methanol to return the H2 target "
     "(fleet-level pre-screen, before any dispatch)."),
    ("installed synthesis fleet cannot meet required throughput",
     "Synthesis trains too few",
     "A site's installed synthesis trains cannot make its share of methanol at nominal "
     "operating hours (pre-screen, before dispatch)."),
    ("installed terminal storage cannot hold a sailing parcel",
     "Terminal tank below ship parcel",
     "Methanol terminal storage at the Australian or Japanese end is smaller than one "
     "ship's cargo, so a vessel cannot be loaded or discharged."),
    ("hourly dispatch and storage limits",
     "Hourly dispatch / storage buffering",
     "Annual totals are sufficient, but battery and H2 storage cannot bridge the hourly "
     "gaps between renewable output and the synthesis schedule."),
    ("annual renewable energy",
     "Renewable energy too low",
     "A site's wind and solar do not generate enough energy in the year for its share "
     "of PEM, compression and synthesis load."),
    ("synthesis availability",
     "Synthesis availability",
     "A site's synthesis trains, at the drawn available hours per year, cannot produce "
     "its methanol share."),
    ("carriers",
     "Carriers too few",
     "The MeOH/LCO2 carrier fleet cannot carry methanol to Japan and the captured CO2 "
     "back in the year (voyage time, fuel and boil-off as drawn). Counted fleet-level."),
    ("PEM nameplate",
     "PEM nameplate too small",
     "A site's installed electrolyser modules cannot produce its H2 share at the drawn "
     "specific energy consumption and degradation."),
    ("trunk pipeline",
     "H2 trunk pipeline too small",
     "The renewables-to-port H2 pipeline cannot move a site's H2 share in the year."),
    ("pipeline compressor",
     "Trunk compressor too small",
     "The trunk pipeline compressor cannot supply the compression energy a site's H2 "
     "flow needs."),
    ("terminal sailing parcel",
     "Terminal tank below ship parcel",
     "Methanol terminal storage at the Australian or Japanese end is smaller than one "
     "ship's cargo, so a vessel cannot be loaded or discharged."),
    ("annual dispatch did not converge to periodic storage inventory",
     "Storage cycle did not converge",
     "A site's dispatch never reached a repeating year-end battery / H2 storage state, "
     "so the year could not be verified."),
)
_UNMET_BALANCE = (
    "Unmet H2 or electricity after dispatch",
    "Dispatch ran with no named equipment limit, but left some H2 demand or electricity "
    "demand unserved.")
_CONSTRAINT_MEANINGS = dict((label, meaning) for _, label, meaning in _CONSTRAINTS)
_CONSTRAINT_MEANINGS[_UNMET_BALANCE[0]] = _UNMET_BALANCE[1]


# Two more message shapes carry per-draw detail that would split the counts:
# the fallback "Unmet H2 {kg} kg; unmet electricity {mwh} MWh" from
# meoh_pathway.monte_carlo._draw_metrics, and a dispatch failure prefixed with
# its port ("Newcastle: Annual dispatch did not...").
_UNMET_BALANCE_RE = re.compile(r"^unmet h2 ")
_SITE_PREFIX_RE = re.compile(r"^[A-Za-z]+: ")
_SITE_REASON_RE = re.compile(r"^([A-Za-z]+): (.*)$")
FLEET = "fleet"
# Shared fleets: a shortfall is the fleet's, even where the build records a
# per-port vessel count.
_FLEET_CONSTRAINTS = {"Carriers too few",
                      "Cracker fleet too small"}


def _constraint_label(reason):
    lower = reason.lower()
    for match, label, _ in _CONSTRAINTS:
        if match.lower() in lower:
            return label
    if _UNMET_BALANCE_RE.match(lower):
        return _UNMET_BALANCE[0]
    return _SITE_PREFIX_RE.sub("", reason)


def _reason_sites(reason):
    """(constraint, site) pairs a failure reason names; fleet-level reasons name no site.

    A shortfall message lists every port's tightest ceiling, not only the ports
    that fell short.
    """
    if _UNMET_BALANCE_RE.match(reason.lower()):
        return [(_UNMET_BALANCE[0], FLEET)]
    pairs = []
    for part in reason.split("; "):
        m = _SITE_REASON_RE.match(part)
        if m and m.group(1) in PORT_SITES:
            label = _constraint_label(m.group(2))
            fleet = label in _FLEET_CONSTRAINTS
            pairs.append((label, FLEET if fleet else m.group(1)))
    return list(dict.fromkeys(pairs)) or [(_constraint_label(reason), FLEET)]


def _failure_constraint_sites(result):
    """One row per (constraint, site, draw): a draw counts at every site its reason names."""
    failed = result.samples.loc[result.samples["error"] != "", "failure_reason"]
    rows = {(label, site, draw) for draw, reason in failed.items()
            for label, site in _reason_sites(reason)}
    return pd.DataFrame(sorted(rows), columns=["constraint", "site", "draw"])


def _failure_constraint_counts(result):
    """Draws naming each constraint, and the sites it bound at, most frequent first."""
    rows = _failure_constraint_sites(result)
    draws = rows.groupby("constraint")["draw"].nunique().sort_values(ascending=False, kind="stable")
    sites = rows.groupby(["constraint", "site"]).size().unstack(fill_value=0)
    order = [s for s in (*PORT_SITES, FLEET) if s in sites.columns]
    return draws, sites.reindex(index=draws.index, columns=order)


def write_failure_constraints(result, outdir):
    draws, sites = _failure_constraint_counts(result)
    _write_json({label: dict(draws=int(n), sites={s: int(v) for s, v in sites.loc[label].items() if v})
                 for label, n in draws.items()}, outdir / "failure_constraints.json")
    failed = int((result.samples["error"] != "").sum())
    lines = [f"# Binding constraints: {result.case}, {result.mode} mode", "",
             f"{failed} of {len(result.samples)} draws failed. A draw counts under every "
             "constraint its failure reason names, once per site that constraint bound at, "
             f"so the columns do not sum to the failed draws. \"{FLEET}\" marks a fleet-level "
             "limit: the cracker and carrier fleets. Full messages are in samples.json "
             "(failure_reason).", "",
             "| Constraint | Draws | Sites | Meaning |", "| --- | ---: | --- | --- |"]
    seen = set()
    for label, n in draws.items():
        meaning = _CONSTRAINT_MEANINGS.get(label, "Unrecognised message; see samples.json.")
        at = ", ".join(f"{port_label(s)} {v}" for s, v in sites.loc[label].items() if v)
        lines.append(f"| {label} | {n} | {at} | {meaning} |")
        seen.add(label)
    for label, meaning in _CONSTRAINT_MEANINGS.items():
        if label not in seen:
            lines.append(f"| {label} | 0 | | {meaning} |")
            seen.add(label)
    (outdir / "failure_constraints.md").write_text("\n".join(lines) + "\n", encoding="ascii")


def _cluster_key(values):
    """Values rounded to 0.1% relative precision.

    A value set by a small integer equipment count (e.g. a resized cracker
    fleet's capacity factor) clusters into a handful of near-identical floats
    rather than varying continuously. Rounding merges floating-point noise
    within a cluster while keeping genuinely different values apart.
    """
    return (values / max(abs(values).max(), 1e-12)).round(3)


# The design point -- the saved build at workbook Typical inputs -- in the same
# red dashed style as the deterministic result on the economic distribution.
DESIGN_STYLE = dict(color=DIVERGING[1], linewidth=1.5, linestyle="--")
DESIGN_LABEL = "Design point (typical inputs)"


def _dist_panel(ax, values, label, color=ACCENT, design=None):
    """Histogram with P10/P50/P90 reference lines, or a placeholder if empty.

    design marks where the build itself sits at Typical inputs.

    Few enough distinct clusters and an auto-binned histogram would spread them
    across mostly-empty bins and look like noise (spike, gap, spike), so plot
    the cluster counts as discrete bars instead.
    """
    values = pd.Series(values).dropna()
    if values.empty:
        ax.text(.5, .5, "No quantified outputs", ha="center", va="center", transform=ax.transAxes)
        ax.set(xlabel=label, ylabel="Draw count", xticks=[], yticks=[])
        return
    scale = max(abs(values).max(), 1e-12)
    cluster_key = _cluster_key(values)
    n_clusters = cluster_key.nunique()
    if n_clusters <= 10:
        grouped = values.groupby(cluster_key.values)
        stats = grouped.agg(["mean", "size"]).sort_values("mean")
        xs, heights = stats["mean"].values, stats["size"].values
        width = np.min(np.diff(xs)) * 0.6 if len(xs) > 1 else scale * 0.02
        ax.bar(xs, heights, width=width, color=color)
    else:
        ax.hist(values, bins=min(30, max(1, int(np.sqrt(len(values))))), color=color)
    if n_clusters > 1:
        for q, style in ((10, ":"), (50, "-"), (90, ":")):
            ax.axvline(np.percentile(values, q), color="black", linewidth=1, linestyle=style, alpha=.6)
    if design is not None and np.isfinite(design):
        ax.axvline(design, **DESIGN_STYLE, label=DESIGN_LABEL)
    ax.set(xlabel=label, ylabel="Draw count")


def plot_drivers(drivers, xlabel, path):
    """Ranked rank-correlation bars; shared-dependency groups appear once."""
    if drivers.empty:
        Path(path).unlink(missing_ok=True)
        return
    colors = [DIVERGING[1] if v >= 0 else DIVERGING[0] for v in drivers.values]
    fig, ax = plt.subplots(figsize=(9, 0.4*len(drivers)+1.2), layout="constrained")
    y = np.arange(len(drivers))[::-1]
    ax.barh(y, drivers.values, color=colors, height=0.6)
    ax.set_yticks(y, drivers.index)
    for yi, rho in zip(y, drivers.values):
        ax.text(rho + (0.02 if rho >= 0 else -0.02), yi, f"{rho:+.2f}",
                va="center", ha="left" if rho >= 0 else "right", fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(xlabel=xlabel, xlim=(-1, 1))
    save_fig(fig, path)


def plot_economic(result, outdir):
    summary = summarise(result)
    values = result.samples[COST].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    axes[0].hist(values, bins=min(40, max(1, int(np.sqrt(len(values))))), color=ACCENT)
    if "P10_cost" in summary:
        axes[0].axvspan(summary["P10_cost"], summary["P90_cost"], color=ACCENT, alpha=.12, label="P10-P90")
        axes[0].axvline(summary["P50_cost"], color="black", linewidth=1.2, label="P50")
    axes[0].axvline(summary["deterministic_cost"], color=DIVERGING[1], linewidth=1.5, linestyle="--",
                     label="Deterministic result")
    axes[0].set(xlabel="Delivered cost [AUD/kg H2]", ylabel="Draw count")
    axes[0].legend(fontsize=8, frameon=False)

    sorted_values = np.sort(values)
    axes[1].step(sorted_values, np.arange(1, len(sorted_values)+1)/len(sorted_values), color=ACCENT)
    axes[1].axvline(summary["deterministic_cost"], color=DIVERGING[1], linewidth=1.5, linestyle="--")
    if "prob_cost_le_deterministic" in summary:
        p = summary["prob_cost_le_deterministic"]
        axes[1].axhline(p, color="black", linewidth=.8, linestyle=":")
        axes[1].annotate(f"P(cost ≤ deterministic) = {p:.0%}", xy=(summary["deterministic_cost"], p),
                          xytext=(8, 8), textcoords="offset points", fontsize=8)
    axes[1].set(xlabel="Delivered cost [AUD/kg H2]", ylabel="Share of draws", ylim=(0, 1))
    axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    save_fig(fig, outdir / "economic_cost_distribution.png")

    drivers = grouped_drivers(result, result.feasible[COST])
    _write_json(drivers.to_dict(), outdir / "cost_drivers_grouped.json")
    plot_drivers(drivers, "Spearman correlation with delivered cost", outdir / "uncertainty_drivers.png")

    components = result.feasible.filter(regex=r"^component\.")
    if not components.empty:
        stats = components.describe(percentiles=[.1, .5, .9])
        keys = [c.removeprefix("component.") for c in components.columns]
        keys.sort(key=lambda k: stats[f"component.{k}"]["50%"])
        p50 = np.array([stats[f"component.{k}"]["50%"] for k in keys])
        lo = p50 - np.array([stats[f"component.{k}"]["10%"] for k in keys])
        hi = np.array([stats[f"component.{k}"]["90%"] for k in keys]) - p50
        labels = [COST_COMPONENTS.get(k, (k.replace("_", " ").capitalize(), None))[0] for k in keys]
        colors = [COST_COMPONENTS.get(k, (None, ACCENT))[1] or ACCENT for k in keys]
        fig, ax = plt.subplots(figsize=(10, 0.4*len(keys)+2), layout="constrained")
        y = np.arange(len(keys))
        ax.barh(y, p50, xerr=[lo, hi], color=colors, error_kw=dict(ecolor="#333333", capsize=3, linewidth=1))
        ax.set_yticks(y, labels)
        ax.set(xlabel="Delivered cost contribution, median with P10-P90 [AUD/kg H2]")
        save_fig(fig, outdir / "cost_components.png")


# Thesis figures compare the LNG benchmarks with the optimised e-MeOH build
# only; the full all-case comparison is still written alongside.
THESIS_COMPARISONS = {
    "cost_distribution_comparison_thesis.png": ("grey", "optimum", "proposed"),
    "cost_distribution_comparison_thesis_ccs.png": ("grey", "grey_ccs", "optimum", "proposed"),
}


def plot_case_comparisons(results, outdir):
    """Write the all-case comparison and each thesis subset whose cases all ran."""
    plot_case_comparison(results, outdir)
    for filename, cases in THESIS_COMPARISONS.items():
        # A subset without the proposed build is still drawn when that case did not run.
        subset = [r for r in results if r.case in cases]
        wanted = [c for c in cases if c != "proposed" or c in {r.case for r in results}]
        if len(subset) == len(wanted):
            plot_case_comparison(subset, outdir, filename=filename)


def plot_case_comparison(results, outdir, filename="cost_distribution_comparison.png"):
    """Overlay delivered-cost densities across the given cases for one mode, P50 marked.

    The feasible count joins a label only where some draws were excluded.
    """
    fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
    for r in results:
        cost = r.feasible[COST].dropna()
        if cost.empty:
            continue
        p50 = np.percentile(cost, 50)
        label = f"{CASE_LABELS[r.case]}: P50 {p50:.2f} AUD/kg"
        if len(cost) < len(r.samples):
            label += f" ({len(cost)}/{len(r.samples)} draws deliver)"
        color = CASE_COLORS[r.case]
        if cost.nunique() <= 1:
            ax.axvline(cost.iloc[0], color=color, label=label)
        else:
            sns.kdeplot(cost, color=color, fill=True, alpha=.15, label=label, ax=ax)
            ax.axvline(p50, color=color, linewidth=1, linestyle="--")
    # The gap between the LNG and e-MeOH costs is free space for the legend.
    ax.legend(frameon=False, fontsize=9, loc="upper center")
    ax.set(xlabel="Delivered cost [AUD/kg H2]", ylabel="Density")
    save_fig(fig, outdir / filename)


_MAX_CONSTRAINT_BARS = 12
# Fixed per port so a site keeps its colour across cases and modes.
# tab10's grey is dropped so no port reads as the fleet-level grey.
_SITE_PALETTE = [c for i, c in enumerate(sns.color_palette("tab10")) if i != 7] + [sns.color_palette("tab20")[1]]
SITE_COLORS = dict(zip(PORT_SITES, _SITE_PALETTE), **{FLEET: NEUTRAL})


def plot_delivery_outcomes(result, outdir):
    categories = result.samples["failure_category"].replace("", "verified full delivery").value_counts()
    draws, sites = _failure_constraint_counts(result)
    # _constraint_label collapses the known message shapes, but failure_reason is
    # free text, so an unrecognised one still gets a bar per draw. Cap the bars
    # so that cannot blow up the figure.
    if len(sites) > _MAX_CONSTRAINT_BARS:
        rest = sites.iloc[_MAX_CONSTRAINT_BARS - 1:].sum().rename("other reasons (grouped)")
        sites = pd.concat([sites.iloc[:_MAX_CONSTRAINT_BARS - 1], rest.to_frame().T])
    height = max(3, 0.5*max(len(categories), len(sites) or 1) + 1.5)
    fig, axes = plt.subplots(1, 2, figsize=(13, height), layout="constrained")
    feasible_frac = len(result.feasible) / len(result.samples)
    fig.suptitle(f"{CASE_LABELS[result.case]}: delivery outcomes across {len(result.samples)} draws "
                 f"({feasible_frac:.1%} verified feasible)")

    cat_colors = [ACCENT if c == "verified full delivery" else DIVERGING[1] for c in categories.index]
    axes[0].barh(categories.index[::-1], categories.values[::-1], color=cat_colors[::-1])
    axes[0].set(xlabel="Draws", title="Outcome category")

    if sites.empty:
        axes[1].text(.5, .5, "No infeasible draws", ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set(xticks=[], yticks=[])
    else:
        rows = sites.iloc[::-1]
        left = np.zeros(len(rows))
        for site in rows.columns:
            axes[1].barh(rows.index, rows[site], left=left, color=SITE_COLORS[site],
                         label=port_label(site) if site != FLEET else "Fleet-level")
            left += rows[site].to_numpy()
        axes[1].set(xlabel="Site occurrences (a draw counts once at each site it names)",
                    title="Binding constraint on infeasible draws, by site")
        axes[1].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    save_fig(fig, outdir / "delivery_outcomes.png")


def plot_delivery_dashboard(result, outdir):
    ok = result.feasible
    excluded = 1 - len(ok)/len(result.samples)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), layout="constrained")
    fig.suptitle(f"{CASE_LABELS[result.case]}: technical delivery and performance\n"
                 f"{len(ok)}/{len(result.samples)} verified deliveries; {excluded:.1%} excluded from feasible distributions")
    design = result.reference_metrics
    ax = axes[0, 0]
    if COST in ok and EI in ok and not ok.empty:
        ax.scatter(ok[COST], ok[EI], s=18, alpha=.6, color=ACCENT, edgecolor="none", label="Feasible draws")
    else:
        ax.text(.5, .5, "No quantified outputs", ha="center", va="center", transform=ax.transAxes)
    if np.isfinite([result.reference_cost, result.reference_ei]).all():
        ax.scatter(result.reference_cost, result.reference_ei, marker="*", s=260, color=DIVERGING[1],
                   edgecolor="white", linewidth=1, zorder=5, label=DESIGN_LABEL)
        ax.legend(frameon=False, fontsize=8, loc="best")
    ax.set(xlabel="Feasible delivered cost [AUD/kg H2]", ylabel="Feasible emissions [kg CO2/kg H2]")
    # hydrogen_shortfall_kg is exactly 0 for every feasible draw by construction
    # (a verified draw delivered the full target) -- plotting all draws buries the
    # excluded-draw shortfall distribution under a spike of zeros. Restrict to
    # excluded draws only, coloured red to match the infeasible-outcome convention.
    excluded_draws = result.samples[result.samples["error"] != ""]
    shortfall_kt = excluded_draws.get("hydrogen_shortfall_kg", pd.Series(dtype=float)) / 1e6
    _dist_panel(axes[0, 1], shortfall_kt, "Hydrogen shortfall [kt/yr], excluded draws only", color=DIVERGING[1])
    _dist_panel(axes[1, 0], ok.get("pem_capacity_factor", pd.Series(dtype=float)), "Feasible PEM utilisation",
                design=design.get("pem_capacity_factor"))
    _dist_panel(axes[1, 1], ok.get("wind_curtailment_fraction", pd.Series(dtype=float)), "Feasible curtailment fraction",
                design=design.get("wind_curtailment_fraction"))
    if design.get("pem_capacity_factor") is not None:
        axes[1, 0].legend(frameon=False, fontsize=8, loc="upper left")
    save_fig(fig, outdir / "delivery_dashboard.png")


_METRIC_LABELS = {
    "energy_intensity_gj_per_t_h2": "Energy intensity [GJ/t H2]",
    "pem_capacity_factor": "PEM capacity factor",
    "synth_capacity_factor": "Synthesis capacity factor",
    "cracker_capacity_factor": "Cracker capacity factor",
    "battery_cycles_per_year": "Battery cycles [per year]",
}


def plot_performance_distributions(result, outdir):
    ok = result.feasible
    excluded = 1 - len(ok)/len(result.samples)
    columns = [c for c in ok if c.endswith("_capacity_factor") or "storage_working_fraction" in c
               or c in ("battery_cycles_per_year", "energy_intensity_gj_per_t_h2")]
    # A frozen design can hold a metric exactly constant (e.g. a fixed cracker
    # fleet's capacity factor in technical mode) -- a histogram of one bar is
    # not a distribution, so drop it rather than chart it.
    columns = [c for c in columns if _cluster_key(ok[c].dropna()).nunique() > 1]
    if not columns:
        return
    storage_cols = [c for c in columns if "storage_working_fraction" in c]
    single_cols = [c for c in columns if c not in storage_cols]
    n_panels = len(single_cols) + (1 if storage_cols else 0)
    ncols = min(3, n_panels)
    nrows = -(-n_panels // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2*ncols, 3.2*nrows), layout="constrained", squeeze=False)
    fig.suptitle(f"Feasible-draw performance; {excluded:.1%} of all draws excluded")
    flat = axes.flatten()
    for ax, column in zip(flat, single_cols):
        _dist_panel(ax, ok[column], _METRIC_LABELS.get(column, column),
                    design=result.reference_metrics.get(column))
    if any(c in result.reference_metrics for c in single_cols):
        fig.legend(handles=[plt.Line2D([], [], **DESIGN_STYLE, label=DESIGN_LABEL)],
                   frameon=False, fontsize=8, loc="outside lower center")
    if storage_cols:
        ax = flat[len(single_cols)]
        data = [ok[c].dropna() for c in storage_cols]
        ports = [port_label(c.split(".")[0]) for c in storage_cols]
        order = np.argsort([d.median() for d in data])
        data = [data[i] for i in order]
        ports = [ports[i] for i in order]
        ax.boxplot(data, orientation="horizontal", patch_artist=True,
                   boxprops=dict(facecolor=ACCENT, alpha=.5), medianprops=dict(color="black"))
        ax.set_yticks(range(1, len(ports)+1), ports)
        ax.set(xlabel="H2 storage working fraction", title="Per-site storage utilisation")
    for ax in flat[n_panels:]:
        ax.set_visible(False)
    save_fig(fig, outdir / "performance_distributions.png")


def plot_binding_constraints(result, outdir):
    """Draws in which each constraint stops the build delivering."""
    draws, _ = _failure_constraint_counts(result)
    path = outdir / "binding_constraints.png"
    if draws.empty:
        path.unlink(missing_ok=True)
        return
    draws = draws.head(_MAX_CONSTRAINT_BARS)
    n = len(result.samples)
    fig, ax = plt.subplots(figsize=(9, 0.45*len(draws)+1.3), layout="constrained")
    y = np.arange(len(draws))[::-1]
    ax.barh(y, draws.values, color=DIVERGING[1], height=0.6)
    ax.set_yticks(y, draws.index)
    for yi, count in zip(y, draws.values):
        ax.text(count + n*0.005, yi, f"{count} ({count/n:.0%})", va="center", fontsize=8)
    ax.set(xlabel=f"Draws in which the constraint binds (of {n}; a draw can name several)",
           xlim=(0, draws.max()*1.18))
    save_fig(fig, path)


def plot_shortfall_drivers(result, outdir):
    """Inputs ranked by correlation with failing to deliver, over every draw.

    Not with shortfall size: a pre-screen (e.g. cracker fleet) stops a draw
    before dispatch and records only its own limit, hiding any larger dispatch
    shortfall, so low cracker availability would wrongly read as a small one.
    """
    failed = (result.samples["error"] != "").astype(float)
    drivers = grouped_drivers(result, failed)
    _write_json(drivers.to_dict(), outdir / "shortfall_drivers_grouped.json")
    plot_drivers(drivers, "Spearman correlation with failing to deliver the target (all draws)",
                 outdir / "shortfall_drivers.png")


def plot_mode(result, outdir):
    (plot_economic if result.mode == "economic" else plot_technical)(result, outdir)


def plot_technical(result, outdir):
    plot_delivery_outcomes(result, outdir)
    plot_delivery_dashboard(result, outdir)
    plot_performance_distributions(result, outdir)
    if UNMET_H2 in result.samples:  # not the LNG cases
        plot_binding_constraints(result, outdir)
        plot_shortfall_drivers(result, outdir)


def _load_result(outdir):
    """Reconstruct an McResult from a previous run's saved outputs, for --replot."""
    metadata = json.loads((outdir / "run_metadata.json").read_text())
    summary = json.loads((outdir / "summary.json").read_text())
    samples = pd.DataFrame(json.loads((outdir / "samples.json").read_text()))
    bounds = [UncertainInput(b["name"], b["group"], b["field"], b["base"], b["low"], b["high"],
                             b["distribution"], b["evidence"], b["dependency"])
              for b in metadata["sampled_inputs"]]
    return McResult(case=metadata["case"], bounds=bounds, samples=samples,
                    reference_cost=summary["deterministic_cost"],
                    reference_ei=summary.get("deterministic_ei", float("nan")),
                    mode=metadata["mode"], fingerprint=metadata["model_fingerprint"],
                    reference_equipment=metadata.get("reference_equipment", {}),
                    reference_metrics=metadata.get("reference_metrics", {}))


def _write_json(obj, path):
    path.write_text(json.dumps(obj, indent=2), encoding="ascii")


def write_outputs(result, outdir, *, seed, workers):
    outdir.mkdir(parents=True, exist_ok=True)
    summary = summarise(result)
    _write_json(result.samples.to_dict(orient="records"), outdir / "samples.json")
    _write_json(summary, outdir / "summary.json")
    _write_json(cost_drivers(result, len(result.inputs)).to_dict(), outdir / "cost_drivers.json")
    components = result.feasible.filter(regex=r"^component\.")
    if not components.empty:
        _write_json(components.describe(percentiles=[.05, .1, .5, .9, .95]).to_dict(), outdir / "cost_components.json")
    if result.mode != "economic":
        _write_json(result.samples["failure_category"].replace("", "verified full delivery")
                    .value_counts().to_dict(), outdir / "failure_categories.json")
        write_failure_constraints(result, outdir)
        performance = result.feasible.drop(columns=result.inputs+['error','failure_reason','failure_category'])
        _write_json(performance.describe(percentiles=[.05, .1, .5, .9, .95]).to_dict(), outdir / "feasible_performance.json")
    metadata = dict(model_fingerprint=result.fingerprint, seed=seed, samples=len(result.samples),
                    case=result.case, mode=result.mode, workers=workers,
                    reference_equipment=result.reference_equipment,
                    reference_metrics=result.reference_metrics,
                    conditional_on_feasibility=result.mode != "economic",
                    fixed_installed_equipment=True,
                    operating_allocation_can_change=result.mode != "economic" and result.case not in LNG_CASES,
                    hourly_dispatch_rerun=result.mode != "economic" and result.case in BUILD_CASES,
                    feasibility_inherited=result.mode == "economic",
                    weather="fixed available single-year profiles; no weather ensemble supplied",
                    sampled_inputs=[asdict(b) for b in result.bounds])
    _write_json(metadata, outdir / "run_metadata.json")
    lines = [summary['distribution_basis'],
             f"Verified delivery: {summary['feasible_fraction']:.1%} of all draws.",
             f"Excluded: {summary['excluded_fraction']:.1%} of all draws.",
             "Explicit workbook distributions are screening assumptions, not empirical confidence intervals.",
             "Annual capacity failures, unsuccessful dispatch searches and numerical errors are reported separately.",
             "A failed dispatch search is not proof that every operating policy fails.",
             "Unquantified shortfalls remain NaN. Failure constraints are counted in failure_constraints.json and explained in failure_constraints.md."]
    (outdir / "limitations.txt").write_text("\n".join(lines), encoding="ascii")
    return summary


def main():
    from scripts.common.cli import build_parser

    parser = build_parser(__doc__, outdir=MC_DIR, plot=True, replot=True)
    parser.add_argument("--mode", nargs="+", choices=MODES+("all",), default=list(MODES))
    parser.add_argument("--case", nargs="+", choices=CASES+("all",), default=["optimum"])
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--optimum", type=Path, default=OPT_DIR / "optimum.json")
    parser.add_argument("--proposed", type=Path, default=PROPOSED_OPTIMUM,
                        help="Saved optimum.json of the proposed build (default: the 10%% "
                             "delivery-margin optimum)")
    parser.add_argument("--margin", type=float, default=None, metavar="PCT",
                        help="Draw the PCT%% delivery-margin build as the optimum case and write "
                             "to outputs/monte_carlo/margin_<PCT>pct/ (Fig 16)")
    parser.add_argument("--allow-stale", action="store_true",
                        help="Load an optimum saved under an older model fingerprint")
    args = parser.parse_args()
    if args.margin is not None:
        args.optimum = MARGIN_DIR / f"margin_{args.margin:g}pct" / "optimum.json"
        args.outdir = margin_mc_dir(args.margin)
        if args.case == ["optimum"] or "all" in args.case:
            args.case = ["optimum"]
    if "all" in args.mode:
        args.mode = list(MODES)
    if "all" in args.case:
        args.case = list(CASES)
    set_build_labels(args.optimum, args.proposed)
    builds = {"optimum": args.optimum, "proposed": args.proposed}
    if args.samples < 2:
        parser.error("--samples must be at least 2")
    if args.replot:
        for mode in dict.fromkeys(args.mode):
            mode_results = []
            for case in dict.fromkeys(args.case):
                outdir = args.outdir / mode / CASE_DIRS[case]
                if not (outdir / "samples.json").exists():
                    print(f"  no saved samples for {mode}/{case} at {outdir}, skipping", flush=True)
                    continue
                result = _load_result(outdir)
                if not result.reference_metrics and case not in LNG_CASES:
                    result.reference_metrics = design_point(case, builds.get(case), mode, args.allow_stale)
                    path = outdir / "run_metadata.json"
                    _write_json(json.loads(path.read_text()) | dict(reference_metrics=result.reference_metrics),
                                path)
                plot_mode(result, outdir)
                if mode != "economic":
                    write_failure_constraints(result, outdir)
                mode_results.append(result)
                print(f"Replotted {mode}/{case} -> {outdir}", flush=True)
            if len(mode_results) > 1:
                plot_case_comparisons(mode_results, args.outdir / mode)
                print(f"Replotted {mode} case comparison -> {args.outdir / mode}", flush=True)
        return
    for case in BUILD_CASES:
        if case in args.case:
            load_optimum(builds[case], allow_stale=args.allow_stale)
    summaries = []
    for mode in dict.fromkeys(args.mode):
        mode_results = []
        for case in dict.fromkeys(args.case):
            print(f"Monte Carlo: {case}, {mode}, {args.samples} draws", flush=True)
            result = run_monte_carlo(case, args.samples, args.seed, builds.get(case),
                                     workers=args.workers, mode=mode,
                                     allow_stale=args.allow_stale)
            outdir = args.outdir / mode / CASE_DIRS[case]
            summary = write_outputs(result, outdir, seed=args.seed, workers=args.workers)
            if not args.no_plot:
                plot_mode(result, outdir)
            print(f"  Verified delivery {summary['feasible_fraction']:.1%}; outputs {outdir}", flush=True)
            mode_results.append(result)
            summaries.append(summary)
        if not args.no_plot and len(mode_results) > 1:
            plot_case_comparisons(mode_results, args.outdir / mode)
    args.outdir.mkdir(parents=True, exist_ok=True)
    # Merge by (mode, case) so a partial run does not drop the other runs' rows.
    path = args.outdir / "summary.json"
    fresh = {(s["mode"], s["case"]): s for s in summaries}
    kept = json.loads(path.read_text()) if path.exists() else []
    merged = [fresh.pop((s["mode"], s["case"]), s) for s in kept] + list(fresh.values())
    _write_json(merged, path)


if __name__ == "__main__":
    main()
