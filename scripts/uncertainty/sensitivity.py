"""One-at-a-time (OAT) sensitivity: tornado charts ranking the most sensitive inputs.

Parameters are sourced from the module Inputs tables in inputs/input_data.xlsx.
Only parameters that reflect genuine engineering or market uncertainty
are bounded; physical constants and model-internal decomposition
parameters are excluded from the sweep.

Usage:
    python -m scripts.uncertainty.sensitivity                # full tornado + breakdowns around Excel base
    python -m scripts.uncertainty.sensitivity --case optimised   # tornadoes around the saved optimum
    python -m scripts.uncertainty.sensitivity --replot       # redraw from saved endpoints.json
    python -m scripts.uncertainty.sensitivity --around-optimum
        # Overall cost + EI tornadoes only, around the saved DE optimum
        # (deployment pinned from outputs/optimised_build/optimum.json).
    python -m scripts.uncertainty.sensitivity --around-optimum outputs/run_ei5/optimum.json
        # Same, from a specific saved optimum.

Typical outputs land in outputs/typical_build/sensitivity/.
Around-optimum outputs land in <optimum_dir>/sensitivity/, e.g.
outputs/optimised_build/sensitivity/, so each run's sensitivity sits next to
the deployment it corresponds to.
"""

import json
from dataclasses import asdict
from pathlib import Path

from scripts.common.cli import init_plotting

init_plotting()

import matplotlib.pyplot as plt
import numpy as np

from meoh_pathway.sensitivity import (
    Bound,
    COST_MODULES,
    EI_MODULES,
    ParamSet,
    SensitivityPoint,
    build_bounds,
    evaluate,
    interaction_index,
    run_oat,
)
from meoh_pathway.monte_carlo import optimum_deployment, DESIGN_FIELDS
from meoh_pathway.supply_chain import SupplyChain
from meoh_pathway.optimisation import model_fingerprint
from meoh_pathway.reporting import print_tornado_table
from scripts.common.outputs import CASE_DIRS, OPT_DIR
from scripts.common.plot_common import save_fig


def _load_deployment_from_optimum(json_path: Path, allow_stale=False):
    """Parse an optimum.json into (base_paramset, deployment_overrides, filtered_bounds).

    Returns:
      base:     Workbook Typical technology and market parameters.
      overrides: dict of SupplyChain kwargs (site_wind_mw, site_solar_mw,
                site_battery_mwh, site_h2_storage_kg, site_local_h2_storage_kg)
                to pin the deployment.
      prune:   design rows excluded from fixed-equipment sensitivities.
    """
    base, overrides = optimum_deployment(json_path, ParamSet.base(), allow_stale=allow_stale)
    return base, overrides, DESIGN_FIELDS


# Pairs with physically meaningful coupling (annotated on the interaction plot)
ANNOTATED_PAIRS = [
    (
        "pem.small_module_mw",
        "pem.operating_hours_per_year",
        "Plant CAPEX vs utilisation - both lower LCOMeOH (substitution)",
    ),
    (
        "cracker.cracker_conversion_rate",
        "cracker.co2_capture_rate",
        "Conversion sets H2 yield; capture sets CO2 return flow (near-independent on cost)",
    ),
    (
        "cracker.cracker_conversion_rate",
        "co2_cap.liquefaction_storage_aud_per_t",
        "Higher conversion -> lower H2 cost basis; liquefaction independent of yield",
    ),
    (
        "ship.carrier_cargo_volume_m3",
        "ship.port_turnaround_days",
        "Bigger tanks and faster turnaround both shrink the fleet (substitutes)",
    ),
]


def tornado_plot(points, metric, title, xlabel, out_path, threshold=1e-4, top_n=10):
    sort_key = (lambda p: p.cost_swing) if metric == "cost" else (lambda p: p.ei_swing)
    # Filter zero-swing parameters (below threshold), then keep the top_n by swing.
    pts = [p for p in points if sort_key(p) > threshold]
    pts = sorted(pts, key=sort_key, reverse=True)[:top_n]
    pts = sorted(pts, key=sort_key)  # ascending for the barh plot

    if not pts:
        print(f"  no non-zero params for {metric}, skipping")
        Path(out_path).unlink(missing_ok=True)
        return

    base = pts[-1].base_cost if metric == "cost" else pts[-1].base_ei
    labels = [p.bound.name for p in pts]

    if metric == "cost":
        lows = np.array([p.low_cost for p in pts]) - base
        highs = np.array([p.high_cost for p in pts]) - base
    else:
        lows = np.array([p.low_ei for p in pts]) - base
        highs = np.array([p.high_ei for p in pts]) - base

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 0.5 * len(labels) + 2.0))

    ax.barh(y, lows, color="#4a7bc8", label="at low bound")
    ax.barh(y, highs, color="#c84a4a", label="at high bound")

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(f"Delta {xlabel} from base ({base:.3f})")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()
    save_fig(fig, out_path, dpi=130)
    print(f"  ({len(pts)} non-zero parameters)")


def print_interaction_analysis(bounds, metric="cost", base=None, deployment_overrides=None):
    """Print interaction indices for the physically annotated pairs, plus
    a ranked table of the top cost-sensitive parameters."""

    bound_by_key = {f"{b.group}.{b.field}": b for b in bounds}

    unit = "AUD/kg H2" if metric == "cost" else "kg CO2/kg H2"
    print(f"\nInteraction analysis ({metric}, {unit})")
    print("  Negative mixed corner difference; interpret direction for each input pair.")
    print("-" * 80)

    for key_a, key_b, note in ANNOTATED_PAIRS:
        if key_a not in bound_by_key or key_b not in bound_by_key:
            continue
        b_a = bound_by_key[key_a]
        b_b = bound_by_key[key_b]
        idx, errors = interaction_index(
            b_a,
            b_b,
            metric=metric,
            base=base,
            deployment_overrides=deployment_overrides,
        )
        label_a = b_a.name[:38]
        label_b = b_b.name[:38]
        if errors:
            print("\n  Interaction unavailable: " + "; ".join(errors))
            continue
        sign = "+" if idx > 0.001 else ("-" if idx < -0.001 else " ~0")
        print(f"\n  {label_a}")
        print(f"  x {label_b}")
        print(f"  index = {idx:+.4f}  [{sign}]  {note}")

    print()


def print_module_breakdown_table(points, metric="cost", threshold=1e-4, top_n=10):
    """Per-parameter table showing how much of each parameter's swing is
    attributable to each MeOH module. The summed module swings won't exactly
    equal the headline swing because module values can move in opposite
    directions; this table is for attribution, not arithmetic reconciliation."""
    if metric == "cost":
        modules = COST_MODULES
        sort_key = lambda p: p.cost_swing
        swing_fn = lambda p, m: p.module_cost_swing(m)
        unit = "AUD/kg H2"
    else:
        modules = EI_MODULES
        sort_key = lambda p: p.ei_swing
        swing_fn = lambda p, m: p.module_ei_swing(m)
        unit = "kg CO2/kg H2"

    pts = sorted(
        [p for p in points if sort_key(p) > threshold], key=sort_key, reverse=True
    )[:top_n]
    if not pts:
        return

    col_w = max(13, max((len(m) for m in modules), default=12) + 2)
    total_w = 42 + col_w * (len(modules) + 1)

    print(f"\nPer-module {metric} swing ({unit})  -- top {len(pts)} parameters")
    print("-" * total_w)
    header = (
        f"{'Parameter':<42}"
        + "".join(f"{m:>{col_w}}" for m in modules)
        + f"{'TOTAL':>{col_w}}"
    )
    print(header)
    print("-" * total_w)
    for p in pts:
        cells = "".join(f"{swing_fn(p, m):>{col_w}.4f}" for m in modules)
        total = sort_key(p)
        print(f"{p.bound.name:<42}{cells}{total:>{col_w}.4f}")


def module_tornado_plot(
    points,
    module,
    metric,
    title,
    xlabel,
    out_path,
    threshold=1e-4,
    group_filter=None,
    top_n=10,
):
    """Tornado of a single module's contribution swing for the bounded parameters.

    group_filter: if set, only include points whose bound.group matches. Used
    for module tornadoes where cross-module bleed (via downstream pricing) is
    not a meaningful engineering lever for that module.
    """
    if metric == "cost":
        getter_low = lambda p: p.low_module_costs.get(module, 0.0)
        getter_high = lambda p: p.high_module_costs.get(module, 0.0)
        getter_base = lambda p: p.base_module_costs.get(module, 0.0)
        sort_key = lambda p: p.module_cost_swing(module)
    else:
        getter_low = lambda p: p.low_module_eis.get(module, 0.0)
        getter_high = lambda p: p.high_module_eis.get(module, 0.0)
        getter_base = lambda p: p.base_module_eis.get(module, 0.0)
        sort_key = lambda p: p.module_ei_swing(module)

    candidates = points
    if group_filter is not None:
        candidates = [p for p in candidates if p.bound.group == group_filter]
    pts = [p for p in candidates if sort_key(p) > threshold]
    pts = sorted(pts, key=sort_key, reverse=True)[:top_n]
    pts = sorted(pts, key=sort_key)  # ascending for the barh plot
    if not pts:
        print(f"  no non-zero params for {module}/{metric}, skipping")
        Path(out_path).unlink(missing_ok=True)
        return

    base = getter_base(pts[-1])
    labels = [p.bound.name for p in pts]
    lows = np.array([getter_low(p) for p in pts]) - base
    highs = np.array([getter_high(p) for p in pts]) - base

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 0.45 * len(labels) + 1.8))
    ax.barh(y, lows, color="#4a7bc8", label="at low bound")
    ax.barh(y, highs, color="#c84a4a", label="at high bound")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(f"Delta {xlabel} from base ({base:.4f})")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()
    save_fig(fig, out_path, dpi=130)
    print(f"  ({len(pts)} non-zero parameters)")


def thesis_tornado_plot(
    points, module, group, title, out_path, top_n=4, fontsize=20, threshold=1e-4
):
    """Thesis-quality cost tornado for a single module, restricted to that
    module's own bounded parameters. Top-N most influential params only."""
    sort_key = lambda p: p.module_cost_swing(module)
    pts = [p for p in points if p.bound.group == group and sort_key(p) > threshold]
    pts = sorted(pts, key=sort_key, reverse=True)[:top_n]
    pts = sorted(pts, key=sort_key)
    if not pts:
        print(f"  no params for {module}, skipping")
        Path(out_path).unlink(missing_ok=True)
        return

    base = pts[-1].base_module_costs.get(module, 0.0)
    labels = [p.bound.name for p in pts]
    lows = np.array([p.low_module_costs.get(module, 0.0) for p in pts]) - base
    highs = np.array([p.high_module_costs.get(module, 0.0) for p in pts]) - base

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(13, 0.9 * len(labels) + 3.5))

    ax.barh(y, lows, color="#4a7bc8", label="Low bound", height=0.55)
    ax.barh(y, highs, color="#c84a4a", label="High bound", height=0.55)
    ax.axvline(0, color="black", linewidth=1.0)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=fontsize)
    ax.tick_params(axis="x", labelsize=fontsize)
    ax.set_xlabel("Change in cost contribution [AUD/kg H2]", fontsize=fontsize)
    ax.set_title(f"{title}   (base = {base:.3f} AUD/kg H2)", fontsize=fontsize, pad=14)
    ax.legend(loc="lower right", fontsize=fontsize, framealpha=0.95)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    save_fig(fig, out_path, dpi=200)
    print(f"  ({len(pts)} params)")


def _load_points(path):
    """Reconstruct SensitivityPoint list from a saved endpoints.json, for --replot."""
    data = json.loads(Path(path).read_text())
    return [SensitivityPoint(bound=Bound(**d["bound"]), **{k: v for k, v in d.items() if k != "bound"})
            for d in data]


def _plot_from_points(points, case, mode_label, outdir, around_optimum):
    """Everything derivable from OAT endpoints alone: tornadoes and module/thesis
    breakdowns. Shared between a fresh run and --replot from saved endpoints.json."""
    failures = [
        f"{p.bound.group}.{p.bound.field}: low={p.low_error or 'ok'}; high={p.high_error or 'ok'}"
        for p in points
        if p.low_error or p.high_error
    ]
    (outdir / "infeasible_endpoints.txt").write_text(
        "\n".join(failures) or "All tested endpoints feasible.\n"
    )
    if failures:
        print(
            f"  {len(failures)} parameters have infeasible endpoints; see infeasible_endpoints.txt. NaN endpoints are not plotted."
        )
    print_tornado_table(points, metric="cost")
    print_tornado_table(points, metric="ei")

    print("\nTornado plots ...")
    tornado_plot(
        points,
        "cost",
        f"{case}: delivered H2 cost sensitivity ({sum(bool(p.low_error)+bool(p.high_error) for p in points)} infeasible endpoints)",
        "delivered cost [AUD/kg H2]",
        outdir / "tornado_cost.png",
    )
    tornado_plot(
        points,
        "ei",
        f"{case}: operational emissions sensitivity ({sum(bool(p.low_error)+bool(p.high_error) for p in points)} infeasible endpoints)",
        "delivered EI [kg CO2/kg H2]",
        outdir / "tornado_ei.png",
    )

    # Around-optimum runs stop here: the point of that mode is "what tech
    # risks matter for THIS build". Module-level analyses are base-case investigations.
    if around_optimum:
        return

    print_module_breakdown_table(points, metric="cost")
    print_module_breakdown_table(points, metric="ei")

    print("\nPer-module tornado plots ...")
    # For shipping, restrict to ship-group bounded parameters: upstream
    # params bleed in via LCOMeOH-priced bunker fuel, but those aren't the
    # shipping levers a designer would change.
    module_group = {"shipping": "ship"}
    for module in COST_MODULES:
        module_tornado_plot(
            points,
            module,
            "cost",
            f"Module-level cost sensitivity: {module}",
            f"{module} cost contribution [AUD/kg H2]",
            outdir / f"tornado_cost_{module}.png",
            group_filter=module_group.get(module),
        )
    for module in EI_MODULES:
        module_tornado_plot(
            points,
            module,
            "ei",
            f"Module-level EI sensitivity: {module}",
            f"{module} EI contribution [kg CO2/kg H2]",
            outdir / f"tornado_ei_{module}.png",
            group_filter=module_group.get(module),
        )

    print("\nThesis tornadoes (PEM, synthesis) ...")
    thesis_tornado_plot(
        points,
        "pem",
        "pem",
        "PEM electrolyser: cost sensitivity",
        outdir / "tornado_pem_thesis.png",
    )
    thesis_tornado_plot(
        points,
        "meoh_synthesis",
        "synth",
        "MeOH synthesis: cost sensitivity",
        outdir / "tornado_synth_thesis.png",
    )


def run_sensitivity_analysis(around_optimum_path: Path = None, case="typical", outdir=None,
                             allow_stale=False, replot=False) -> None:
    """Callable entry point (used by the CLI and builds.optimisation --sensitivity).

    around_optimum_path: if set, run the abbreviated tornadoes-only sensitivity
    around a saved DE optimum. Outputs to <path>.parent/sensitivity/.
    If None, run the full base-case suite (module + thesis tornadoes and the
    interaction analysis) to the case's sensitivity/ folder.

    replot: regenerate plots from this run's saved endpoints.json without
    recomputing the OAT endpoints. Interaction analysis is console-only and
    not saved, so it has no replot path.
    """
    base = None
    overrides = None
    prune = DESIGN_FIELDS
    outdir = Path(outdir) if outdir is not None else CASE_DIRS[case] / "sensitivity"
    mode_label = "Typical build (hourly)"
    around_optimum = False
    if case == "optimised" and around_optimum_path is None:
        around_optimum_path = OPT_DIR / "optimum.json"

    if around_optimum_path is not None:
        json_path = Path(around_optimum_path)
        print(f"Loading optimum deployment from {json_path} ...")
        base, overrides, prune = _load_deployment_from_optimum(json_path, allow_stale=allow_stale)
        # Sensitivity outputs live next to the optimum they were computed from,
        # so each optimisation run has its own sensitivity/ subdir.
        outdir = json_path.parent / "sensitivity"
        mode_label = "saved candidate (hourly)"
        around_optimum = True
        case = "optimised"

    outdir.mkdir(parents=True, exist_ok=True)
    base = base or ParamSet.base()

    if replot:
        points_path = outdir / "endpoints.json"
        if points_path.exists():
            _plot_from_points(_load_points(points_path), case, mode_label, outdir, around_optimum)
        else:
            print(f"  no saved endpoints.json at {points_path}, skipping tornado replot")
        return

    reference = evaluate(base, overrides)
    overrides = SupplyChain.fixed_deployment(reference)
    reference = evaluate(base, overrides)
    (outdir / "run_metadata.json").write_text(json.dumps(dict(
        model_fingerprint=model_fingerprint(), fixed_installed_equipment=True,
        case=case, model_basis=reference.model_basis,
        basis="OAT engineering and scenario stress bounds, not statistical confidence"), indent=2))
    bounds = build_bounds()
    if prune:
        bounds = [b for b in bounds if (b.group, b.field) not in prune]
    print(f"Sensitivity mode  : {mode_label}")
    print(f"Bounded parameters: {len(bounds)}")

    print("\nOAT sensitivity ...")
    points = run_oat(bounds, base=base, deployment_overrides=overrides, base_result=reference)
    (outdir / "endpoints.json").write_text(json.dumps([asdict(p) for p in points], indent=2), encoding="ascii")
    _plot_from_points(points, case, mode_label, outdir, around_optimum)

    # Around-optimum runs stop here: the point of that mode is "what tech
    # risks matter for THIS build". Interaction analysis is a base-case investigation.
    if around_optimum:
        print(f"\nDone. Overall tornadoes -> {outdir}")
        return

    print("\nInteraction analysis ...")
    print_interaction_analysis(bounds, metric="cost", base=base, deployment_overrides=overrides)


def main():
    from scripts.common.cli import build_parser

    p = build_parser(__doc__, plot=False, replot=True)
    p.add_argument(
        "--around-optimum",
        nargs="?",
        const=str(OPT_DIR / "optimum.json"),
        default=None,
        metavar="JSON_PATH",
        help="Run sensitivity around a saved DE optimum instead of "
        "the Excel base case. Optional path; defaults to "
        "outputs/optimised_build/optimum.json. Output goes to "
        "<optimum_dir>/sensitivity/.",
    )
    p.add_argument("--case", choices=(*CASE_DIRS, "all"), default="typical")
    p.add_argument("--allow-stale", action="store_true",
                   help="Load an optimum saved under an older model fingerprint")
    args = p.parse_args()
    path = Path(args.around_optimum) if args.around_optimum else None
    for case in CASE_DIRS if args.case == "all" else (args.case,):
        run_sensitivity_analysis(path if case == "optimised" or args.case != "all" else None,
                                 case=case, allow_stale=args.allow_stale, replot=args.replot)


if __name__ == "__main__":
    main()
