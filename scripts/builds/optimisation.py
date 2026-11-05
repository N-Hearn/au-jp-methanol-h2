"""Run the MeOH supply-chain deployment optimiser.

Usage:
    python -m scripts.builds.optimisation
        # cost-only DE + save + plot
    python -m scripts.builds.optimisation --no-plot
        # DE + save, skip plots
    python -m scripts.builds.optimisation --ei-weight X
        # cost + X * EI
    python -m scripts.builds.optimisation --pareto
        # sweep weights for Pareto
    python -m scripts.builds.optimisation --storage local_only
        # re-optimise with regional H2 storage pinned to zero (also regional_only)
    python -m scripts.builds.optimisation --delivery-margin 0.1
        # require 10% production headroom against technical shortfalls
    python -m scripts.builds.optimisation --outdir outputs/some_run
        # save + plot to specific directory
    python -m scripts.builds.optimisation --replot
        # skip DE, replay saved optimum
    python -m scripts.builds.optimisation --sensitivity
        # skip DE, run the OAT tornadoes around the saved optimum
    python -m scripts.builds.optimisation --replot --sensitivity
        # replot + OAT tornadoes in one go, no DE
    python -m scripts.builds.optimisation --replot --outdir outputs/some_run
        # specific run

Every non-Pareto run writes <outdir>/optimum.json (decision vector + headline
metrics). --replot reads that JSON, re-runs the supply chain once (~seconds)
and regenerates all plots -- fast path for iterating on plot styling.
--sensitivity reads the same JSON and produces overall cost + EI tornadoes
in <outdir>/sensitivity/ (equivalent to `python -m scripts.uncertainty.sensitivity
--around-optimum <outdir>/optimum.json`). Each run also writes
refined_candidates.json, the refined post-DE starts.

The optimiser starts from the calculated Typical balance and searches wind,
solar, battery, H2 storage, production allocation and integer process fleets.
Workbook Min/Max cells bound renewables and storage; physical capacity and
equipment assumptions determine PEM and synthesis fleet ceilings in code.
Technology and market assumptions remain fixed and can be explored with
sensitivity analysis.

--delivery-margin m trades cost for design flexibility. The cheapest build is
sized exactly to the target, so any technical shortfall (higher electrolyser
SEC, faster degradation, lower availability, a weaker resource year) cuts
delivery. With a margin, every candidate must stay hourly feasible at
(1 + m) x target and is costed operating at the contracted target, so the
reported cost includes what the headroom costs. It covers the Australian
production decisions searched here; the cracker and carrier fleets follow the
contracted target. Results go to outputs/optimised_build/margin_cases/
margin_<pct>/ unless --outdir is set; seed from the saved optimum by default.

--storage regional_only|local_only pins the other H2 storage type to zero and
re-optimises the whole build, seeded from the saved mixed optimum when one
exists. Results go to outputs/optimised_build/storage_cases/<case>/ unless
--outdir is set.
"""

import argparse
import json
import math
from pathlib import Path

from base_case.run_base_case import run_grey
from meoh_pathway.optimisation import (
    optimise,
    pareto_sweep,
    print_result,
    save_optimum,
    result_from_saved,
    build_supply_chain,
    DEFAULT_BOUNDS,
    warm_start,
)


from scripts.common.outputs import MARGIN_DIR, OPT_DIR, PROPOSED_DIR, write_result

OPTIMUM_JSON_NAME = "optimum.json"


def storage_bounds(storage):
    """Workbook bounds with the excluded H2 storage type pinned to zero."""
    if storage == "mixed":
        return None
    blocked = "local_h2_storage_kg_" if storage == "regional_only" else "h2_storage_kg_"
    return tuple((name, 0.0, 0.0, integer) if name.startswith(blocked) else (name, lo, hi, integer)
                 for name, lo, hi, integer in DEFAULT_BOUNDS)


# The proposed build is the 10% delivery-margin optimum, as in the Monte Carlo,
# and both builds keep their Monte Carlo colours.
COST_OPTIMUM_COLOUR = "#2ca02c"  # FLOW_COLORS["h2"], the e-MeOH green
PROPOSED_COLOUR = "#01672d"  # COST_COMPONENTS["pem_capital"], the proposed-build green


def _build_label(margin):
    if not margin:
        return "e-MeOH\ncost-optimum build"
    if math.isclose(margin, 0.10):
        return f"e-MeOH proposed build\n(+{margin:.0%} delivery margin)"
    return f"e-MeOH build\n(+{margin:.0%} delivery margin)"


def _comparison_builds(result, search_settings):
    """The e-MeOH bars: the cost-optimum and the proposed build, plus this build if neither."""
    from scripts.common.outputs import read_result

    margin = (search_settings or {}).get("delivery_margin", 0.0)
    builds = []
    for reference_margin, directory, colour in ((0.0, OPT_DIR, COST_OPTIMUM_COLOUR),
                                                (0.10, PROPOSED_DIR, PROPOSED_COLOUR)):
        if math.isclose(margin, reference_margin):
            builds.append((_build_label(margin), result, colour))
        elif (directory / "results.json").exists():
            builds.append((_build_label(reference_margin), read_result(directory), colour))
    if not any(math.isclose(margin, m) for m in (0.0, 0.10)):
        from scripts.common.plot_common import margin_colour
        builds.append((_build_label(margin), result, margin_colour(margin * 100)))
    return builds


def _plot_results(opt_r, outdir: Path) -> None:
    from scripts.plots.site_allocation import plot_site_allocation
    from scripts.plots.dispatch_profiles import plot_optimised_dispatch
    from scripts.common.flow_diagrams import plot_meoh_flow
    from scripts.common.pathway_plots import plot_pathway_comparison, plot_cost_breakdown

    outdir.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating plots -> {outdir}")
    grey = run_grey()
    plot_meoh_flow(opt_r.full_result, outdir=outdir)
    plot_meoh_flow(opt_r.full_result, outdir=outdir, slide=True)
    plot_site_allocation(opt_r.full_result, outpath=outdir / "site_allocation.png")
    chain = build_supply_chain([opt_r.x[n] for n, _, _, _ in DEFAULT_BOUNDS])
    chain.run()
    plot_optimised_dispatch(opt_r.full_result, outdir / "dispatch_profiles", chain.dispatch_traces())
    plot_pathway_comparison(grey, opt_r.full_result, outdir=outdir,
                     builds=_comparison_builds(opt_r.full_result, opt_r.search_settings))
    plot_cost_breakdown(opt_r.full_result, outdir=outdir)
    plot_cost_breakdown(opt_r.full_result, outdir=outdir, slide=True)
    plot_cost_breakdown(opt_r.full_result, outdir=outdir, slide=True, thesis=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ei-weight",
        type=float,
        default=0.0,
        help="USD/kg-H2 weight on EI in objective (0 = cost only)",
    )
    p.add_argument(
        "--pareto",
        action="store_true",
        help="Sweep EI weights to trace cost-EI Pareto front",
    )
    p.add_argument(
        "--maxiter",
        type=int,
        default=120,
        help="DE max generations (default 120). Runtime scales linearly.",
    )
    p.add_argument("--popsize", type=int, default=10)
    p.add_argument("--refinement-budget", type=int, default=8000,
                   help="Maximum objective evaluations per refinement start (default 8000)")
    p.add_argument("--refinement-rounds", type=int, default=40,
                   help="Maximum repeated refinement rounds (default 40); the improvement "
                        "threshold or stage retirement normally stops sooner")
    p.add_argument("--refinement-starts", type=int, default=4,
                   help="Refine this many post-DE designs independently and keep the best "
                        "(default 4); their spread measures how much the answer depends on "
                        "which basin DE ended in")
    p.add_argument("--refinement-min-improvement", type=float, default=0.001,
                   help="Stop when a round improves the objective by at most this amount")
    p.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Processes evaluating each generation (8). Any value "
        "other than 1 switches DE to deferred updating, so the search path "
        "differs from a serial run with the same seed.",
    )
    p.add_argument(
        "--polish",
        action="store_true",
        help="Run L-BFGS-B polish after DE. Off by default: with integer "
        "fleet counts the objective has O(0.05-0.1) discrete steps and "
        "polish can waste thousands of evaluations without improving.",
    )
    p.add_argument("--delivery-margin", type=float, default=0.0,
                   help="Production headroom the build must carry, as a fraction of the "
                        "target (e.g. 0.1 = feasible at 110%% of target); default 0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warm-start", type=Path, help="Seed from a previous vector; re-evaluate under current inputs")
    p.add_argument(
        "--storage",
        choices=("mixed", "regional_only", "local_only"),
        default="mixed",
        help="H2 storage types the search may build (default mixed). Restricted "
        "cases seed from the saved mixed optimum and write to "
        "outputs/optimised_build/storage_cases/<case>/.",
    )
    p.add_argument(
        "--outdir",
        default=None,
        help="Output directory (default: outputs/optimised_build/, or "
        "storage_cases/<case>/ below it for a restricted storage case)",
    )
    p.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating dispatch and allocation plots",
    )
    p.add_argument(
        "--replot",
        action="store_true",
        help="Skip the DE run, reload the saved optimum from "
        "<outdir>/optimum.json, and regenerate plots. Fast path "
        "for iterating on plot styling without re-running the DE.",
    )
    p.add_argument(
        "--sensitivity",
        action="store_true",
        help="Skip the DE run, load <outdir>/optimum.json, and run "
        "the overall cost + EI tornadoes around that deployment. "
        "Writes to <outdir>/sensitivity/. Combine with --replot "
        "to regenerate deployment plots too.",
    )
    p.add_argument("--allow-stale", action="store_true",
                   help="Load an optimum saved under an older model fingerprint "
                        "(--replot / --sensitivity)")
    args = p.parse_args()

    bounds = storage_bounds(args.storage)
    if args.delivery_margin and bounds is not None:
        raise SystemExit("--delivery-margin with a restricted --storage case is not supported")
    outdir = Path(args.outdir) if args.outdir else (
        MARGIN_DIR / f"margin_{args.delivery_margin*100:g}pct" if args.delivery_margin
        else OPT_DIR if bounds is None else OPT_DIR / "storage_cases" / args.storage)
    json_path = outdir / OPTIMUM_JSON_NAME

    if args.replot or args.sensitivity:
        if not json_path.exists():
            raise SystemExit(
                f"No saved optimum at {json_path}. Run the "
                f"optimiser once (without --replot / --sensitivity) "
                f"to create it."
            )
        if args.replot:
            print(f"Reloading saved optimum from {json_path} ...")
            r = result_from_saved(json_path, allow_stale=args.allow_stale)
            write_result(r.full_result, outdir, "optimised" if bounds is None else f"optimised_{args.storage}")
            print_result(r)
            if not args.no_plot:
                _plot_results(r, outdir)
        if args.sensitivity:
            # Delegate to the sensitivity script's callable entry point so both
            # entry points share exactly one implementation.
            from scripts.uncertainty.sensitivity import run_sensitivity_analysis

            print(f"\nRunning sensitivity around {json_path} ...")
            run_sensitivity_analysis(json_path, allow_stale=args.allow_stale)
        return

    x0 = warm_start(args.warm_start) if args.warm_start else None
    if x0 is None and (bounds is not None or args.delivery_margin) and (OPT_DIR / OPTIMUM_JSON_NAME).exists():
        print(f"Seeding the {args.storage if bounds is not None else 'margin'} search from {OPT_DIR / OPTIMUM_JSON_NAME}")
        x0 = warm_start(OPT_DIR / OPTIMUM_JSON_NAME)

    print(
        f"Running optimisation with ei_weight={args.ei_weight}, maxiter={args.maxiter} ..."
    )

    if args.pareto:
        weights = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0]
        print(f"Pareto sweep: ei_weights = {weights}")
        results = pareto_sweep(
            weights,
            bounds=bounds,
            maxiter=args.maxiter,
            popsize=args.popsize,
            seed=args.seed,
            x0=x0,
            progress=True,
            workers=args.workers,
            refinement_budget=args.refinement_budget,
            refinement_rounds=args.refinement_rounds,
            refinement_starts=args.refinement_starts,
            refinement_min_improvement=args.refinement_min_improvement,
            delivery_margin=args.delivery_margin,
        )
        print()
        print(
            f"{'weight':>8}{'cost AUD/kg':>14}{'EI kg/kg':>12}"
            f"{'port':>14}{'solar MW':>10}{'wind MW':>10}{'synth hr':>10}"
        )
        for w, r in zip(weights, results):
            print(
                f"{w:>8.2f}{r.delivered_cost_aud_per_kg_h2:>14.4f}"
                f"{r.delivered_ei_kg_co2_per_kg_h2:>12.4f}"
                f"{r.wind_region:>14}"
                f"{r.solar_capacity_mw:>10.0f}{r.wind_capacity_mw:>10.0f}"
                f"{r.full_result.synth_actual_op_hrs:>10.0f}"
            )
    else:
        r = optimise(
            bounds=bounds,
            ei_weight=args.ei_weight,
            maxiter=args.maxiter,
            popsize=args.popsize,
            seed=args.seed,
            x0=x0,
            progress=True,
            polish=args.polish,
            workers=args.workers,
            refinement_budget=args.refinement_budget,
            refinement_rounds=args.refinement_rounds,
            refinement_starts=args.refinement_starts,
            refinement_min_improvement=args.refinement_min_improvement,
            delivery_margin=args.delivery_margin,
        )
        print_result(r)
        save_optimum(r, json_path)
        if r.refined_starts:
            (outdir / "refined_candidates.json").write_text(
                json.dumps(r.refined_starts, indent=2), encoding="ascii")
            print(f"Refined {len(r.refined_starts)} builds -> "
                  f"{outdir / 'refined_candidates.json'}")
        write_result(r.full_result, outdir, "optimised" if bounds is None else f"optimised_{args.storage}")
        print(f"\nSaved optimum -> {json_path}  (use --replot to regenerate plots)")
        if not args.no_plot:
            _plot_results(r, outdir)


if __name__ == "__main__":
    main()
