"""Solve the Typical build: workbook Typical inputs, capacities balanced by the model.

Prints the grey reference case and the hourly MeOH pathway (flows, routes and
costs) and saves outputs/typical_build/typical_build.json, which the optimiser
reads for its search bounds and starting designs. --plot also saves the flow
diagrams, pathway comparison and cost pie.

Usage:
    python -m scripts.builds.typical [--pathway grey|meoh|both] [--plot] [--json PATH]
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from base_case.run_base_case import run_grey
from base_case.parameters import load_lng
from base_case.reporting import print_base_summary
from meoh_pathway.reporting import print_meoh_summary
from scripts.common.flow_diagrams import plot_base_case_flow, plot_meoh_flow


def main():
    from scripts.common.outputs import TYPICAL_DIR, write_result
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pathway",
        choices=("grey", "meoh", "both"),
        default="both",
        help="Which pathway to run (default both)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Also save flow diagrams (and, for both pathways, the comparison "
        "and cost breakdown charts) to outputs/typical_build/",
    )
    parser.add_argument(
        "--json", type=Path, help="Write complete MeOH results as JSON"
    )
    parser.add_argument("--outdir", type=Path, default=TYPICAL_DIR)
    parser.add_argument("--sensitivity", action="store_true")
    parser.add_argument("--replot", action="store_true",
                        help="Regenerate plots from a saved results.json without resolving "
                             "the MeOH allocation search (the grey side is cheap and reruns either way)")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.replot:
        grey = None
        if args.pathway in ("grey", "both"):
            grey = run_grey()
            lng_params = load_lng()
            print_base_summary(grey, lng_params)
            if args.plot:
                plot_base_case_flow(grey, lng_params, outdir=args.outdir)
        if args.pathway != "grey":
            from scripts.common.outputs import read_result
            results_path = args.outdir / "results.json"
            if not results_path.exists():
                parser.error(f"no saved {results_path}; run without --replot first")
            r = read_result(args.outdir)
            print_meoh_summary(r)
            if args.plot:
                plot_meoh_flow(r, outdir=args.outdir)
                if args.pathway == "both":
                    from scripts.common.pathway_plots import plot_pathway_comparison, plot_cost_breakdown
                    plot_pathway_comparison(grey, r, outdir=args.outdir)
                    plot_cost_breakdown(r, outdir=args.outdir)
        return
    if args.pathway in ("grey", "both"):
        grey = run_grey()
        lng_params = load_lng()
        print_base_summary(grey, lng_params)
        if args.plot:
            plot_base_case_flow(grey, lng_params, outdir=args.outdir)
    if args.pathway == "grey":
        return
    from meoh_pathway.typical import solve_typical
    solved = solve_typical(allow_infeasible=True, save_path=args.outdir / "typical_build.json")
    r = solved.full_result
    print(solved.message)
    print(f"Fleet wind {sum(r.site_wind_mw.values())/1000:.1f} GW, "
          f"solar {sum(r.site_solar_mw.values())/1000:.1f} GW")
    print_meoh_summary(r)
    if not r.hourly_feasible:
        raise SystemExit(2)
    write_result(r, args.outdir, "typical")
    if args.plot:
        plot_meoh_flow(r, outdir=args.outdir)
        if args.pathway == "both":
            from scripts.common.pathway_plots import plot_pathway_comparison, plot_cost_breakdown

            plot_pathway_comparison(grey, r, outdir=args.outdir)
            plot_cost_breakdown(r, outdir=args.outdir)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(asdict(r), indent=2))
    if args.sensitivity:
        from scripts.uncertainty.sensitivity import run_sensitivity_analysis
        run_sensitivity_analysis(case="typical", outdir=args.outdir / "sensitivity")


if __name__ == "__main__":
    main()
