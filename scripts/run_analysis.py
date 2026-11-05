"""Run the core analysis in resumable stages.

python -m scripts.run_analysis --workers 8
python -m scripts.run_analysis --reuse-build          # replay the saved optimum
python -m scripts.run_analysis --reoptimise
python -m scripts.run_analysis --reoptimise --workers 8 --refinement-starts 6 --refinement-budget 16000

Stages, in order: grey reference and blue base cases, their cost pies, the
Typical build, the cost-optimum build, OAT sensitivity and cracker heat
configurations (Typical and optimised builds), then the economic and technical
Monte Carlo. The delivery-margin builds, co-location, capture configurations
and the remaining thesis figures are separate steps; scripts/README.md lists
them in order.

Each stage is skipped when its saved output is still current. --reoptimise
searches again anyway, and everything downstream of the optimised build is then
rebuilt with it: model_fingerprint covers the inputs, not the search, so nothing
else would notice that the build underneath it had changed.

The optimisation runs with builds.optimisation's own defaults (no warm start)
unless --maxiter, --refinement-starts, --refinement-budget or
--refinement-min-improvement override them.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

# The workflow entry point is often launched as a file (IDE run button), which
# puts scripts/ on sys.path instead of the repo root. The stages it spawns use
# -m from the repo root, so they need no shim of their own.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from base_case.run_base_case import GREY_JSON, load_grey
from meoh_pathway.optimisation import load_optimum, model_fingerprint
from scripts.common.outputs import (CASE_DIRS, MC_DIR, OPT_DIR, OUTPUT_DIR, PROPOSED_DIR, TYPICAL_DIR,
                                    design_dir)

MC_MODES = ("economic", "technical")
# Mirrors uncertainty.monte_carlo.PROPOSED_OPTIMUM; the proposed build joins the
# Monte Carlo whenever it has been optimised (builds.optimisation --delivery-margin 0.1).
PROPOSED_OPTIMUM = PROPOSED_DIR / "optimum.json"
MC_CASES = ("grey", "grey_ccs", "optimum") + (("proposed",) if PROPOSED_OPTIMUM.exists() else ())
# Mirrors uncertainty.monte_carlo.CASE_DIRS; importing that module pulls in matplotlib.
MC_CASE_DIRS = {"grey": "lng", "grey_ccs": "lng_ccs", "optimum": "optimised_build",
                "proposed": "proposed_build"}
MC_PLOTS = {"economic": "economic_cost_distribution.png",
            "technical": "delivery_dashboard.png"}
# Mirrors plot_lng_cases' output folders; importing that module pulls in matplotlib.
LNG_CASE_DIRS = (OUTPUT_DIR / "reference_case", OUTPUT_DIR / "base_case")
CASE_PLOTS = ("meoh_flow_diagram.png", "pathway_comparison.png", "cost_breakdown.png")
OPT_PLOTS = CASE_PLOTS + ("site_allocation.png",)


def _current_json(path, loader):
    try:
        loader(path)
    except (FileNotFoundError, OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False
    return True


def _load_typical(path):
    payload = json.loads(Path(path).read_text())
    if payload.get("model_fingerprint") != model_fingerprint():
        raise ValueError("stale Typical build")
    if not payload.get("hourly_feasible", False):
        raise ValueError("Typical build is not feasible")
    return payload


def _current_result(directory, case):
    metadata_path = directory / "run_metadata.json"
    results_path = directory / "results.json"
    if not metadata_path.exists() or not results_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text())
        return (metadata.get("case") == case
                and metadata.get("model_fingerprint") == model_fingerprint())
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _plots_present(directory, names):
    return all((directory / name).exists() for name in names)


def _current_metadata(path, required=()):
    if not path.exists() or not all(p.exists() for p in required):
        return False
    try:
        return json.loads(path.read_text()).get("model_fingerprint") == model_fingerprint()
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _current_sensitivity(directory):
    return _current_metadata(directory / "run_metadata.json",
                             [directory / "endpoints.json", directory / "tornado_cost.png",
                              directory / "tornado_ei.png"])


def _current_heat(directory):
    return _current_metadata(directory / "heat_config_metadata.json",
                             [directory / "heat_config_comparison.json",
                              directory / "heat_config_comparison.png"])


def _current_monte_carlo():
    if not (MC_DIR / "summary.json").exists():
        return False
    for mode in MC_MODES:
        for case in MC_CASES:
            path = MC_DIR / mode / MC_CASE_DIRS[case] / "run_metadata.json"
            try:
                if json.loads(path.read_text()).get("model_fingerprint") != model_fingerprint():
                    return False
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                return False
    return True


def _monte_carlo_plots():
    for mode in MC_MODES:
        if not (MC_DIR / mode / "cost_distribution_comparison.png").exists():
            return False
        for case in MC_CASES:
            if not (MC_DIR / mode / MC_CASE_DIRS[case] / MC_PLOTS[mode]).exists():
                return False
    return True


def _run(module, arguments, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", module, *map(str, arguments)]
    print("\n" + " ".join(command), flush=True)
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(command, cwd=OUTPUT_DIR.parent, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        for line in process.stdout:
            handle.write(line)
            handle.flush()
            print(line, end="", flush=True)
        if process.wait():
            raise SystemExit(f"{module} failed; see {log}")


def _run_if_needed(label, module, arguments, log, current, plots_current=True, replot=None):
    """True when the full stage ran, False when it was skipped or only replotted."""
    if current and plots_current:
        print(f"\nSkipping {label}: current output already exists", flush=True)
        return False
    if current and replot is not None:
        print(f"\n{label}: results are current, regenerating plots only", flush=True)
        _run(module, replot, log)
        return False
    _run(module, arguments, log)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reuse-build", action="store_true",
                        help="Replay the saved optimum instead of searching again; fails "
                             "if it is missing or predates the current model")
    parser.add_argument("--reoptimise", action="store_true",
                        help="Search again even though the saved optimum is current. "
                             "model_fingerprint covers the model, not the search settings, "
                             "so a changed --seed is otherwise skipped over with the "
                             "existing build")
    parser.add_argument("--allow-stale", action="store_true",
                        help="Treat an optimum saved under an older model fingerprint as "
                             "current, and pass --allow-stale to every stage that loads it")
    parser.add_argument("--maxiter", type=int, default=None,
                        help="Passed to builds.optimisation (DE generations); default is its own")
    parser.add_argument("--refinement-starts", type=int, default=None,
                        help="Passed to builds.optimisation; default is its own")
    parser.add_argument("--refinement-budget", type=int, default=None,
                        help="Passed to builds.optimisation (evaluations per start); default is its own")
    parser.add_argument("--refinement-min-improvement", type=float, default=None,
                        help="Passed to builds.optimisation (stop when a round improves by at most "
                             "this); default is its own")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.samples < 2:
        parser.error("Invalid worker or sample count")
    if args.reuse_build and args.reoptimise:
        parser.error("--reuse-build and --reoptimise are opposites")
    plot = not args.no_plot
    no_plot = [] if plot else ["--no-plot"]
    stale = ["--allow-stale"] if args.allow_stale else []
    _run_if_needed("reference and base cases", "base_case.run_base_case", [],
                   OUTPUT_DIR / "reference_case" / "run.log",
                   _current_json(GREY_JSON, load_grey))
    if plot:
        _run_if_needed("LNG case pies", "scripts.plots.lng_cases", [],
                       OUTPUT_DIR / "base_case" / "plots.log",
                       all(_plots_present(d, ("cost_breakdown.png",)) for d in LNG_CASE_DIRS))
    _run_if_needed(
        "typical build", "scripts.builds.typical", ["--plot"] if plot else [],
        TYPICAL_DIR / "run.log",
        (_current_json(TYPICAL_DIR / "typical_build.json", _load_typical)
         and _current_result(TYPICAL_DIR, "typical")),
        plots_current=not plot or _plots_present(TYPICAL_DIR, CASE_PLOTS),
        replot=["--plot", "--replot"],
    )

    optimum_json = OPT_DIR / "optimum.json"
    load_saved = lambda path: load_optimum(path, allow_stale=args.allow_stale)
    replot_args = ["--outdir", OPT_DIR, "--replot", *no_plot, *stale]
    search_args = ["--workers", args.workers, "--seed", args.seed,
                   "--outdir", OPT_DIR, *no_plot]
    for flag, value in (("--maxiter", args.maxiter), ("--refinement-starts", args.refinement_starts),
                        ("--refinement-budget", args.refinement_budget),
                        ("--refinement-min-improvement", args.refinement_min_improvement)):
        if value is not None:
            search_args += [flag, value]
    if args.reuse_build:
        # Fail here rather than let the stage fall through into a fresh search.
        try:
            load_saved(optimum_json)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise SystemExit(f"--reuse-build needs a current {optimum_json}: {exc}") from exc
        search_args = replot_args
    rebuilt = _run_if_needed("optimisation refinement", "scripts.builds.optimisation", search_args,
                             OPT_DIR / "run.log",
                             not args.reoptimise and _current_json(optimum_json, load_saved),
                             plots_current=not plot or _plots_present(OPT_DIR, OPT_PLOTS),
                             replot=replot_args)
    # A fresh optimum invalidates every stage that reads it. The fingerprint they
    # each check cannot see that: the model did not change, the build did.
    rebuilt = rebuilt and not args.reuse_build

    for case, directory in CASE_DIRS.items():
        stale_build = rebuilt and directory == OPT_DIR
        _run_if_needed(
            f"{case} OAT sensitivity", "scripts.uncertainty.sensitivity", ["--case", case, *stale],
            directory / "sensitivity" / "run.log",
            _current_sensitivity(directory / "sensitivity") and not stale_build,
        )
        _run_if_needed(
            f"{case} heat configurations", "scripts.design_analysis.heat_configs",
            ["--case", case, *stale], design_dir(directory) / "heat_configs.log",
            _current_heat(design_dir(directory)) and not stale_build,
        )

    mc_selection = ["--case", *MC_CASES, "--mode", *MC_MODES]
    mc_args = [*mc_selection, "--samples", args.samples, "--seed", args.seed,
               "--workers", args.workers, *no_plot, *stale]
    # The optimum case draws against the saved build, so a re-search redoes all
    # cases -- uncertainty.monte_carlo takes them in one invocation.
    _run_if_needed("Monte Carlo", "scripts.uncertainty.monte_carlo", mc_args, MC_DIR / "run.log",
                   not rebuilt and _current_monte_carlo(),
                   plots_current=not plot or _monte_carlo_plots(),
                   replot=[*mc_selection, "--replot"])


if __name__ == "__main__":
    main()
