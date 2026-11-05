"""Redraw the thesis-sized figures from saved outputs, without re-optimising:
every delivered-cost pie, pathway comparison, site allocation chart and
optimised build's dispatch profiles and every cracker heat configuration
comparison.

python -m scripts.plots.thesis_figures

Each e-MeOH figure is redrawn from the results.json beside it; an optimised
build's comparison also reads its optimum.json for the delivery margin it was
searched at. The grey and blue LNG benchmarks are recomputed (run_blue takes
seconds).
"""

import json

from scripts.common.cli import init_plotting

init_plotting()

from base_case.run_base_case import run_grey
from meoh_pathway.optimisation import DEFAULT_BOUNDS, build_supply_chain, result_from_saved
from scripts.common.outputs import OUTPUT_DIR, read_result
from scripts.plots.dispatch_profiles import plot_optimised_dispatch
from scripts.plots.lng_cases import plot_lng_cases
from scripts.plots.site_allocation import plot_site_allocation
from scripts.builds.optimisation import OPTIMUM_JSON_NAME, _comparison_builds
from scripts.common.pathway_plots import plot_pathway_comparison, plot_cost_breakdown
from scripts.design_analysis.heat_configs import _load_heat_configs, heat_config_plot


def _saved(root, figure):
    """Directories under root holding figure beside a results.json to redraw it
    from: results.json also sits in directories (Monte Carlo) that never draw it."""
    return [png.parent for png in sorted(root.rglob(figure))
            if (png.parent / "results.json").exists()]


def plot_thesis_figures(root=OUTPUT_DIR):
    grey = run_grey()
    for outdir in _saved(root, "cost_breakdown.png"):
        plot_cost_breakdown(read_result(outdir), outdir=outdir)
    for outdir in _saved(root, "cost_breakdown_thesis.png"):
        plot_cost_breakdown(read_result(outdir), outdir=outdir, slide=True, thesis=True)
    for outdir in _saved(root, "pathway_comparison.png"):
        result = read_result(outdir)
        builds = None
        if (outdir / OPTIMUM_JSON_NAME).exists():
            settings = json.loads((outdir / OPTIMUM_JSON_NAME).read_text()).get("search_settings")
            builds = _comparison_builds(result, settings)
        plot_pathway_comparison(grey, result, outdir=outdir, builds=builds)
    for outdir in _saved(root, "site_allocation.png"):
        plot_site_allocation(read_result(outdir), outpath=outdir / "site_allocation.png")
    # The hourly traces are not saved, so an optimised build's chain is re-run
    # from its optimum.json (seconds per build).
    for optimum in sorted(root.rglob(OPTIMUM_JSON_NAME)):
        if (optimum.parent / "dispatch_profiles").is_dir():
            opt_r = result_from_saved(optimum)
            chain = build_supply_chain([opt_r.x[n] for n, _, _, _ in DEFAULT_BOUNDS])
            chain.run()
            plot_optimised_dispatch(opt_r.full_result, optimum.parent / "dispatch_profiles",
                                    chain.dispatch_traces())
    for saved in sorted(root.rglob("heat_config_comparison.json")):
        heat_config_plot(_load_heat_configs(saved), None, saved.with_suffix(".png"))
    plot_lng_cases()


if __name__ == "__main__":
    plot_thesis_figures()
