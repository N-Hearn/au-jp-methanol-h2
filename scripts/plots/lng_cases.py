"""Cost breakdown pies for the two LNG benchmarks.

python -m scripts.plots.lng_cases

Grey H2 (LNG, ATR without capture) is the reference case; blue H2 (the same chain
plus capture, CO2 return and storage in Australia) is the base case.
"""

from scripts.common.cli import init_plotting

init_plotting()

from base_case.parameters import load_lng
from base_case.run_base_case import run_blue
from scripts.common.outputs import OUTPUT_DIR
from scripts.common.plot_common import (BASE_CASE, LNG_SLIDE_COST_STAGES, REFERENCE_CASE, lng_case_costs,
                                  plot_cost_pie, plot_cost_pie_slide)

REFERENCE_DIR = OUTPUT_DIR / "reference_case"
BASE_DIR = OUTPUT_DIR / "base_case"


def plot_lng_cases():
    blue = run_blue()
    lng = load_lng()
    for (name, role, _), costs, outdir in (
        (REFERENCE_CASE, lng_case_costs(blue.grey, lng), REFERENCE_DIR),
        (BASE_CASE, lng_case_costs(blue.grey, lng, blue), BASE_DIR),
    ):
        outdir.mkdir(parents=True, exist_ok=True)
        plot_cost_pie(costs, f"{name}, {role}: delivered cost breakdown", outdir / "cost_breakdown.png")
        plot_cost_pie_slide(costs, outdir / "cost_breakdown_thesis.png",
                            stages=LNG_SLIDE_COST_STAGES, thesis=True)


if __name__ == "__main__":
    plot_lng_cases()
