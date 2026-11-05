"""Slide-style flowsheets of the three pathways, for the thesis results.

python -m scripts.plots.flow_slides

Redraws flow_diagram_thesis.png for the cost-optimised and proposed
(10 % margin) builds from their saved results.json, and draws matching grey
and blue LNG flowsheets (reference_case and base_case flow_diagram_thesis.png) from run_blue,
which recomputes the LNG benchmarks in seconds. Each is also combined with its
cost pie at thesis width as flow_cost_thesis.png: the LNG pie beside the
flowsheet, the wider MeOH one's below it. Nothing is re-optimised and no
result file is written.
"""

import tempfile
from pathlib import Path

from scripts.common.cli import init_plotting

init_plotting()

from base_case.parameters import load_lng
from base_case.run_base_case import run_blue
from scripts.common.flow_diagrams import plot_lng_flow_slide, plot_meoh_flow
from scripts.common.outputs import OPT_DIR, OUTPUT_DIR, PROPOSED_DIR, read_result
from scripts.common.plot_common import LNG_SLIDE_COST_STAGES, lng_case_costs, plot_flow_cost

def main():
    blue, lng = run_blue(), load_lng()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for outdir in (OPT_DIR, PROPOSED_DIR):
            r = read_result(outdir)
            vented = r.delivered_ei_kg_co2_per_kg_h2 * r.fleet_h2_delivered_tpa / 1e6
            print(f"{outdir.name}: vented {vented:.3f} Mtpa, DAC make-up "
                  f"{r.fleet_annual_co2_makeup_tpa / 1e6:.3f} Mtpa")
            plot_meoh_flow(r, outdir=outdir, slide=True)
            plot_meoh_flow(r, outdir=tmp, slide=True, panel=None)
            costs = {k: v for k, v in r.cost_breakdown_aud_per_kg_h2.items() if k != "Total"}
            plot_flow_cost(tmp / "flow_diagram_thesis.png", costs, outdir / "flow_cost_thesis.png",
                           beside=False)
        for outdir, ccs in ((OUTPUT_DIR / "reference_case", False), (OUTPUT_DIR / "base_case", True)):
            plot_lng_flow_slide(blue, lng, outdir, ccs=ccs)
            plot_lng_flow_slide(blue, lng, tmp, ccs=ccs, panel=None)
            costs = lng_case_costs(blue.grey, lng, blue if ccs else None)
            plot_flow_cost(tmp / "flow_diagram_thesis.png", costs, outdir / "flow_cost_thesis.png",
                           stages=LNG_SLIDE_COST_STAGES)


if __name__ == "__main__":
    main()
