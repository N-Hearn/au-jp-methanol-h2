# Scripts

Run everything from the repo root as a module: `python -m scripts.<folder>.<name>`.
Every script takes `--help`. Scripts with `--replot` redraw their figures from saved
outputs in seconds, without re-running the model.

| Folder | What it holds |
|---|---|
| `run_analysis.py` | The core pipeline in resumable stages; skips any stage whose output is current. |
| `builds/` | Make the e-MeOH builds: the Typical build and the optimised builds. |
| `design_analysis/` | Alternatives around the optimised build: co-location, heat and capture configurations. |
| `uncertainty/` | OAT sensitivity (tornadoes) and the economic/technical Monte Carlo. |
| `plots/` | Thesis figures drawn from saved results only. |
| `common/` | Shared helpers: CLI, output paths, plot styling, flow diagrams, pathway charts. |

## Reproducing the thesis results

Run the steps in order; later steps read earlier outputs.

| Step | Command | Writes | Thesis |
|---|---|---|---|
| 1. Core pipeline | `python -m scripts.run_analysis --workers 8` | grey/blue cases, Typical build, cost-optimum, OAT and heat configurations, Monte Carlo | Figs 8-10, 14-15, 18 |
| 2. Margin builds | `python -m scripts.builds.optimisation --delivery-margin 0.05` (and `0.1`, `0.2`) | `optimised_build/margin_cases/margin_<pct>pct/` | Fig 20 |
| 3. Proposed build Monte Carlo | `python -m scripts.uncertainty.monte_carlo --case all --workers 8` | adds `proposed_build` to `monte_carlo/` | Figs 14-15 |
| 4. Margin Monte Carlo | `python -m scripts.uncertainty.monte_carlo --margin 5 --mode technical --workers 8` (and `10`, `20`) | `monte_carlo/margin_<pct>pct/` | Fig 16 |
| 5. Co-location | `python -m scripts.design_analysis.colocation --workers 8` | `optimised_build/design_analysis/colocation/` | Fig 17 |
| 6. Capture configurations | `python -m scripts.design_analysis.capture_configs` | `optimised_build/design_analysis/capture_config_*` | Fig 19 |
| 7. Figures | `python -m scripts.plots.flow_slides`, `plots.site_allocation --replot outputs/optimised_build`, `plots.stage_breakdown`, `plots.margin_cases`, `plots.thesis_figures` | flow + cost figures, site allocation, stage breakdown, margin figures | Figs 8-13, 16, 20 |

Step 1 runs the Monte Carlo on the proposed build too once step 2 has made it, so on a
fresh checkout run step 1, then 2, then 1 again (or step 3).

## Script reference

`builds/`
- `typical.py` solves the Typical build (`outputs/typical_build/`). The optimiser reads
  `typical_build.json` for its bounds and starting designs.
- `optimisation.py` runs the deployment search (DE + refinement). `--delivery-margin m`
  makes a margin build, `--replot` replays a saved `optimum.json`, `--sensitivity` adds
  OAT tornadoes around it.

`design_analysis/`, outputs in `<build>/design_analysis/`
- `colocation.py`: cost and emissions of alternative production footprints.
- `heat_configs.py`: cracker heat supply options (`--case typical|optimised`).
- `capture_configs.py`: CO2 capture options at the Japanese cracker.

`uncertainty/`
- `sensitivity.py`: one-at-a-time tornadoes (`--case typical|optimised|all`).
- `monte_carlo.py`: `--mode economic|technical`, `--case grey grey_ccs optimum proposed|all`,
  `--margin <pct>` for a margin build.

`plots/`
- `lng_cases.py`: grey and blue cost pies.
- `flow_slides.py`: thesis flowsheets with their cost pies.
- `site_allocation.py`, `dispatch_profiles.py`: per-site capacities and dispatch.
- `stage_breakdown.py`: cost by supply-chain stage across the pathways.
- `margin_cases.py`: delivered cost and technical reliability of the margin builds.
- `thesis_figures.py`: redraws every pie, comparison, allocation, dispatch and heat
  figure from saved results.
