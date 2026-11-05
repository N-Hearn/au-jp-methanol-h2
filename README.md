# AU-JP methanol H2 supply chain

Australia to Japan hydrogen via a circular methanol loop, compared against grey H2 (LNG + ATR in Japan)
and blue H2 (LNG + ATR + CCS).

## Layout

```text
base_case/      grey (reference case) and blue CCS (base case) LNG + ATR models
meoh_pathway/   methanol supply chain model, optimiser, sensitivity, Monte Carlo
inputs/         input_data.xlsx, loader, renewables.ninja profiles
scripts/        run scripts and plotting; see scripts/README.md for usage
outputs/        results for each case
```

```text
outputs/
  reference_case/           grey H2 (Fig 8)
  base_case/                blue H2 (Fig 9)
  typical_build/            Typical build (starting point and bounds for the optimiser)
  optimised_build/          cost-optimum build (Figs 10-13)
    sensitivity/            OAT tornadoes
    design_analysis/        co-location (Fig 17), heat (Fig 18) and capture (Fig 19) configurations,
                            margin build delivered cost (Fig 20)
    margin_cases/           5/10/20 % delivery-margin builds; the 10 % build is the proposed build
  monte_carlo/
    economic/, technical/   grey, blue, cost-optimum and proposed builds (Figs 14-15)
    margin_<pct>pct/        technical Monte Carlo of each margin build (Fig 16)
```

## Current results

| Case | AUD/kg H2 | USD/kg H2 | kg CO2/kg H2 |
|---|---:|---:|---:|
| Grey (LNG + ATR) | 4.98 | 3.51 | 12.70 |
| Blue (LNG + ATR + CCS) | 6.44 | 4.53 | 3.97 |
| Typical build | 22.18 | 15.61 | 1.31 |
| Optimised build (cost-optimum, 4 sites) | 17.22 | 12.11 | 1.30 |
| Optimised build, 5% delivery margin | 17.80 | 12.52 | 1.30 |
| Optimised build, 10% delivery margin (proposed) | 18.89 | 13.29 | 1.30 |
| Optimised build, 20% delivery margin (6 sites) | 20.44 | 14.38 | 1.30 |

The cost-optimum produces at Port Hedland, Ashburton, Dampier and Gladstone. A
delivery margin (`builds.optimisation --delivery-margin`) requires the build to stay
hourly feasible at (1 + margin) x target and costs it at the contracted target.

Economic Monte Carlo P10/P50/P90 (AUD/kg): cost-optimum 16.2/17.9/19.8, proposed
17.7/19.5/21.5, grey 4.2/4.8/5.2, blue 5.8/6.4/6.9. Technical Monte Carlo: the
cost-optimum delivers the full target in 12% of draws, the 5/10/20% margin
builds in 27/44/46%.

See report for the comparison with published studies and the findings.
