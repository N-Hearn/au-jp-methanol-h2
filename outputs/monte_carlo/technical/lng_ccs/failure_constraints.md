# Binding constraints: grey_ccs, technical mode

0 of 1000 draws failed. A draw counts under every constraint its failure reason names, once per site that constraint bound at, so the columns do not sum to the failed draws. "fleet" marks a fleet-level limit: the cracker and carrier fleets. Full messages are in samples.json (failure_reason).

| Constraint | Draws | Sites | Meaning |
| --- | ---: | --- | --- |
| No hourly schedule after reallocation | 0 | | Every built site passes its annual ceilings, but no split of the target across them could be verified hour by hour. |
| Cracker fleet too small | 0 | | The Japan cracker units cannot crack enough methanol to return the H2 target (fleet-level pre-screen, before any dispatch). |
| Synthesis trains too few | 0 | | A site's installed synthesis trains cannot make its share of methanol at nominal operating hours (pre-screen, before dispatch). |
| Terminal tank below ship parcel | 0 | | Methanol terminal storage at the Australian or Japanese end is smaller than one ship's cargo, so a vessel cannot be loaded or discharged. |
| Hourly dispatch / storage buffering | 0 | | Annual totals are sufficient, but battery and H2 storage cannot bridge the hourly gaps between renewable output and the synthesis schedule. |
| Renewable energy too low | 0 | | A site's wind and solar do not generate enough energy in the year for its share of PEM, compression and synthesis load. |
| Synthesis availability | 0 | | A site's synthesis trains, at the drawn available hours per year, cannot produce its methanol share. |
| Carriers too few | 0 | | The MeOH/LCO2 carrier fleet cannot carry methanol to Japan and the captured CO2 back in the year (voyage time, fuel and boil-off as drawn). Counted fleet-level. |
| PEM nameplate too small | 0 | | A site's installed electrolyser modules cannot produce its H2 share at the drawn specific energy consumption and degradation. |
| H2 trunk pipeline too small | 0 | | The renewables-to-port H2 pipeline cannot move a site's H2 share in the year. |
| Trunk compressor too small | 0 | | The trunk pipeline compressor cannot supply the compression energy a site's H2 flow needs. |
| Storage cycle did not converge | 0 | | A site's dispatch never reached a repeating year-end battery / H2 storage state, so the year could not be verified. |
| Unmet H2 or electricity after dispatch | 0 | | Dispatch ran with no named equipment limit, but left some H2 demand or electricity demand unserved. |
