# Hourly port renewable profiles

Renewables.ninja MERRA-2 hourly capacity factors for 2025 at 1 MW nameplate, UTC.
`meoh_pathway.renewables_dispatch.PORT_PROFILE_COLUMNS` maps each of the ten ports
to a wind and a solar column; `load_wind_profile()` / `load_solar_profile()` read them.
PortDarwin and PortAbbotPoint have no wind profile and their wind maxima are zero.

Solar: fixed tilt 35 deg, azimuth 180 deg (north-facing for southern latitudes in the
Renewables.ninja convention -- do not flip it), 10% system loss, no tracking.
Wind: Vestas V90 2000 at 80 m hub height. These are resource proxies, not validated
layouts for large future developments. Generation and mean CF come from the hourly
arrays, not a scalar CF input.

## Files

- `wind_profile_hourly_ports.csv`, `solar_profile_hourly_ports.csv` -- `hour` (0-8759)
  index, one column per site, fractional CF (`pd.read_csv(path, index_col=0)`).
- `port_renewable_sites.csv` -- per column: ports served, project, status, lat/lon,
  mean CF and raw file.
- `raw/` -- original Renewables.ninja downloads (kW at 1 MW). The PortHedland and
  Newcastle raw files are missing; their columns exist in the wide CSVs only.

## Site coverage

| column_id | resource | port(s) | project | mean CF |
|---|---|---|---|---|
| PortHastings_StarOfTheSouth | wind | PortHastings | Star of the South (offshore) | 0.358 |
| PortHastings_Gippsland | solar | PortHastings | Gippsland Renewable Energy Park | 0.177 |
| PortBonython_Nonowie | wind | PortBonython | Nonowie Wind Farm | 0.401 |
| PortBonython_Cultana | solar | PortBonython | Cultana Solar Farm | 0.203 |
| Gladstone_BoulderCreek | wind | Gladstone | Boulder Creek (Q6 Fitzroy REZ) | 0.298 |
| Gladstone_Aldoga | solar | Gladstone | Aldoga Solar Farm | 0.200 |
| HayPoint_Collinsville | wind | HayPoint | Collinsville Green Energy Hub (Q10 REZ) | 0.247 |
| AbbotPoint_Collinsville | solar | PortAbbotPoint, HayPoint | Collinsville solar cluster | 0.196 |
| Ashburton_AshburtonNorthSIA | wind | Ashburton | Ashburton North SIA | 0.313 |
| Ashburton_AshburtonNorthSIA | solar | Ashburton | Ashburton North SIA | 0.215 |
| Dampier_Woodside | wind | Dampier | Woodside/Ngarluma renewables | 0.315 |
| Dampier_RioTintoNgarluma | solar | Dampier | Rio Tinto/Ngarluma solar | 0.211 |
| PortDarwin_DarwinREH | solar | PortDarwin | Darwin Renewable Energy Hub | 0.173 |
| PortHedland_AREH | wind | PortHedland | Australian Renewable Energy Hub | 0.376 |
| PortHedland_NorthStarJunction | solar | PortHedland | North Star Junction Solar Farm | 0.212 |
| Newcastle_LiverpoolRange | wind | Newcastle | Liverpool Range Wind Farm | 0.294 |
| Newcastle_Liddell | solar | Newcastle | Liddell renewable precinct | 0.185 |
| Gladstone_ClarkeCreek | wind | not mapped | Clarke Creek (superseded Q4 Isaac proxy) | 0.295 |

Sharing a profile (Collinsville solar) means a shared resource proxy, not the same
physical capacity counted twice.

## Limitations

One reanalysis year does not establish multi-year reliability. Site capacity maxima,
storage geology and pipeline distances are separate workbook assumptions. Re-run
dispatch and optimisation after changing profiles; saved-result fingerprints prevent
accidental reuse.

## Licence and citation

The profiles in this folder (raw downloads and the derived hourly files) come from
[Renewables.ninja](https://www.renewables.ninja) and are licensed under
[Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/).
They are provided as-is, without warranty, and may only be reused for
non-commercial purposes with credit. Please cite:

1. S. Pfenninger and I. Staffell (2016). Long-term patterns of European PV output using 30 years of
   validated hourly reanalysis and satellite data. *Energy* 114, 1251-1265.
   doi: [10.1016/j.energy.2016.08.060](https://doi.org/10.1016/j.energy.2016.08.060)
2. I. Staffell and S. Pfenninger (2016). Using bias-corrected reanalysis to simulate current and future
   wind power output. *Energy* 114, 1224-1239.
   doi: [10.1016/j.energy.2016.08.068](https://doi.org/10.1016/j.energy.2016.08.068)
