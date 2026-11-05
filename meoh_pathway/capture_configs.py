"""CO2 capture design options for the Japanese methanol reformer (MSR) plant.

A design comparison in the style of run_heat_configs(): every input is held at
its workbook Typical value except the capture position, rate and energy basis,
the reference deployment (sites, renewables, storage, synthesis trains) is kept,
process fleets are resized, and each build must pass a frozen-equipment replay.

Nothing here changes the workbook, the reference case or any saved output; the
options only override cracker capture fields on a copy of the base ParamSet.

Options
-------
pre_psa_90     Pre-combustion CO2 PSA ahead of the H2 PSA, 90 % (workbook Typical).
pre_psa_95     As above at 95 %, the upper syngas capture reported by Al Ghafri et al.
pre_mea_90     Pre-combustion MEA absorption at 90 % (Lin et al. 2024, intensified design:
               2.4 GJ/t CO2 reboiler heat, about 130 kWh/t CO2 electricity).
post_amine_90  Single H2 PSA, tail gas burned, amine capture of 90 % of the flue CO2
               (workbook config 1 Typical: 2.4 GJ/t reboiler, 100 kWh/t, 80 USD/t).
no_capture     Reformer CO2 vented; all synthesis CO2 supplied as make-up (DAC price),
               i.e. a linear chain like Rezaei et al. S2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .sensitivity import ParamSet, evaluate
from .supply_chain import SupplyChain

CAPTURE_CONFIGS = (
    ("pre_psa_90", "Typical: pre-combustion CO2 PSA ahead of the H2 PSA, 90 % capture (Rezaei/Crandall energy, Al Ghafri cost).",
     {"co2_capture_config": 0, "co2_capture_rate": 0.90}),
    ("pre_psa_95", "Pre-combustion CO2 PSA at 95 % capture (upper syngas capture, Al Ghafri et al.).",
     {"co2_capture_config": 0, "co2_capture_rate": 0.95}),
    ("pre_mea_90", "Pre-combustion MEA absorption at 90 % (Lin et al. 2024 intensified: 2.4 GJ/t reboiler, ~130 kWh/t), same unit cost.",
     {"co2_capture_config": 0, "co2_capture_rate": 0.90,
      "capture_regen_heat_concentrated_gj_per_t_co2": 2.4,
      "capture_electricity_concentrated_kwh_per_t_co2": 130.0}),
    ("post_amine_90", "Single H2 PSA; tail gas burned; flue-gas amine capture of 90 % (workbook config 1 Typical: 2.4 GJ/t, 100 kWh/t, 80 USD/t).",
     {"co2_capture_config": 1, "co2_capture_rate": 0.90}),
    ("no_capture", "No capture at the reformer; all synthesis CO2 bought as non-fossil make-up (linear chain).",
     {"co2_capture_config": 0, "co2_capture_rate": 0.0}),
)


@dataclass
class CaptureConfigResult:
    name: str
    note: str
    delivered_cost_aud_per_kg_h2: float = float("nan")
    delivered_ei_kg_co2_per_kg_h2: float = float("nan")
    net_ei_kg_co2_per_kg_h2: float = float("nan")
    reaction_vent_ei: float = float("nan")
    other_ei: float = float("nan")
    captured_co2_mtpa: float = float("nan")
    makeup_co2_mtpa: float = float("nan")
    capture_electricity_kwh_per_kg_h2: float = float("nan")
    regeneration_electricity_kwh_per_kg_h2: float = float("nan")
    reformer_heat_electricity_kwh_per_kg_h2: float = float("nan")
    circularity_rate: float = float("nan")
    n_ships: int = 0
    cost_components: Dict[str, float] = field(default_factory=dict)
    error: str = ""


def capture_cost_components(r) -> Dict[str, float]:
    """Split delivered cost [AUD/kg H2] into the parts a capture option moves."""
    c = r.cracker
    kg_h2 = c.annual_h2_tpa * 1000
    price = c.power_price_aud_per_kwh
    capture = (c.annual_capture_cost_aud + c.capture_electricity_mwh * 1000 * price) / kg_h2
    regeneration = c.regeneration_electricity_mwh * 1000 * price / kg_h2
    reformer_heat = max(0.0, c.heat_electricity_mwh - c.regeneration_electricity_mwh) * 1000 * price / kg_h2
    cb = r.cost_breakdown_aud_per_kg_h2
    co2_return = cb.get("co2_capture", 0.0)
    makeup = cb.get("co2_makeup", 0.0)
    moved = capture + regeneration + reformer_heat + co2_return + makeup
    return {
        "capture": capture,
        "regeneration": regeneration,
        "reformer_heat": reformer_heat,
        "co2_return": co2_return,
        "makeup": makeup,
        "rest_of_chain": r.delivered_cost_aud_per_kg_h2 - moved,
    }


def run_capture_configs(base: Optional[ParamSet] = None,
                        deployment_overrides: Optional[Dict] = None,
                        build_results: Optional[Dict] = None) -> List[CaptureConfigResult]:
    """Evaluate each capture option on the reference build (process fleets resized)."""
    from .validation import validate_result
    base = base or ParamSet.base()
    reference = evaluate(base, deployment_overrides)
    design = SupplyChain.fixed_deployment(reference)
    design.pop("fixed_design")
    design["site_h2_allocation_weight"] = dict(reference.site_h2_frac)
    design["site_synth_modules"] = dict(reference.site_n_synth)
    out = []
    for name, note, changes in CAPTURE_CONFIGS:
        print(f"  Capture configuration: {name}", flush=True)
        p = base.copy()
        for key, value in changes.items():
            p.set("cracker", key, float(value))
        try:
            r = evaluate(p, dict(design))
            r = evaluate(p, SupplyChain.fixed_deployment(r))
            validate_result(r)
        except (ValueError, ArithmeticError, RuntimeError) as exc:
            out.append(CaptureConfigResult(name, note, error=str(exc)))
            continue
        if build_results is not None:
            build_results[name] = r
        c = r.cracker
        kg_h2 = c.annual_h2_tpa * 1000
        target = r.target_h2_tpa
        out.append(CaptureConfigResult(
            name=name, note=note,
            delivered_cost_aud_per_kg_h2=r.delivered_cost_aud_per_kg_h2,
            delivered_ei_kg_co2_per_kg_h2=r.delivered_ei_kg_co2_per_kg_h2,
            net_ei_kg_co2_per_kg_h2=r.delivered_ei_kg_co2_per_kg_h2 - r.fleet_annual_co2_makeup_tpa / target,
            reaction_vent_ei=r.cracker_vent_ei,
            other_ei=r.delivered_ei_kg_co2_per_kg_h2 - r.cracker_vent_ei,
            captured_co2_mtpa=r.fleet_annual_co2_returned_tpa / 1e6,
            makeup_co2_mtpa=r.fleet_annual_co2_makeup_tpa / 1e6,
            capture_electricity_kwh_per_kg_h2=c.capture_electricity_mwh * 1000 / kg_h2,
            regeneration_electricity_kwh_per_kg_h2=c.regeneration_electricity_mwh * 1000 / kg_h2,
            reformer_heat_electricity_kwh_per_kg_h2=max(0.0, c.heat_electricity_mwh - c.regeneration_electricity_mwh) * 1000 / kg_h2,
            circularity_rate=r.circularity_rate,
            n_ships=r.fleet_ship_total,
            cost_components=capture_cost_components(r),
        ))
    return out
