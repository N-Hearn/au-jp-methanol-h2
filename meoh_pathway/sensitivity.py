"""Sensitivity analysis for the MeOH pathway.

One-at-a-time perturbation of each literature-bounded parameter, with
all others held at base. Records delivered cost (AUD/kg H2) and
delivered emissions intensity (kg CO2/kg H2) at low/base/high values,
plus the per-module cost and EI breakdowns so each parameter's effect
can be attributed to a specific stage of the chain.

Failed endpoints are reported as NaN with their failure reason. Cracker heat
configurations (methanol-fired, H2-fired or electric) are compared by
run_heat_configs() with equipment resized for each; there is no zero-emission
recapture preset.

Bounds and their literature sources are read exclusively from
the module Inputs tables in inputs/input_data.xlsx. The Group and
Field columns map each row to the correct Python
dataclass and field. To change a bound, edit the Excel; there are no
hardcoded ranges here.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from pathlib import Path
from typing import Dict, List, Optional

from inputs.loader import (
    load_input_rows,
    MEOH_SHEETS,
    SHEET_TO_PARAMSET_ATTR,
    _split_field,
    numeric_bounds,
    load_site_inputs,
)

from .parameters import (
    Economics,
    PemParameters,
    MeohSynthParameters,
    ShipParameters,
    CrackerParameters,
    Co2CaptureParameters,
    DacParameters,
    TargetParameters,
    H2TransportParameters,
    load_h2_transport,
    default_target,
    load_economics,
    load_pem,
    load_synth,
    load_ship,
    load_cracker,
    load_co2_capture,
    load_dac,
    load_routes,
)
from .supply_chain import SupplyChain, SupplyChainResult
from .meoh_cracking import net_h2_per_t_feed, supplemental_meoh_per_t_feed


XLSX_PATH = Path(__file__).resolve().parent.parent / "inputs" / "input_data.xlsx"


@dataclass(frozen=True)
class Bound:
    name: str
    group: str
    field: str
    base: float
    low: float
    high: float


@dataclass
class ParamSet:
    """Mutable bundle of every dataclass the MeOH chain reads."""

    pem: PemParameters
    synth: MeohSynthParameters
    ship: ShipParameters
    cracker: CrackerParameters
    co2_cap: Co2CaptureParameters
    dac: DacParameters
    econ: Economics
    target: TargetParameters
    transport: H2TransportParameters
    site_storage_costs: Dict[str, float] = field(default_factory=dict)
    # Drawable route lengths keyed by the workbook's '<Port>.<field>'.
    site_routes: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def base(cls) -> "ParamSet":
        return cls(
            pem=load_pem(),
            synth=load_synth(),
            ship=load_ship(),
            cracker=load_cracker(),
            co2_cap=load_co2_capture(),
            dac=load_dac(),
            econ=load_economics(),
            target=default_target(),
            transport=load_h2_transport(),
            site_storage_costs={p: float(v["h2_storage_capex_aud_per_kg"]["Value"])
                                for p, v in load_site_inputs().items()},
            site_routes={f"{p}.{f}": v[f] for p, v in load_routes().items()
                         for f in ("renewables_to_port_km", "pipeline_distance_km")},
        )

    def copy(self) -> "ParamSet":
        return ParamSet(
            pem=replace(self.pem),
            synth=replace(self.synth),
            ship=replace(self.ship),
            cracker=replace(self.cracker),
            co2_cap=replace(self.co2_cap),
            dac=replace(self.dac),
            econ=replace(self.econ),
            target=replace(self.target),
            transport=replace(self.transport),
            site_storage_costs=dict(self.site_storage_costs),
            site_routes=dict(self.site_routes),
        )

    def set(self, group: str, field: str, value: float) -> None:
        obj = getattr(self, group)
        if isinstance(obj, dict):
            if field not in obj:
                raise ValueError(f"Unknown parameter {group}.{field}")
            obj[field] = value
            return
        if not hasattr(obj, field):
            raise ValueError(f"Unknown parameter {group}.{field}")
        setattr(obj, field, value)


def load_bounds_from_excel(
    path: Path = XLSX_PATH,
    sheet=MEOH_SHEETS,
    base=None,
) -> List[Bound]:
    """Read sensitivity bounds from the Excel workbook.

    Accepts one module sheet or a tuple. Column headers identify the bounds,
    units and Python keys; reference sections are excluded by the table range.
    The base value is read from the model defaults. base takes another bundle
    with the same group attributes, so the base-case sheets can be read too.
    """
    base_ps = ParamSet.base() if base is None else base
    bounds = []
    seen: set = set()

    for sheet_name, row in load_input_rows(path, sheet):
        site, field = _split_field(row.get("Field"))
        if site is not None and field == "h2_storage_capex_aud_per_kg":
            if row.get("Lower bound") is not None or row.get("Upper bound") is not None:
                typical, lo, hi = numeric_bounds(row, f"{site}.{field}")
                if hi > lo:
                    bounds.append(Bound(f"{site} regional storage CAPEX [AUD/kg]",
                        "site_storage_costs", site, typical, lo, hi))
            continue
        if site is not None or not field:
            continue
        group = SHEET_TO_PARAMSET_ATTR.get(sheet_name)
        if group is None or not hasattr(base_ps, group):
            continue  # sheet not modelled on ParamSet (base-case sheets etc.)
        obj = getattr(base_ps, group)
        if not hasattr(obj, field):
            continue  # deployment metadata is consumed outside ParamSet
        if row.get("Lower bound") is None and row.get("Upper bound") is None:
            continue
        _, lo_f, hi_f = numeric_bounds(row, f"{group}.{field}")
        if lo_f == hi_f:
            continue
        if (group, field) in {
            ("pem", "operating_hours_per_year"),
            ("pem", "ramp_rate_percent_per_s"),
            ("pem", "hybrid_solar_share_percent"),
            ("pem", "h2_storage_capex_aud_per_kg"),
            ("econ", "power_price_aud_per_kwh"),
            ("ship", "shipping_distance_nm"),
        }:
            continue  # standalone/annual-only inputs are not integrated sensitivities
        key = (group, field)
        if key in seen:
            raise ValueError(f"Duplicate sensitivity field {group}.{field}")
        seen.add(key)
        base_val = float(getattr(obj, field))
        variable, units = row.get("Variable"), row.get("Units")
        label = f"{variable} [{units}]" if units else variable

        bounds.append(
            Bound(
                name=label,
                group=group,
                field=field,
                base=base_val,
                low=lo_f,
                high=hi_f,
            )
        )

    return bounds


def build_bounds() -> List[Bound]:
    return load_bounds_from_excel()


def evaluate(
    params: ParamSet,
    deployment_overrides: Optional[Dict] = None,
    allow_infeasible: bool = False,
    allow_capacity_overrides: bool = False,
) -> SupplyChainResult:
    """Run one SupplyChain evaluation.

    deployment_overrides: optional dict of SupplyChain kwargs
    (site_wind_mw, site_solar_mw, site_battery_mwh) to pin the physical
    deployment while sweeping technology/market parameters. Used by the
    "around optimum" sensitivity mode. When None, uses the Excel-default
    Typical portfolio. allow_infeasible returns a short
    deployment's result, with its unmet supply, instead of raising."""
    kwargs = dict(deployment_overrides) if deployment_overrides else {}
    return SupplyChain(
        economics=params.econ,
        pem=params.pem,
        synth=params.synth,
        ship=params.ship,
        cracker=params.cracker,
        co2_capture=params.co2_cap,
        dac=params.dac,
        target=params.target,
        h2_transport=params.transport,
        site_storage_costs=params.site_storage_costs,
        site_routes=params.site_routes,
        allow_capacity_overrides=allow_capacity_overrides,
        **kwargs,
    ).run(allow_infeasible=allow_infeasible)


COST_MODULES = (
    "pem",
    "meoh_synthesis",
    "shipping",
    "meoh_cracking",
    "co2_capture",
)
EI_MODULES = COST_MODULES


def _module_costs(r: SupplyChainResult) -> Dict[str, float]:
    """Per-module AUD/kg-H2 contributions (sum to delivered cost).

    LCOMeOH is unpacked into its PEM (H2 feed) and synth-plant shares; the
    CO2 feed share of LCOMeOH (returned CO2 plus DAC make-up) is grouped
    with co2_capture.
    """
    lcomeoh_aud = r.lcomeoh_aud_per_kg_meoh or 1.0  # guard against zero
    lco = r.lcomeoh_component_aud_per_kg_h2
    pem_share = lco * (r.h2_feed_aud_per_kg_meoh / lcomeoh_aud)
    synth_share = lco * (r.synth_plant_aud_per_kg_meoh / lcomeoh_aud)
    co2_feed_share = lco * (r.co2_feed_aud_per_kg_meoh / lcomeoh_aud)
    return {
        "pem": pem_share,
        "meoh_synthesis": synth_share,
        "shipping": r.ship_component_aud_per_kg_h2,
        "meoh_cracking": r.cracker_component_aud_per_kg_h2,
        "co2_capture": co2_feed_share,
    }


def _module_eis(r: SupplyChainResult) -> Dict[str, float]:
    """Per-module kg-CO2/kg-H2 contributions (sum to delivered EI).

    Purchased Australian electricity is attributed to its consuming process.
    """
    return {
        "pem": r.pem_electricity_ei,
        "meoh_synthesis": r.synth_electricity_ei + r.synth_process_ei,
        "shipping": r.ship_fuel_ei + r.ship_fossil_fuel_ei + r.meoh_loss_ei,
        "meoh_cracking": r.cracker_vent_ei
        + r.cracker_electricity_ei
        + r.cracker_heat_ei,
        "co2_capture": r.co2_capture_ei,
    }


@dataclass
class SensitivityPoint:
    bound: Bound
    low_cost: float
    base_cost: float
    high_cost: float
    low_ei: float
    base_ei: float
    high_ei: float
    # Per-module breakdowns at low/base/high (keyed by module name).
    low_module_costs: Dict[str, float] = field(default_factory=dict)
    base_module_costs: Dict[str, float] = field(default_factory=dict)
    high_module_costs: Dict[str, float] = field(default_factory=dict)
    low_module_eis: Dict[str, float] = field(default_factory=dict)
    base_module_eis: Dict[str, float] = field(default_factory=dict)
    high_module_eis: Dict[str, float] = field(default_factory=dict)

    low_error: str = ""
    high_error: str = ""

    @property
    def cost_swing(self) -> float:
        return abs(self.high_cost - self.low_cost)

    @property
    def ei_swing(self) -> float:
        return abs(self.high_ei - self.low_ei)

    def module_cost_swing(self, module: str) -> float:
        return abs(
            self.high_module_costs.get(module, 0.0)
            - self.low_module_costs.get(module, 0.0)
        )

    def module_ei_swing(self, module: str) -> float:
        return abs(
            self.high_module_eis.get(module, 0.0) - self.low_module_eis.get(module, 0.0)
        )


def run_oat(
    bounds: List[Bound],
    base: Optional[ParamSet] = None,
    deployment_overrides: Optional[Dict] = None,
    base_result: Optional[SupplyChainResult] = None,
) -> List[SensitivityPoint]:
    """One-at-a-time sensitivity sweep across low/base/high for each bound.

    base: optional pre-configured ParamSet (used by "around optimum" mode
    to pin deployment-linked fields like cracker.operating_hours_per_year).
    deployment_overrides: passed through to evaluate() so every point uses
    the same physical deployment (only the tech/market param varies)."""
    if base is None:
        base = ParamSet.base()
    base_res = base_result or evaluate(base, deployment_overrides)
    deployment_overrides = SupplyChain.fixed_deployment(base_res)
    base_costs = _module_costs(base_res)
    base_eis = _module_eis(base_res)
    points = []

    for b in bounds:
        lo = base.copy()
        lo.set(b.group, b.field, b.low)
        hi = base.copy()
        hi.set(b.group, b.field, b.high)

        low_error = high_error = ""
        try:
            lo_res = evaluate(lo, deployment_overrides)
        except (ValueError, ArithmeticError, RuntimeError) as exc:
            low_error = str(exc)
            lo_res = None

        try:
            hi_res = evaluate(hi, deployment_overrides)
        except (ValueError, ArithmeticError, RuntimeError) as exc:
            high_error = str(exc)
            hi_res = None

        points.append(
            SensitivityPoint(
                bound=b,
                low_error=low_error,
                high_error=high_error,
                low_cost=lo_res.delivered_cost_aud_per_kg_h2
                if lo_res
                else float("nan"),
                base_cost=base_res.delivered_cost_aud_per_kg_h2,
                high_cost=hi_res.delivered_cost_aud_per_kg_h2
                if hi_res
                else float("nan"),
                low_ei=lo_res.delivered_ei_kg_co2_per_kg_h2 if lo_res else float("nan"),
                base_ei=base_res.delivered_ei_kg_co2_per_kg_h2,
                high_ei=hi_res.delivered_ei_kg_co2_per_kg_h2
                if hi_res
                else float("nan"),
                low_module_costs=_module_costs(lo_res)
                if lo_res
                else {k: float("nan") for k in COST_MODULES},
                base_module_costs=base_costs,
                high_module_costs=_module_costs(hi_res)
                if hi_res
                else {k: float("nan") for k in COST_MODULES},
                low_module_eis=_module_eis(lo_res)
                if lo_res
                else {k: float("nan") for k in COST_MODULES},
                base_module_eis=base_eis,
                high_module_eis=_module_eis(hi_res)
                if hi_res
                else {k: float("nan") for k in COST_MODULES},
            )
        )

    return points


def sweep_1d(group: str, field: str, values, base: ParamSet = None,
             deployment_overrides=None):
    """Return cost, EI and failure reasons for a sweep around a fixed build."""
    base = base or ParamSet.base()
    deployment = SupplyChain.fixed_deployment(evaluate(base, deployment_overrides))
    costs, eis, errors = [], [], []
    for value in values:
        p = base.copy()
        p.set(group, field, float(value))
        try:
            r = evaluate(p, deployment)
            costs.append(r.delivered_cost_aud_per_kg_h2)
            eis.append(r.delivered_ei_kg_co2_per_kg_h2)
            errors.append("")
        except (ValueError, ArithmeticError, RuntimeError) as exc:
            costs.append(float("nan"))
            eis.append(float("nan"))
            errors.append(str(exc))
    return costs, eis, errors


def interaction_index(
    bound_a: Bound,
    bound_b: Bound,
    metric: str = "cost",
    base: Optional[ParamSet] = None,
    deployment_overrides: Optional[Dict] = None,
) -> tuple:
    """Second-order interaction index and endpoint failure reasons.

    This is the negative mixed corner difference. Its sign depends on which
    input directions are beneficial; it is not a universal synergy measure.
    """
    if base is None:
        base = ParamSet.base()

    if not deployment_overrides or "fixed_design" not in deployment_overrides:
        deployment_overrides = SupplyChain.fixed_deployment(evaluate(base, deployment_overrides))
    errors = []

    def f(pa, va, pb, vb):
        p = base.copy()
        p.set(pa.group, pa.field, float(va))
        p.set(pb.group, pb.field, float(vb))
        try:
            r = evaluate(p, deployment_overrides)
        except (ValueError, ArithmeticError, RuntimeError) as exc:
            errors.append(f"{pa.field}={va}, {pb.field}={vb}: {exc}")
            return float("nan")
        return (
            r.delivered_cost_aud_per_kg_h2
            if metric == "cost"
            else r.delivered_ei_kg_co2_per_kg_h2
        )

    f_ll = f(bound_a, bound_a.low, bound_b, bound_b.low)
    f_hh = f(bound_a, bound_a.high, bound_b, bound_b.high)
    f_lh = f(bound_a, bound_a.low, bound_b, bound_b.high)
    f_hl = f(bound_a, bound_a.high, bound_b, bound_b.low)

    return f_hl + f_lh - f_hh - f_ll, errors


@dataclass
class HeatConfigResult:
    """Delivered cost and EI for one cracker heat configuration, EI split by origin."""

    name: str
    note: str
    delivered_cost_aud_per_kg_h2: float = float("nan")
    delivered_ei_kg_co2_per_kg_h2: float = float("nan")
    reaction_vent_ei: float = float("nan")
    fired_heat_ei: float = float("nan")
    electric_heat_ei: float = float("nan")
    parasitic_power_ei: float = float("nan")
    other_ei: float = float("nan")
    heat_meoh_mtpa: float = float("nan")
    heat_h2_kg_per_kg_h2: float = float("nan")
    heat_electricity_kwh_per_kg_h2: float = float("nan")
    power_price_aud_per_kwh: float = float("nan")
    parasitic_kwh_per_kg_h2: float = float("nan")
    heat_pump_heat_share: float = float("nan")
    heat_pump_capacity_mw_th: float = float("nan")
    heat_pump_capex_aud: float = float("nan")
    capture_power_ei: float = float("nan")
    liquefaction_power_ei: float = float("nan")
    capture_electricity_kwh_per_kg_h2: float = float("nan")
    liquefaction_electricity_kwh_per_kg_h2: float = float("nan")
    capture_cost_aud_per_kg_h2: float = float("nan")
    regeneration_electricity_kwh_per_kg_h2: float = float("nan")
    n_cracker: int = 0
    cracker_gross_capacity_tpd_h2: float = float("nan")
    heat_pump_source_limited: bool = False
    tail_gas_heat_capped: bool = False
    production_scale: float = 1.0
    capacity_scale: float = 1.0
    meoh_production_mtpa: float = float("nan")
    upstream_h2_mtpa: float = float("nan")
    gross_cracker_h2_mtpa: float = float("nan")
    delivered_h2_mtpa: float = float("nan")
    upstream_h2_scale: float = float("nan")
    wind_gw: float = float("nan")
    solar_gw: float = float("nan")
    battery_gwh: float = float("nan")
    h2_storage_kt: float = float("nan")
    capacity_bounds_exceeded: List[str] = field(default_factory=list)
    deployment: Dict = field(default_factory=dict)
    site_n_pem: Dict[str, int] = field(default_factory=dict)
    site_n_synth: Dict[str, int] = field(default_factory=dict)
    cost_components: Dict[str, float] = field(default_factory=dict)
    error: str = ""


def heat_config_cost_components(r):
    """Allocate existing costs without charging internal fuel a second time."""
    cracker = r.cracker
    kg_h2 = cracker.annual_h2_tpa * 1000
    price = cracker.power_price_aud_per_kwh
    methanol_fuel = r.cracker_heat_production_aud_per_kg_h2
    h2_fuel_share = r.fleet_h2_fuel_tpa / (r.target_h2_tpa + r.fleet_h2_fuel_tpa)
    hydrogen_fuel = (r.lcomeoh_component_aud_per_kg_h2 - methanol_fuel) * h2_fuel_share
    electric_heat = cracker.heat_electricity_mwh * 1000 * price / kg_h2
    process_power = (cracker.annual_electricity_mwh - cracker.heat_electricity_mwh
                     - cracker.capture_electricity_mwh) * 1000 * price / kg_h2
    capture = (cracker.annual_capture_cost_aud
               + cracker.capture_electricity_mwh * 1000 * price) / kg_h2
    other_cracker = r.cracker_component_aud_per_kg_h2 - electric_heat - process_power - capture
    return {
        "methanol_heat": methanol_fuel,
        "hydrogen_heat": hydrogen_fuel,
        "electric_heat": electric_heat,
        "process_power": process_power,
        "capture": capture,
        "other_cracker": other_cracker,
        "rest_of_chain": r.delivered_cost_aud_per_kg_h2 - r.cracker_component_aud_per_kg_h2
                         - methanol_fuel - hydrogen_fuel,
    }


# Every configuration sets both shares so the workbook switches cannot leak in.
METHANOL_HEAT = {"electric_heat_share": 0.0, "h2_heat_share": 0.0, "heat_pump_enabled": 0.0, "cracker_power_source": 0}
H2_HEAT = {"electric_heat_share": 0.0, "h2_heat_share": 1.0, "heat_pump_enabled": 0.0, "cracker_power_source": 0}
ELECTRIC_HEAT = {"electric_heat_share": 1.0, "h2_heat_share": 0.0, "heat_pump_enabled": 0.0, "cracker_power_source": 0}
HEAT_CONFIGS = (
    (
        "electric_zero_carbon_hp",
        "Typical: zero-carbon Japan power with heat pump for eligible steam/preheat duty.",
        {**ELECTRIC_HEAT, "cracker_power_source": 1, "heat_pump_enabled": 1},
    ),
    (
        "electric_zero_carbon",
        "Zero-carbon Japan power with resistance heating only.",
        {**ELECTRIC_HEAT, "cracker_power_source": 1},
    ),
    (
        "methanol_fired",
        "Net external heat from internal methanol; Japan grid for process and capture power.",
        METHANOL_HEAT,
    ),
    (
        "h2_fired",
        "Net reformer heat from burning product hydrogen; the extra methanol cracked meets the capture split.",
        H2_HEAT,
    ),
    (
        "electric_grid_jp",
        "All net reformer heat electric, at the workbook Japan grid emission factor.",
        ELECTRIC_HEAT,
    ),
    (
        "electric_grid_jp_hp",
        "Current Japan grid for all Japanese loads, with eligible heat-pump duty.",
        {**ELECTRIC_HEAT, "heat_pump_enabled": 1},
    ),
    (
        "electric_grid_jp_2030",
        "Electric heat; heat and parasitic power at the grid emission factor Min (2030 target).",
        {**ELECTRIC_HEAT, "cracker_grid_ei_kg_co2_per_kwh": "low"},
    ),
    (
        "electric_grid_jp_2030_hp",
        "2030 Japan grid EI with eligible heat-pump duty; current tariff retained.",
        {**ELECTRIC_HEAT, "cracker_grid_ei_kg_co2_per_kwh": "low", "heat_pump_enabled": 1},
    ),
)


def run_heat_configs(
    base: Optional[ParamSet] = None,
    deployment_overrides: Optional[Dict] = None,
    find_fired_feasible: bool = False,
    build_results: Optional[Dict[str, SupplyChainResult]] = None,
) -> List[HeatConfigResult]:
    """Compare cracker heat supply options with every other input at base.

    deployment_overrides is the design deployment (workbook Typical when None).
    When find_fired_feasible is true, the methanol- and H2-fired cases scale
    the supplied production portfolio in proportion to their extra methanol
    feed requirement before the frozen replay check.
    Equipment is resized for each option. Allocation weights and additional
    synthesis trains retain the verified reference values. Renewable/storage
    capacities retain the reference values except for explicit fired-case scaling.
    Each resized build must pass a frozen-equipment replay before reporting.
    """
    base = base or ParamSet.base()
    reference = evaluate(base, deployment_overrides)
    design = SupplyChain.fixed_deployment(reference)
    design.pop("fixed_design")
    design["site_h2_allocation_weight"] = dict(reference.site_h2_frac)
    # synth_modules is an absolute installed count (like pem_modules), so the
    # verified reference's site_n_synth is the value directly.
    design["site_synth_modules"] = dict(reference.site_n_synth)
    from .sizing import fleet_bounds
    from .validation import validate_result
    site_inputs = load_site_inputs()
    limits = {b.field: b for b in build_bounds() if b.group == "cracker"}
    out = []
    for name, note, changes in HEAT_CONFIGS:
        print(f"  Heat configuration: {name}", flush=True)
        p = base.copy()
        for key, value in changes.items():
            if value in ("low", "high"):
                value = getattr(limits[key], value)
            p.set("cracker", key, float(value))
        case_design = dict(design)
        scale = 1.0
        try:
            if find_fired_feasible and name in ("methanol_fired", "h2_fired"):
                base_feed = 1 + supplemental_meoh_per_t_feed(base.cracker)
                fired_feed = 1 + supplemental_meoh_per_t_feed(p.cracker)
                scale = (net_h2_per_t_feed(base.cracker) / net_h2_per_t_feed(p.cracker)
                         * fired_feed / base_feed)
                base_design = design
                for _ in range(16):
                    trial_design = dict(base_design)
                    for field in ("site_wind_mw", "site_solar_mw", "site_battery_mwh",
                                  "site_h2_storage_kg", "site_local_h2_storage_kg"):
                        trial_design[field] = {
                            port: value * scale
                            for port, value in base_design.get(field, {}).items()
                        }
                    # synth_modules is an absolute installed count, so it does
                    # not follow the higher fuel-driven throughput on its own:
                    # scale it like the capacities above.
                    trial_design["site_synth_modules"] = {
                        port: math.ceil(value * scale)
                        for port, value in base_design.get("site_synth_modules", {}).items()
                    }
                    r = evaluate(p, trial_design, allow_infeasible=True,
                                 allow_capacity_overrides=True)
                    if r.hourly_feasible:
                        case_design = trial_design
                        break
                    scale *= 1.15
                else:
                    raise ValueError("Proportional heat build remained hourly infeasible")
                print(f"  Renewable/storage capacity scale: {scale:.4f}", flush=True)
            else:
                r = evaluate(p, case_design)
            pem_limits = {port: values["pem_modules"] for port, values in
                          fleet_bounds(deployment=case_design, pem=p.pem, synth=p.synth).items()}
            for port, count in r.site_n_pem.items():
                lo, hi = pem_limits[port]
                if not lo <= count <= hi:
                    raise ValueError(f"{port}: resized PEM count {count} outside physical fleet bounds [{lo:g}, {hi:g}]")
            r = evaluate(p, SupplyChain.fixed_deployment(r),
                         allow_capacity_overrides=scale != 1.0)
            validate_result(r)
        except (ValueError, ArithmeticError, RuntimeError) as exc:
            out.append(HeatConfigResult(name, note, error=str(exc)))
            continue
        kg_h2 = r.cracker.annual_h2_tpa * 1000
        heat_kwh = r.cracker.heat_electricity_mwh * 1000 / kg_h2
        electric_heat_ei = heat_kwh * p.cracker.power_ei_kg_co2_per_kwh
        capture_kwh = r.cracker.capture_electricity_mwh*1000/kg_h2
        capture_ei = capture_kwh*p.cracker.power_ei_kg_co2_per_kwh
        liquid_ei = sum(s.liquefaction_electricity_co2_tpa for s in r.site_co2_capture_results.values())/r.target_h2_tpa
        cracker_ei = r.cracker_vent_ei + r.cracker_heat_ei + r.cracker_electricity_ei
        if build_results is not None:
            build_results[name] = r
        out.append(
            HeatConfigResult(
                name=name,
                note=note,
                delivered_cost_aud_per_kg_h2=r.delivered_cost_aud_per_kg_h2,
                cost_components=heat_config_cost_components(r),
                delivered_ei_kg_co2_per_kg_h2=r.delivered_ei_kg_co2_per_kg_h2,
                reaction_vent_ei=r.cracker_vent_ei,
                fired_heat_ei=r.cracker_heat_ei,
                electric_heat_ei=electric_heat_ei,
                parasitic_power_ei=r.cracker_electricity_ei - electric_heat_ei - capture_ei,
                capture_power_ei=capture_ei,
                liquefaction_power_ei=liquid_ei,
                other_ei=r.delivered_ei_kg_co2_per_kg_h2 - cracker_ei - liquid_ei,
                heat_meoh_mtpa=r.fleet_heat_meoh_tpa / 1e6,
                heat_h2_kg_per_kg_h2=r.fleet_h2_fuel_tpa / r.target_h2_tpa,
                heat_electricity_kwh_per_kg_h2=heat_kwh,
                power_price_aud_per_kwh=p.cracker.power_price_aud_per_kwh,
                parasitic_kwh_per_kg_h2=(r.cracker.annual_electricity_mwh
                    - r.cracker.heat_electricity_mwh-r.cracker.capture_electricity_mwh) * 1000 / kg_h2,
                capture_electricity_kwh_per_kg_h2=capture_kwh,
                liquefaction_electricity_kwh_per_kg_h2=sum(s.liquefaction_electricity_mwh
                    for s in r.site_co2_capture_results.values())/r.target_h2_tpa,
                capture_cost_aud_per_kg_h2=r.cracker.annual_capture_cost_aud/kg_h2,
                regeneration_electricity_kwh_per_kg_h2=r.cracker.regeneration_electricity_mwh*1000/kg_h2,
                n_cracker=r.n_cracker,
                cracker_gross_capacity_tpd_h2=r.cracker.gross_capacity_tpd_h2,
                heat_pump_source_limited=r.cracker.heat_pump_source_limited,
                tail_gas_heat_capped=r.cracker.tail_gas_heat_capped,
                production_scale=r.fleet_annual_meoh_tpa / reference.fleet_annual_meoh_tpa,
                capacity_scale=scale,
                meoh_production_mtpa=r.fleet_annual_meoh_tpa / 1e6,
                upstream_h2_mtpa=r.fleet_pem_h2_tpa / 1e6,
                gross_cracker_h2_mtpa=(r.target_h2_tpa + r.fleet_h2_fuel_tpa) / 1e6,
                delivered_h2_mtpa=r.fleet_h2_delivered_tpa / 1e6,
                upstream_h2_scale=r.fleet_pem_h2_tpa / reference.fleet_pem_h2_tpa,
                wind_gw=sum(r.site_wind_mw.values()) / 1000,
                solar_gw=sum(r.site_solar_mw.values()) / 1000,
                battery_gwh=sum(r.site_battery_mwh.values()) / 1000,
                h2_storage_kt=r.h2_storage_capacity_kg / 1e6,
                capacity_bounds_exceeded=[
                    f"{port}.{field}"
                    for field in ("wind_mw", "solar_mw", "battery_mwh", "h2_storage_kg", "local_h2_storage_kg")
                    for port, value in case_design.get(f"site_{field}", {}).items()
                    if value > float(site_inputs[port][field]["Upper bound"]) + 1e-6
                ],
                deployment=SupplyChain.fixed_deployment(r),
                heat_pump_heat_share=(r.cracker.heat_pump_heat_mwh /
                    (r.cracker.resistance_heat_mwh+r.cracker.heat_pump_heat_mwh)
                    if r.cracker.resistance_heat_mwh+r.cracker.heat_pump_heat_mwh else 0.0),
                heat_pump_capacity_mw_th=r.cracker.heat_pump_capacity_mw_th,
                heat_pump_capex_aud=r.cracker.heat_pump_capex_aud,
                site_n_pem=dict(r.site_n_pem),
                site_n_synth=dict(r.site_n_synth),
            )
        )
    return out
