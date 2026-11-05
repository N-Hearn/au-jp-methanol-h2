"""Economic and technical uncertainty around verified frozen builds.

Economic draws reuse the nominal operating schedule and inherit its feasibility.
Technical draws hold costs central and redispatch the as-built fleet: every
equipment count (PEM modules, synthesis trains, cracker units, ships, CO2
vessels) and the renewable/storage footprint stay frozen, so the mode measures
robustness -- whether the build as sized still delivers. Only the continuous
ratings (transmission, trunk/regional flow, pipeline compressor, terminal
storage) re-rate and recost to what a draw needs. Unsuccessful draws are
retained. Named LHS streams put each input at the same quantile in every mode
and case.
Probability models and dependencies are explicit workbook metadata; sensitivity
bounds alone never opt an input into sampling.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import List, Optional
import hashlib

import numpy as np
import pandas as pd

from base_case.blue_h2 import BlueH2
from base_case.delivered_h2 import DeliveredH2
from base_case.parameters import (
    AtrCcsParameters,
    AtrPlantParameters,
    CarrierParameters,
    Co2LiquefactionParameters,
    DeliveredH2Target,
    DiscountParameters,
    JapanPlantParameters,
    LngSupplyParameters,
    default_target as grey_target,
    load_atr,
    load_atr_ccs,
    load_carrier,
    load_co2_liquefaction,
    load_discount,
    load_japan_plant,
    load_lng,
)

from .optimisation import (
    DEFAULT_BOUNDS,
    load_optimum,
    deployment_from_vector,
    model_fingerprint,
    require_current_inputs,
)
from .sensitivity import Bound, ParamSet, build_bounds, load_bounds_from_excel
from .supply_chain import SupplyChain
from .meoh_synthesis import synth_feed_ratios
from .uncertainty import (
    GREY_SHARED_FIELDS, LNG_CASES, MODES, SHARED_SHEETS, role_fields, uncertainty_bounds,
)
from .validation import validate_result, validate_traces

CASES = ("grey", "grey_ccs", "optimum", "proposed")
# Saved hourly builds: the cost-optimum, and the proposed build (by default the
# 10% delivery-margin optimum) drawn the same way from its own optimum.json.
BUILD_CASES = ("optimum", "proposed")
GREY_SHEETS = ("lng_supply", "atr_plant")
PERCENTILES = (5, 10, 50, 90, 95)
COST = "cost_aud_per_kg_h2"
EI = "ei_kg_co2_per_kg_h2"
UNMET_H2 = "unmet_h2_fraction"
UNMET_ELECTRICITY = "unmet_electricity_fraction"
SHORTFALL = "Fixed build falls short of the H2 target"
ANNUAL_CAPACITY = "annual capacity insufficient"
DISPATCH_FAILURE = "no feasible dispatch found"
NUMERICAL_ERROR = "numerical error"
# The frozen wind/solar cannot generate the year's load; no equipment count binds.
ENERGY_LIMITED = "renewable energy insufficient"

# Workbook Role design: choices about the plant (train sizes, the LNG train that
# sets the common H2 target, configuration switches), never sampled.
# Workbook Role scenario: a different configuration or assumed market rather
# than uncertainty around the selected plant; workbook ranges stay for OAT.
DESIGN_FIELDS = role_fields("design")
SCENARIO_FIELDS = role_fields("scenario")


@dataclass
class McResult:
    case: str
    bounds: List[Bound]
    samples: pd.DataFrame  # one row per draw: inputs, cost, EI, shortfall, error
    reference_cost: float  # deterministic run at workbook Typical values
    reference_ei: float
    mode: str = "economic"
    fingerprint: str = ""
    reference_equipment: dict = field(default_factory=dict)
    # The design point: the saved build's own performance at workbook Typical
    # inputs, before any draw -- where the build sits in each distribution.
    reference_metrics: dict = field(default_factory=dict)

    @property
    def inputs(self) -> List[str]:
        return [f"{b.group}.{b.field}" for b in self.bounds]

    @property
    def feasible(self) -> pd.DataFrame:
        return self.samples[self.samples["error"] == ""]

    @property
    def short(self) -> pd.DataFrame:
        return self.samples[self.samples["error"] == SHORTFALL]


@dataclass
class GreyParams:
    """The grey dataclasses in the bundle shape the draw loop expects."""

    lng: LngSupplyParameters
    atr: AtrPlantParameters
    target: DeliveredH2Target
    ccs: AtrCcsParameters  # read by grey_ccs only
    # Rows shared with the methanol loop, named by the ParamSet groups their sheets route to.
    cracker: JapanPlantParameters
    co2_cap: Co2LiquefactionParameters  # grey_ccs only, as are ship and econ
    ship: CarrierParameters
    econ: DiscountParameters

    @classmethod
    def base(cls) -> "GreyParams":
        return cls(lng=load_lng(), atr=load_atr(), target=grey_target(), ccs=load_atr_ccs(),
                   cracker=load_japan_plant(), co2_cap=load_co2_liquefaction(),
                   ship=load_carrier(), econ=load_discount())

    def copy(self) -> "GreyParams":
        # target is deliberately not re-derived from the drawn capacity and yield:
        # the workbook H2 demand is the common basis both pathways are sized to.
        return GreyParams(replace(self.lng), replace(self.atr), replace(self.target), replace(self.ccs),
                          replace(self.cracker), replace(self.co2_cap), replace(self.ship),
                          replace(self.econ))

    def set(self, group: str, field: str, value: float) -> None:
        obj = getattr(self, group)
        if not hasattr(obj, field):
            raise ValueError(f"Unknown parameter {group}.{field}")
        setattr(obj, field, value)


def grey_bounds() -> List[Bound]:
    """Delivered LNG price, feed gas consumption and upstream leakage.

    Train sizes are design choices. ATR cost ex feed and plant-gate EI carry no
    workbook bounds, so the calibrated literature means stay fixed.
    """
    shared = [b for b in load_bounds_from_excel(sheet=SHARED_SHEETS, base=GreyParams.base())
              if (b.group, b.field) in GREY_SHARED_FIELDS]
    return [b for b in load_bounds_from_excel(sheet=GREY_SHEETS, base=GreyParams.base()) + shared
            if (b.group, b.field) not in DESIGN_FIELDS | SCENARIO_FIELDS]


def meoh_bounds() -> List[Bound]:
    return [b for b in build_bounds()
            if (b.group, b.field) not in DESIGN_FIELDS | SCENARIO_FIELDS]


def _case_bounds(bounds, reference):
    selected = []
    for bound in bounds:
        group, field = bound.group, bound.field
        if group == "site_storage_costs" and reference.site_regional_h2_storage_kg.get(field, 0) == 0:
            continue
        if group == "site_routes":
            port, _, route = field.partition(".")
            if reference.site_h2_frac.get(port, 0) == 0 or (route == "pipeline_distance_km"
                    and reference.site_regional_h2_storage_kg.get(port, 0) == 0):
                continue
        selected.append(bound)
    return selected


def optimum_deployment(path: Path, base: ParamSet, allow_stale=False):
    """Decode the saved physical design using the optimiser's own mapping."""
    payload = load_optimum(path, allow_stale=allow_stale)
    names = [name for name, _, _, _ in DEFAULT_BOUNDS]
    if payload.get("bounds_names") != names:
        raise ValueError(
            f"{path} was saved with different decision variables. Re-run the optimiser."
        )
    return base, deployment_from_vector(payload["x"])


def _setup(case: str, optimum_path: Optional[Path], mode="technical", allow_stale=False):
    if case in LNG_CASES:
        def run(p):
            if case == "grey_ccs":
                return BlueH2(p.lng, p.atr, p.ccs, p.target, p.cracker, p.co2_cap,
                              p.ship, p.econ).run()
            return DeliveredH2(p.lng, p.atr, p.target, p.cracker).run()

        return uncertainty_bounds(case, mode), GreyParams.base(), GreyParams.copy, run

    if optimum_path is None:
        raise ValueError(f"The {case} case needs a saved optimum.json")
    base, overrides = optimum_deployment(optimum_path, ParamSet.base(), allow_stale=allow_stale)

    def chain(p, deployment, resize=False, overrides_allowed=False):
        return SupplyChain(economics=p.econ, pem=p.pem, synth=p.synth, ship=p.ship,
                           cracker=p.cracker, co2_capture=p.co2_cap, dac=p.dac, target=p.target,
                           h2_transport=p.transport, site_storage_costs=p.site_storage_costs,
                           site_routes=p.site_routes, allow_capacity_overrides=overrides_allowed,
                           resize_capacities=resize, **(deployment or {}))

    reference = chain(base, overrides).run()
    validate_result(reference)
    overrides = SupplyChain.fixed_deployment(reference)
    frozen = chain(base, overrides)
    reference = frozen.run()
    validate_result(reference)
    validate_traces(reference, frozen.dispatch_traces())
    h2_ratio = synth_feed_ratios(base.synth)[0]
    schedules = {port: args[0][3]/h2_ratio/1000 for port, args in frozen._trace_inputs.items()}
    demands = {port: args[0][3] for port, args in frozen._trace_inputs.items()}

    def run(p):
        if p is base:
            return reference
        # Technical mode keeps every frozen equipment count (fixed_design intact)
        # and only re-rates the continuous ratings to what the draw needs.
        candidate = chain(p, overrides, resize=mode == "technical")
        if mode == "economic":
            candidate._fixed_operation = dict(result=reference, schedules=schedules, demands=demands)
            candidate._fixed_dispatch_cache = dict(frozen._fixed_dispatch_cache)
        result = candidate.run(allow_infeasible=True)
        feasible = result.hourly_feasible
        if feasible:
            validate_result(result)
            if mode != "economic":
                traces = candidate.dispatch_traces()
                validate_traces(result, traces)
                result.uncertainty_storage_metrics = {
                    f"{port}.h2_storage_working_fraction":
                        float(np.ptp(trace["soc_h2_kg"]))/result.site_h2_storage_kg[port]
                        if result.site_h2_storage_kg[port] else 0.0
                    for port, trace in traces.items()}
        if mode == "economic":
            if not feasible or result.site_h2_frac != reference.site_h2_frac:
                raise ValueError("Economic recost changed verified operation")
            for field in ("fleet_total_electricity_kwh_per_year", "fleet_annual_meoh_tpa",
                          "delivered_ei_kg_co2_per_kg_h2"):
                if not np.isclose(getattr(result, field), getattr(reference, field), rtol=1e-10):
                    raise ValueError(f"Economic recost changed {field}")
        return result

    return _case_bounds(uncertainty_bounds(case, mode), reference), base, ParamSet.copy, run


def triangular_lhs(bounds: List[Bound], n: int, rng: np.random.Generator) -> np.ndarray:
    """Named Latin hypercube streams mapped through explicit marginal CDFs.

    Rows with one shared dependency read the same stream, so they sit at the
    same quantile of their own ranges (one route allowance, one build-year
    CAPEX trajectory) rather than needing identical bounds.
    """
    draws = np.empty((n, len(bounds)))
    seed = int(rng.integers(0, 2**63))
    for j, b in enumerate(bounds):
        # Named streams keep common inputs paired when cases use different rows.
        key = getattr(b, "dependency", "") or f"{b.group}.{b.field}"
        name_seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")
        stream = np.random.default_rng(np.random.SeedSequence([seed, name_seed]))
        u = (stream.permutation(n) + stream.random(n)) / n
        lo, hi = b.low, b.high
        if not all(np.isfinite(v) for v in (lo, b.base, hi)) or not lo <= b.base <= hi or lo >= hi:
            raise ValueError(f"{b.group}.{b.field}: invalid triangular bounds")
        mode = b.base
        if getattr(b, "distribution", "triangular") == "uniform":
            if not np.isclose((lo+hi)/2, mode):
                raise ValueError(f"{b.group}.{b.field}: uniform range must centre on nominal")
            draws[:, j] = lo+u*(hi-lo)
            continue
        split = (mode - lo) / (hi - lo)
        draws[:, j] = np.where(
            u < split,
            lo + np.sqrt(u * (hi - lo) * (mode - lo)),
            hi - np.sqrt((1 - u) * (hi - lo) * (hi - mode)),
        )
    return draws


def _draw_metrics(row, case, bounds, base, copier, run):
    params = copier(base)
    for b, value in zip(bounds, row):
        params.set(b.group, b.field, float(value))
    achievable = None  # fraction of target a hard pre-screen could still deliver
    try:
        r = run(params)
        feasible = case in LNG_CASES or r.hourly_feasible
        error = "" if feasible else SHORTFALL
        category = "" if feasible else (ANNUAL_CAPACITY if r.equipment_shortfalls else DISPATCH_FAILURE)
        reason = "; ".join(r.equipment_shortfalls) if error else ""
        if error and not reason:
            reason = "; ".join(f"{port}: {short['limit']} ({short['unmet_h2_kg']:.6g} kg H2 unmet)"
                               for port, short in r.site_shortfalls.items() if "limit" in short)
        if error and not reason:
            reason = f"Unmet H2 {r.unmet_h2_kg:.6g} kg; unmet electricity {r.unmet_demand_mwh:.6g} MWh"
        if error and _energy_limited(reason):
            category = ENERGY_LIMITED
        if not error and not np.isfinite([r.delivered_cost_aud_per_kg_h2, r.delivered_ei_kg_co2_per_kg_h2]).all():
            raise ValueError("Non-finite model result")
    except (ValueError, ArithmeticError, RuntimeError) as exc:
        r, reason = None, str(exc) or type(exc).__name__
        achievable = getattr(exc, "achievable_fraction", None)
        error = SHORTFALL if any(word in reason.lower() for word in
            ("installed", "infeasible", "full-load hours")) else reason
        category = (ANNUAL_CAPACITY if "annual equipment and energy ceilings" in reason
                    or "installed" in reason.lower() or "full-load hours" in reason.lower()
                    else DISPATCH_FAILURE if "dispatch" in reason.lower() or "infeasible" in reason.lower()
                    else NUMERICAL_ERROR)
        if error and _energy_limited(reason):
            category = ENERGY_LIMITED
    metrics = {COST: float("nan") if error else r.delivered_cost_aud_per_kg_h2,
               EI: float("nan") if error else r.delivered_ei_kg_co2_per_kg_h2,
               "error": error, "failure_reason": reason, "failure_category": category}
    if case not in LNG_CASES:
        if r:
            metrics[UNMET_H2] = r.unmet_h2_kg / max(1, r.fleet_pem_h2_tpa*1000)
            metrics[UNMET_ELECTRICITY] = r.unmet_demand_mwh / max(1, r.fleet_total_electricity_kwh_per_year/1000)
            metrics["hydrogen_shortfall_kg"] = max(0, (r.target_h2_tpa-r.fleet_h2_delivered_tpa)*1000) if error else 0.0
        elif achievable is not None:
            # A hard pre-screen raised before dispatch ran, but reported how much
            # of the target its own installed equipment could still support.
            shortfall_fraction = min(1.0, max(0.0, 1 - achievable))
            metrics[UNMET_H2] = shortfall_fraction
            metrics[UNMET_ELECTRICITY] = float("nan")
            metrics["hydrogen_shortfall_kg"] = shortfall_fraction * base.target.target_h2_tpa * 1000
        else:
            metrics[UNMET_H2] = float("nan")
            metrics[UNMET_ELECTRICITY] = float("nan")
            metrics["hydrogen_shortfall_kg"] = float("nan")
        if r and not error:
            for field in ("energy_intensity_gj_per_t_h2", "pem_capacity_factor", "synth_capacity_factor",
                          "cracker_capacity_factor", "wind_curtailment_fraction", "battery_cycles_per_year",
                          "fleet_total_electricity_kwh_per_year"):
                metrics[field] = getattr(r, field)
            metrics.update(_equipment_counts(r))
            metrics.update({f"component.{k}": v for k, v in r.cost_breakdown_aud_per_kg_h2.items() if k != "Total"})
            metrics.update(getattr(r, "uncertainty_storage_metrics", {}))
    return metrics


def _energy_limited(reason):
    # Only when generation is every site's binding limit: a named equipment count
    # alongside it is still an equipment shortfall. The ceiling message opens
    # with a fleet summary clause before the per-site reasons.
    parts = [part for part in reason.split(";")
             if part.strip() and "Infeasible fixed build" not in part]
    return bool(parts) and all("annual renewable energy" in part for part in parts)


DESIGN_POINT_FIELDS = ("energy_intensity_gj_per_t_h2", "pem_capacity_factor", "synth_capacity_factor",
                       "cracker_capacity_factor", "wind_curtailment_fraction", "battery_cycles_per_year")


def design_point_metrics(result):
    """Cost, emissions and the performance metrics the draws report, for the reference run."""
    metrics = {COST: result.delivered_cost_aud_per_kg_h2, EI: result.delivered_ei_kg_co2_per_kg_h2}
    metrics.update({f: getattr(result, f) for f in DESIGN_POINT_FIELDS})
    return metrics


def design_point(case, optimum_path=None, mode="technical", allow_stale=False):
    """Design-point metrics of a saved run's build, for runs saved before they were recorded."""
    if case in LNG_CASES:
        return {}
    _, base, _, run = _setup(case, optimum_path, mode, allow_stale)
    return design_point_metrics(run(base))


def _equipment_counts(result):
    """Fleet equipment of the build a draw ran on; frozen in both modes."""
    counts = {"equipment.pem_modules": sum(result.site_n_pem.values()),
              "equipment.synthesis_trains": sum(result.site_n_synth.values()),
              "equipment.cracker_units": result.n_cracker,
              "equipment.carriers": result.fleet_ship_total}
    counts.update({f"equipment.pem_modules.{port}": n for port, n in result.site_n_pem.items()})
    return counts


_MC_CONTEXT = None


def _initialise_worker(case, optimum_path, mode, allow_stale):
    global _MC_CONTEXT
    _MC_CONTEXT = (case, *_setup(case, optimum_path, mode, allow_stale))


def _worker_draw(row):
    return _draw_metrics(row, *_MC_CONTEXT)


def run_monte_carlo(
    case: str,
    n_samples: int = 1000,
    seed: int = 0,
    optimum_path: Optional[Path] = None,
    progress: bool = True,
    workers: int = 1,
    mode: str = "economic",
    allow_stale: bool = False,
) -> McResult:
    if case not in CASES:
        raise ValueError(f"Unknown case {case!r}; choose from {CASES}")
    if mode not in MODES:
        raise ValueError(f"Unknown uncertainty mode {mode!r}; choose from {MODES}")
    if not isinstance(n_samples, int) or n_samples < 1:
        raise ValueError("n_samples must be a positive integer")
    require_current_inputs()
    fingerprint = model_fingerprint()
    bounds, base, copy, run = _setup(case, optimum_path, mode, allow_stale)
    reference = run(base)
    if case in BUILD_CASES and not reference.hourly_feasible:
        raise ValueError("The saved build falls short of the target at Typical inputs")
    # Each named input uses the same random stream in every case.
    draws = triangular_lhs(bounds, n_samples, np.random.default_rng(seed))

    from concurrent.futures import ProcessPoolExecutor
    if not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    records = []
    step = max(1, n_samples // 10)
    def collect(iterator):
        for i, metrics in enumerate(iterator, 1):
            records.append(metrics)
            if progress and (i % step == 0 or i == n_samples):
                print(f"  {case}: {i}/{n_samples} draws", flush=True)
    if workers == 1 or case in LNG_CASES:
        collect(_draw_metrics(row, case, bounds, base, copy, run) for row in draws)
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_initialise_worker,
                                 initargs=(case, optimum_path, mode, allow_stale)) as executor:
            collect(executor.map(_worker_draw, draws, chunksize=4))
    samples = pd.concat([
        pd.DataFrame(draws, columns=[f"{b.group}.{b.field}" for b in bounds]),
        pd.DataFrame(records),
    ], axis=1)
    if mode == "economic" and (samples["error"] != "").any():
        reasons = samples.loc[samples["error"] != "", "failure_reason"].unique()
        raise ValueError("Economic recost failed; cannot report an unconditional distribution: "
                         + "; ".join(reasons))
    if model_fingerprint() != fingerprint:
        raise ValueError("Model or inputs changed during uncertainty analysis; rerun before reporting results")
    return McResult(
        case=case,
        bounds=bounds,
        samples=samples,
        reference_cost=reference.delivered_cost_aud_per_kg_h2,
        reference_ei=reference.delivered_ei_kg_co2_per_kg_h2,
        mode=mode, fingerprint=fingerprint,
        reference_equipment={} if case in LNG_CASES else _equipment_counts(reference),
        reference_metrics={} if case in LNG_CASES else design_point_metrics(reference),
    )


def bootstrap_ci(values, q, n_boot=2000, seed=0):
    """95% bootstrap interval of the q-th percentile of values."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    stats = np.percentile(rng.choice(values, (n_boot, len(values))), q, axis=1)
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def summarise(result: McResult) -> dict:
    ok = result.feasible
    summary = {
        "case": result.case,
        "mode": result.mode,
        "draws": len(result.samples),
        "failed": len(result.samples) - len(ok),
        "falls_short": len(result.short),
        "deterministic_cost": result.reference_cost,
        "feasible_fraction": len(ok)/len(result.samples),
        "excluded_fraction": 1-len(ok)/len(result.samples),
        "failure_categories": result.samples["failure_category"].replace("", np.nan).value_counts().to_dict(),
        "distribution_basis": {
            "economic": "fixed design and verified operating schedule; feasibility inherited",
            "technical": "frozen equipment counts and renewable/storage footprint; only the continuous "
                         "ratings re-rate and recost to what each draw needs",
        }[result.mode],
        "sampling": "explicit workbook distributions and dependencies; screening assumptions, not calibrated probabilities",
    }
    if not result.short.empty:
        for q in (50, 90):
            values = result.short[UNMET_H2].dropna()
            summary[f"P{q}_unmet_h2_fraction"] = np.percentile(values, q) if len(values) else float("nan")
    if ok.empty:
        return summary
    cost, ei = ok[COST], ok[EI]
    summary.update(
        {
            "mean_cost": cost.mean(),
            "std_cost": cost.std(),
            **{f"P{q}_cost": np.percentile(cost, q) for q in PERCENTILES},
            # Sampling error only: whether the draw count is enough.
            **{f"P{q}_cost_ci95": bootstrap_ci(cost, q) for q in (10, 50, 90)},
            # Share of feasible draws at or below the single-point estimate.
            "prob_cost_le_deterministic": float((cost <= result.reference_cost).mean()),
            "deterministic_ei": result.reference_ei,
            "mean_ei": ei.mean(),
            **{f"P{q}_ei": np.percentile(ei, q) for q in (10, 50, 90)},
        }
    )
    return summary


# Inputs on one shared dependency are drawn at the same quantile, so they have
# identical rank correlations; grouped_drivers reports each group once.
DEPENDENCY_LABELS = {
    "shared:isp-step-change-capex": "PEM, renewable + battery CAPEX (one ISP trajectory)",
    "shared:storage-depleted-field": "Depleted-field H2 storage CAPEX",
    "shared:storage-salt-cavern": "Salt-cavern H2 storage CAPEX",
    "shared:lrc-capex": "Lined rock cavern CAPEX",
    "shared:h2-pipeline-capex": "H2 pipeline CAPEX (onshore + offshore)",
    "shared:meoh-ship-speed-fuel": "Ship speed + sea fuel consumption",
    "shared:route-allowance": "Route length allowance (all sites)",
}


GROUP_LABELS = {"pem": "PEM", "synth": "Synthesis", "cracker": "Cracker", "ship": "Shipping",
                "co2_cap": "CO2 capture", "transport": "H2 transport", "econ": "Economics"}


def grouped_drivers(result: McResult, target: pd.Series, top_n: int = 12) -> pd.Series:
    """Spearman correlation of each input, or shared-input group, with target.

    target is indexed like result.samples; rows where it is NaN are left out.
    The index holds display labels.
    """
    target = target.dropna()
    if len(target) < 3 or target.nunique() < 2:
        return pd.Series(dtype=float)
    inputs = result.samples.loc[target.index, result.inputs]
    rho = inputs.corrwith(target, method="spearman")
    members = {}
    for b in result.bounds:
        key = getattr(b, "dependency", "") or f"{b.group}.{b.field}"
        members.setdefault(key, []).append(b)
    names = [b.name for b in result.bounds]
    out = {}
    for key, bounds in members.items():
        value = rho.get(f"{bounds[0].group}.{bounds[0].field}")
        if value is None or not np.isfinite(value):
            continue
        label = DEPENDENCY_LABELS.get(key, key) if len(bounds) > 1 else bounds[0].name
        if len(bounds) == 1 and names.count(label) > 1:
            # Same workbook label on two plants (e.g. available hours).
            label = f"{GROUP_LABELS.get(bounds[0].group, bounds[0].group)}: {label}"
        out[label] = value
    out = pd.Series(out)
    return out.reindex(out.abs().sort_values(ascending=False).index).head(top_n)


def cost_drivers(result: McResult, top_n: int = 10) -> pd.Series:
    """Spearman rank correlation of each sampled input with delivered cost."""
    ok = result.feasible
    if len(ok) < 3 or ok[COST].nunique() < 2 or not result.inputs:
        return pd.Series(dtype=float)
    rho = ok[result.inputs].corrwith(ok[COST], method="spearman").dropna()
    return rho.reindex(rho.abs().sort_values(ascending=False).index).head(top_n)
