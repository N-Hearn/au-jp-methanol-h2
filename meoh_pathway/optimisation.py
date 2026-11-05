"""Search for the lowest-cost hourly feasible MeOH deployment found within a budget.

Workbook bounds cover per-site renewables, battery, regional/local H2 storage,
normalised delivered-H2 allocation weights and integer PEM modules and synthesis trains.
Minimum process fleets follow ceiling division at fixed physical availability;
actual utilisation and independent hourly dispatch follow the selected throughput.
Technology and market assumptions stay at workbook Typical values.

Co-location means choosing production ports and their independent capacities,
fleets and regional/local storage split. PEM follows each site's renewable
resource and synthesis stays at its port; workbook routes fix their separation.
The search does not relocate equipment along a route or pool sites.

The objective is USD/kg delivered H2 plus ei_weight times kg CO2/kg H2.
A feasible incumbent is retained, replayed and validated before saving. Finite
stochastic search and the dispatch policy do not establish global optimality.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version, PackageNotFoundError
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .parameters import (
    PORT_SITES,
    default_target,
    load_economics,
    load_pem,
    load_synth,
    load_ship,
    load_cracker,
    load_co2_capture,
    load_dac,
    XLSX_PATH,
)
from inputs.loader import load_site_inputs, capacity_bounds, inputs_digest, parameter_input
from .supply_chain import SupplyChain, SupplyChainResult
from .renewables_dispatch import load_wind_profile, load_solar_profile
from .sizing import fleet_bounds

SITE_DECISIONS = (
    "wind_mw",
    "solar_mw",
    "battery_mwh",
    "h2_storage_kg",
    "local_h2_storage_kg",
    "pem_modules",
    "h2_allocation_weight",
    "synth_modules",
)
CAPACITY_DECISIONS = SITE_DECISIONS[:6]
INTEGER_DECISIONS = ("synth_modules", "pem_modules")


def deployment_inputs(path=XLSX_PATH):
    """Return named decision rows from their owning modules, in stable site order."""
    sites = load_site_inputs(path)
    derived = fleet_bounds(path)
    rows = []
    for field in SITE_DECISIONS:
        for port, values in sites.items():
            if field not in values:
                rows.append((f"{field}_{port}",
                             {"Value": derived[port][field][0],
                              "Lower bound": derived[port][field][0],
                              "Upper bound": derived[port][field][1]}))
                continue
            row = values[field]
            # The workbook is the source of truth, but it must not claim more
            # equipment than the site's own generation could ever run: the Max
            # is documented as a physical ceiling, so a drifting copy would
            # silently widen the search into designs that cannot be supplied.
            ceiling = derived[port].get(field)
            if ceiling is not None and row.get("Upper bound") is not None:
                if float(row["Upper bound"]) > ceiling[1] + 1e-9:
                    raise ValueError(
                        f"{port}.{field}: workbook Max {row['Upper bound']} exceeds the "
                        f"physical ceiling {ceiling[1]} from fleet_bounds()")
            rows.append((f"{field}_{port}", row))
    return rows


def load_deployment_bounds(path=XLSX_PATH):
    """Split the workbook rows into searchable bounds and values fixed by Min == Max.

    A port with no wind profile has Min == Max == 0, which gives DE a dimension
    it cannot move. Since the population is popsize * n_vars, carrying those
    costs evaluations without ever changing a deployment. The same holds for
    every decision at a site whose renewable caps cannot reach the minimum
    viable site size: the model never switches it on.
    """
    rows = deployment_inputs(path)
    limits = {name: capacity_bounds(row, name) for name, row in rows}
    floor = float(parameter_input(path, "renewable_supply", "min_site_renewables_mw")["Value"])
    dead = {
        port
        for port in load_site_inputs(path)
        if limits[f"wind_mw_{port}"][2] + limits[f"solar_mw_{port}"][2] < floor
    }
    bounds, frozen = [], {}
    for name, row in rows:
        _, lo, hi = limits[name]
        if name.endswith("_operating_hours_per_year") and not 0 < lo <= hi <= 8760:
            raise ValueError(f"{name}: operating hours must be in (0, 8760]")
        if any(name == f"{field}_{port}" for field in SITE_DECISIONS for port in dead):
            frozen[name] = 0.0
        elif lo == hi:
            frozen[name] = lo
        else:
            bounds.append((name, lo, hi, name.startswith(tuple(f"{f}_" for f in INTEGER_DECISIONS))))
    if not bounds:
        raise ValueError("Every deployment input is fixed; nothing to optimise")
    return tuple(bounds), frozen


DEFAULT_BOUNDS, FROZEN_DECISIONS = load_deployment_bounds()
_WORKBOOK_DIGEST = hashlib.sha256(XLSX_PATH.read_bytes()).hexdigest()


def require_current_inputs():
    if hashlib.sha256(XLSX_PATH.read_bytes()).hexdigest() != _WORKBOOK_DIGEST:
        raise ValueError("Workbook changed after parameters were loaded; restart this analysis with current inputs")


@dataclass
class OptResult:
    success: bool
    message: str
    x: Dict[str, float]  # decision vars at optimum
    wind_region: str  # selected port site (from region_index)
    objective: float  # combined objective at optimum
    delivered_cost_aud_per_kg_h2: float
    delivered_cost_usd_per_kg_h2: float
    delivered_ei_kg_co2_per_kg_h2: float
    pem_lcoh_aud_per_kg_h2: float
    n_pem: int
    n_synth: int
    n_cracker: int
    solar_capacity_mw: float
    wind_capacity_mw: float
    battery_energy_capacity_mwh: float
    battery_power_capacity_mw: float
    wind_curtailment_fraction: float
    h2_storage_capacity_kg: float
    region_renewables_capex_multiplier: float
    region_battery_capex_multiplier: float
    site_wind_mw: Dict[str, float] = field(default_factory=dict)
    site_solar_mw: Dict[str, float] = field(default_factory=dict)
    site_n_pem: Dict[str, int] = field(default_factory=dict)
    site_n_synth: Dict[str, int] = field(default_factory=dict)
    site_h2_storage_kg: Dict[str, float] = field(default_factory=dict)
    site_battery_mwh: Dict[str, float] = field(default_factory=dict)
    site_pem_operating_hours: Dict[str, float] = field(default_factory=dict)
    nfev: int = 0  # number of objective evaluations
    full_result: Optional[SupplyChainResult] = field(default=None, repr=False)
    search_settings: dict = field(default_factory=dict)
    # Bounded, diversity-preserving sample of the feasible candidates seen during
    # the search (see _CandidateArchive) -- {objective, active_sites, x} per entry,
    # raw (not yet re-evaluated for cost/EI/footprint; see evaluate_candidates).
    # Kept out of search_settings so it is not embedded in optimum.json.
    candidates: list = field(default_factory=list, repr=False)
    # Every refined start, decision vector included -- the archive and trajectory
    # record what the search saw, this records what it actually refined. Written
    # to refined_candidates.json so alternative builds can be compared, not just
    # the winner. Kept out of search_settings so it is not embedded in optimum.json.
    refined_starts: list = field(default_factory=list, repr=False)


# Excel I/O is the slowest part of an evaluation, so cache the base
# parameter set once and copy from it inside the inner loop.
_BASE_CACHE: Dict[str, object] = {}


def _base_params():
    if not _BASE_CACHE:
        _BASE_CACHE["econ"] = load_economics()
        _BASE_CACHE["pem"] = load_pem()
        _BASE_CACHE["synth"] = load_synth()
        _BASE_CACHE["ship"] = load_ship()
        _BASE_CACHE["cracker"] = load_cracker()
        _BASE_CACHE["co2_cap"] = load_co2_capture()
        _BASE_CACHE["dac"] = load_dac()
        _BASE_CACHE["target"] = default_target()
    return _BASE_CACHE


def deployment_from_vector(x, preserve_assets=False):
    """Decode one validated decision vector for every model entry point."""
    if len(x) != len(DEFAULT_BOUNDS):
        raise ValueError(
            f"Expected {len(DEFAULT_BOUNDS)} decision values, got {len(x)}"
        )
    values = {name: float(value) for (name, _, _, _), value in zip(DEFAULT_BOUNDS, x)}
    if not all(math.isfinite(v) for v in values.values()):
        raise ValueError("Decision values must be finite")
    for name, lo, hi, integer in DEFAULT_BOUNDS:
        if not lo <= values[name] <= hi:
            raise ValueError(f"{name}: decision is outside the current Excel Min/Max")
        if integer and int(values[name]) != values[name]:
            raise ValueError(f"{name}: train count must be an integer")
    values.update(FROZEN_DECISIONS)
    maps = {
        field: {p: values[f"{field}_{p}"] for p in PORT_SITES}
        for field in SITE_DECISIONS
    }
    if any(v < 0 for m in maps.values() for v in m.values()):
        raise ValueError("Site capacities cannot be negative")
    base = _base_params()
    floor = max(base["pem"].min_site_renewables_mw, 1e-9)
    active = {p for p in PORT_SITES
              if maps["wind_mw"][p] + maps["solar_mw"][p] >= floor
              and maps["h2_allocation_weight"][p] > 0}
    if not active:
        raise ValueError("At least one site needs renewable capacity")
    # An unused port has no storage plant. Preserve explicit zero at active ports.
    for field in SITE_DECISIONS:
        maps[field] = {
            p: (int(v) if field in INTEGER_DECISIONS else v)
            if p in active or (preserve_assets and field in SITE_DECISIONS[:5]) else 0
            for p, v in maps[field].items()
        }
    deployment = {f"site_{field}": values for field, values in maps.items()}
    if preserve_assets:
        deployment["preserve_assets"] = True
    return deployment


def _active_sites(x) -> frozenset:
    """Ports this decision vector actually builds -- the archive's diversity key."""
    try:
        deployment = deployment_from_vector(list(x))
    except ValueError:
        return frozenset()
    return frozenset(p for p, w in deployment["site_h2_allocation_weight"].items() if w > 0)


class _CandidateArchive:
    """Bounded, diversity-preserving store of feasible (vector, objective) pairs
    seen during a search.

    Purely observational: offer() never changes what the search evaluates next,
    so the optimum found is identical whether or not an archive is attached.
    Bucketing by active-site signature (see _active_sites) is what keeps the
    archive from collapsing onto one basin the way a plain top-K would.
    """

    def __init__(self, bounds, per_signature=10, total_cap=500):
        self._bounds = bounds
        self._per_signature = per_signature
        self._total_cap = total_cap
        self._by_signature: Dict[frozenset, List[Tuple[float, Tuple[float, ...]]]] = {}
        self._seen_keys = set()

    def _dedup_key(self, x):
        # Round each coordinate to 1% of its range so near-identical population
        # members (DE mutants a float apart) collapse to one archive entry.
        return tuple(
            round((v - lo) / (hi - lo), 2) if hi > lo else 0.0
            for v, (_, lo, hi, _) in zip(x, self._bounds)
        )

    def offer(self, x, value):
        value = float(value)
        if not math.isfinite(value) or value >= 1e6:
            return
        bucket_key = self._dedup_key(x)
        if bucket_key in self._seen_keys:
            return
        signature = _active_sites(x)
        bucket = self._by_signature.setdefault(signature, [])
        if len(bucket) >= self._per_signature and value >= bucket[-1][0]:
            return
        self._seen_keys.add(bucket_key)
        bucket.append((value, tuple(float(v) for v in x)))
        bucket.sort(key=lambda item: item[0])
        del bucket[self._per_signature:]

    def entries(self):
        """Up to total_cap (objective, vector, active_sites) triples, cheapest first."""
        rows = [(value, x, sig) for sig, bucket in self._by_signature.items() for value, x in bucket]
        rows.sort(key=lambda item: item[0])
        return rows[: self._total_cap]


def _seed_footprints(x, bnds, ei_weight, archive, evaluate):
    """Offer one-port-closed designs to the archive before starts are chosen.

    _refinement_starts wants one design per distinct footprint, but at DE
    completion the archive holds almost nothing but all-ports-on members -- DE
    rarely lands on an exact closure -- so it falls through to its
    cheapest-remaining top-up and every start shares the incumbent's footprint.
    The closures evaluated here are the same ones refinement's first stage would
    try anyway; doing them up front is what gives the selection real choices.
    Returns the evaluations spent.
    """
    spent = 0
    for port in sorted(_active_sites(x)):
        for margin in _CLOSURE_MARGINS:
            trial = _close_and_redistribute(x, bnds, port, margin)
            if trial is None:
                continue
            value = evaluate(trial, ei_weight, None)
            spent += 1
            archive.offer(trial, value)
    return spent


def _canonical_start(archive, x):
    """x as the model sees it: integer trains rounded, floats copied."""
    x = np.asarray(x, dtype=float).copy()
    integer = [is_int for _, _, _, is_int in archive._bounds]
    x[integer] = np.rint(x[integer])
    return x


def _same_design(archive, a, a_value, b, b_value, tol=1e-6):
    """True if two starts are one design: equal objective, or equal vectors once
    integer trains are rounded and continuous values compared to tol of their range.

    DE archives population members that evaluate to the incumbent's exact
    objective yet differ from it in a float or an unrounded train count, so
    np.array_equal missed them and the incumbent was refined twice.
    """
    if abs(a_value-b_value) <= 1e-9*max(1.0, abs(a_value)):
        return True
    span = np.array([max(hi-lo, 1e-12) for _, lo, hi, _ in archive._bounds])
    return bool(np.all(np.abs(a-b)/span <= tol))


def _refinement_starts(archive, incumbent, incumbent_value, count):
    """The incumbent plus the best archived design of each other site footprint.

    Refinement is a greedy hill-climb, so whichever basin DE happens to end in
    decides the answer -- two runs of this model differed by 0.8 AUD/kg purely
    that way. Starting it from the cheapest design of each distinct footprint
    the search actually saw turns that into a measured spread instead of luck,
    and unlike warm-starting it does not bias DE's own exploration.

    Returns (x, value, origin) triples; origin is "incumbent", "distinct footprint"
    or "top-up". No start repeats the incumbent or an earlier start (see
    _same_design), so fewer than count are returned if the archive runs out.
    """
    starts = [(_canonical_start(archive, incumbent), float(incumbent_value), "incumbent")]
    if count <= 1:
        return starts
    entries = archive.entries()
    seen = {_active_sites(incumbent)}
    taken = set()

    def duplicate(x, value):
        return any(_same_design(archive, x, value, sx, sv) for sx, sv, _ in starts)

    for index, (value, x, signature) in enumerate(entries):
        if len(starts) >= count:
            break
        x = _canonical_start(archive, x)
        if signature in seen or duplicate(x, float(value)):
            continue
        seen.add(signature)
        taken.add(index)
        starts.append((x, float(value), "distinct footprint"))
    # Most DE members leave every port switched on, so distinct footprints can
    # run out well before count. Top up with the cheapest designs left so the
    # spread still has something to measure -- but the caller reports how many
    # came from here, because a spread of same-footprint starts measures nothing.
    for index, (value, x, _) in enumerate(entries):
        if len(starts) >= count:
            break
        x = _canonical_start(archive, x)
        if index in taken or duplicate(x, float(value)):
            continue
        starts.append((x, float(value), "top-up"))
    return starts


def build_supply_chain(x: List[float], parameter_overrides=None) -> SupplyChain:
    base = _base_params()
    params = {name: replace(value) for name, value in base.items()}
    if parameter_overrides:
        params.update({name: replace(value) for name, value in parameter_overrides.items()
                       if not name.startswith("_")})
    return SupplyChain(
        economics=params["econ"],
        pem=params["pem"],
        synth=params["synth"],
        ship=params["ship"],
        cracker=params["cracker"],
        co2_capture=params["co2_cap"],
        dac=params["dac"],
        target=params["target"],
        h2_transport=params.get("transport"),
        site_storage_costs=(parameter_overrides or {}).get("_site_storage_costs"),
        **deployment_from_vector(x, bool((parameter_overrides or {}).get("_preserve_assets"))),
    )


def _margin_overrides(parameter_overrides, margin):
    """parameter_overrides with the H2 target raised by the delivery margin."""
    overrides = dict(parameter_overrides or {})
    target = overrides.get("target", _base_params()["target"])
    overrides["target"] = replace(target, target_h2_tpa=target.target_h2_tpa * (1 + margin))
    return overrides


def _penalised(r, obj):
    obj += 1e6
    obj += 10000 * r.unmet_h2_kg / max(r.fleet_pem_h2_tpa * 1000, 1)
    obj += (
        10000
        * r.unmet_demand_mwh
        / max(r.fleet_total_electricity_kwh_per_year / 1000, 1)
    )
    return obj


def _evaluate(x: List[float], ei_weight: float, parameter_overrides=None) -> float:
    """Combined cost+EI objective. Smooth penalty for infeasible runs so DE
    can navigate toward feasibility rather than hitting a flat 1e9 wall.

    With a delivery margin m (parameter_overrides["_delivery_margin"]) the
    build must first be hourly feasible at (1 + m) x target -- production
    headroom held against technical shortfalls -- and is then costed operating
    at the contracted target, so the objective is what the headroom costs.
    """
    margin = (parameter_overrides or {}).get("_delivery_margin", 0.0)
    try:
        if margin:
            stressed = build_supply_chain(x, _margin_overrides(parameter_overrides, margin)).run(
                allow_infeasible=True)
            if not stressed.hourly_feasible:
                return _penalised(stressed, stressed.delivered_cost_usd_per_kg_h2
                                  + ei_weight * stressed.delivered_ei_kg_co2_per_kg_h2)
        r = build_supply_chain(x, parameter_overrides).run(allow_infeasible=True)
    except (ValueError, ArithmeticError, RuntimeError):
        return 1e9
    obj = r.delivered_cost_usd_per_kg_h2 + ei_weight * r.delivered_ei_kg_co2_per_kg_h2
    return obj if r.hourly_feasible else _penalised(r, obj)


def _default_x0() -> np.ndarray:
    """Seed the full search from a current calculated Typical balance."""
    from .typical import typical_seed
    return typical_seed()


def _workbook_x0() -> np.ndarray:
    """Deployment capacities only; process decisions are calculated in code.

    Every process-decision row's synthetic Value equals its own Lower bound
    (see deployment_inputs's fleet_bounds fallback), and the five capacity
    rows have no Typical any more, so Min is the honest seed for all of them.
    """
    return np.array([lo for _, lo, _, _ in DEFAULT_BOUNDS], dtype=float)


def warm_start(path: Path) -> np.ndarray:
    """Read an old vector as a search seed, without claiming it is current or feasible."""
    payload = json.loads(Path(path).read_text())
    if len(payload["bounds_names"]) != len(payload["x"]):
        raise ValueError("Warm-start vector has inconsistent names and values")
    old = dict(zip(payload["bounds_names"], payload["x"]))
    design = payload.get("deployment", {})
    if not design and (Path(path).parent / "results.json").exists():
        design = json.loads((Path(path).parent / "results.json").read_text())
    for port, share in design.get("site_h2_frac", {}).items():
        old.setdefault(f"h2_allocation_weight_{port}", share)
    # synth_modules is an absolute installed count (like pem_modules), so an
    # old saved deployment's site_n_synth is the value directly -- no need to
    # back out an increment above the availability minimum any more.
    for port, count in design.get("site_n_synth", {}).items():
        old.setdefault(f"synth_modules_{port}", count)
    if design.get("site_h2_frac"):
        for port in PORT_SITES:
            old.setdefault(f"h2_allocation_weight_{port}", 0.0)
    typical = _default_x0()
    return _clip_x0_to_bounds(
        np.array([old.get(name, typical[i]) for i, (name, _, _, _) in enumerate(DEFAULT_BOUNDS)]),
        [(lo, hi) for _, lo, hi, _ in DEFAULT_BOUNDS])


def _clip_x0_to_bounds(x0: np.ndarray, bnds) -> np.ndarray:
    values = np.asarray(x0, dtype=float)
    if values.shape == (1, len(bnds)):
        values = values[0]
    if values.shape != (len(bnds),) or not np.all(np.isfinite(values)):
        raise ValueError(f"x0 must contain {len(bnds)} finite values")
    return np.clip(values, [lo for lo, _ in bnds], [hi for _, hi in bnds])


def _initial_population(x0, bnds, popsize, seed, spread=0.25):
    """Stratified population on both sides of the seed, retaining the seed itself.

    An upper-side-only population has zero spread when a seed is at its bound,
    so differential mutation cannot reduce that coordinate.
    """
    lo, hi = np.asarray(bnds, dtype=float).T
    count = max(5, popsize * int(np.count_nonzero(hi > lo)))
    rng = np.random.default_rng(seed)
    samples = np.column_stack(
        [(rng.permutation(count) + rng.random(count)) / count for _ in bnds]
    )
    population = lo + samples * (hi - lo)
    population = np.clip(x0 + spread * (population - x0), lo, hi)
    if not 0 < spread <= 1:
        raise ValueError("spread must be in (0, 1]")
    population[0] = x0
    population[1] = hi
    return population


@lru_cache(maxsize=None)
def _profile_energy(port):
    """Annual wind and solar energy per MW at a port.

    Refinement asks for these on every share move; the profile loaders are
    cached but re-summing 8760 values each call is not free.
    """
    return float(load_wind_profile(port).sum()), float(load_solar_profile(port).sum())


def _site_energy(port, wind_mw, solar_mw):
    wind, solar = _profile_energy(port)
    return wind_mw*wind + solar_mw*solar


def _transfer_share(x, bnds, donor, receiver, fraction):
    """Move a fraction of the donor's production to the receiver, capacity included.

    A weight change alone leaves one site short of generation and storage, so
    nearly every reallocation looks infeasible. Capacities scale with each
    site's share; an unused receiver takes the donor's released energy on its
    own profiles and the donor's storage per unit share. Returns None when the
    move is not available.
    """
    names = [b[0] for b in DEFAULT_BOUNDS]
    positions = {name: i for i, name in enumerate(names)}
    wd, wr = f"h2_allocation_weight_{donor}", f"h2_allocation_weight_{receiver}"
    if donor == receiver or wd not in positions or wr not in positions or not 0 < fraction <= 1:
        return None
    x = np.asarray(x, dtype=float)
    values = dict(zip(names, x)) | FROZEN_DECISIONS
    upper = lambda name: bnds[positions[name]][1] if name in positions else values[name]
    energy = _site_energy
    floor = max(_base_params()["pem"].min_site_renewables_mw, 1e-9)
    renewables = lambda port: values[f"wind_mw_{port}"] + values[f"solar_mw_{port}"]
    if values[wd] <= 0 or renewables(donor) < floor:
        return None
    opened = values[wr] > 0 and renewables(receiver) >= floor
    old = values[wr] if opened else 0.0
    moved = min(fraction*values[wd], values[wd] - bnds[positions[wd]][0], upper(wr) - old)
    if moved <= 0:
        return None
    share = moved / values[wd]
    new = dict(values)
    new[wd], new[wr] = values[wd] - moved, old + moved
    for field in CAPACITY_DECISIONS:
        new[f"{field}_{donor}"] = values[f"{field}_{donor}"] * (1 - share)
        new[f"{field}_{receiver}"] = (values[f"{field}_{receiver}"] * new[wr] / old if opened
                                      else values[f"{field}_{donor}"] * share)
    # Synthesis trains travel with the production they serve. Scaling them by the
    # receiver's weight ratio, as the capacities are, blows up once the receiver's
    # weight has shrunk near zero while its count, rounded up, cannot fall below
    # one: a thousandfold ratio on one train hit the train ceiling.
    moved_trains = values[f"synth_modules_{donor}"] * share
    new[f"synth_modules_{donor}"] = values[f"synth_modules_{donor}"] - moved_trains
    new[f"synth_modules_{receiver}"] = (values[f"synth_modules_{receiver}"] if opened else 0.0) + moved_trains
    if not opened:
        released = share * energy(donor, values[f"wind_mw_{donor}"], values[f"solar_mw_{donor}"])
        room = {kind: upper(f"{kind}_mw_{receiver}") for kind in ("wind", "solar")}
        room_energy = energy(receiver, room["wind"], room["solar"])
        if room_energy <= 0:
            return None
        # Split the released energy by where the receiver's caps can supply it.
        for kind in room:
            new[f"{kind}_mw_{receiver}"] = room[kind] * released / room_energy
    trial = x.copy()
    for name, i in positions.items():
        value = min(bnds[i][1], max(bnds[i][0], new[name]))
        # A synthesis shortfall fails the run outright, a penalty the search cannot
        # climb; round trains up and let refinement trim any spare one.
        trial[i] = (math.ceil(value - 1e-9) if name.startswith("synth_modules_")
                    else round(value) if DEFAULT_BOUNDS[i][3] else value)
    return None if np.array_equal(trial, x) else trial


def _search_population(x0, bnds, popsize, seed, spread):
    """Keep most starting candidates close to hourly feasibility."""
    population = _initial_population(x0, bnds, popsize, seed, spread)
    names = [b[0] for b in DEFAULT_BOUNDS]
    positions = {name: i for i, name in enumerate(names)}
    resource_seed = population[1].copy()
    for i, (name, _, _, integer) in enumerate(DEFAULT_BOUNDS):
        hi = bnds[i][1]
        if name.startswith(tuple(
            f"{field}_" for field in CAPACITY_DECISIONS + ("synth_modules",)
        )):
            resource_seed[i] = round(hi) if integer else hi
    def resource_value(name):
        position = positions.get(name)
        return resource_seed[position] if position is not None else FROZEN_DECISIONS.get(name, 0.0)
    energy = {}
    for port in PORT_SITES:
        wind = resource_value(f"wind_mw_{port}")
        solar = resource_value(f"solar_mw_{port}")
        energy[port] = (
            wind * load_wind_profile(port).sum()
            + solar * load_solar_profile(port).sum()
        )
    total_energy = sum(energy.values())
    if total_energy > 0:
        for port in PORT_SITES:
            position = positions.get(f"h2_allocation_weight_{port}")
            if position is not None:
                lo, hi = bnds[position]
                resource_seed[position] = np.clip(
                    energy[port] / total_energy, lo, hi
                )
    seed_values = dict(zip(names, x0)) | FROZEN_DECISIONS
    active = {port for port in PORT_SITES if seed_values[f"h2_allocation_weight_{port}"] > 0}
    live = [i for i, name in enumerate(names) if bnds[i][0] < bnds[i][1]
            and any(name.endswith("_"+port) for port in active)]
    if not live:
        return population
    rng = np.random.default_rng(seed)
    pairs = [(donor, receiver) for donor in PORT_SITES for receiver in PORT_SITES
             if _transfer_share(x0, bnds, donor, receiver, 0.5) is not None]
    for k in range(1, len(population)):
        if k % 5 in (3, 4) and pairs:
            # Transfers give DE difference vectors that move a share and its
            # capacity together, including into sites the seed does not use.
            donor, receiver = pairs[rng.integers(len(pairs))]
            trial = _transfer_share(x0, bnds, donor, receiver, rng.uniform(0.05, 1))
            population[k] = x0 if trial is None else trial
        elif k % 5:
            # Changing an entire portfolio at once usually destroys hourly
            # feasibility. Sparse moves give DE useful feasible alternatives.
            population[k] = x0
            chosen = rng.choice(live, size=int(rng.integers(1, min(4, len(live))+1)), replace=False)
            for j in chosen:
                lo, hi = bnds[j]
                if DEFAULT_BOUNDS[j][3]:
                    population[k, j] = np.clip(x0[j]+rng.choice((-1, 1)), lo, hi)
                else:
                    scale = x0[j] if x0[j] else hi-lo
                    population[k, j] = np.clip(x0[j]+spread*scale*rng.uniform(-1, 1)/len(chosen), lo, hi)
        else:
            # Resource-proportional weights are seeds only. DE subsequently
            # optimises the allocations independently, including new sites.
            values = dict(zip(names, population[k])) | FROZEN_DECISIONS
            energy = {p: values[f"wind_mw_{p}"]*load_wind_profile(p).sum()
                      + values[f"solar_mw_{p}"]*load_solar_profile(p).sum() for p in PORT_SITES}
            total = sum(energy.values())
            if total > 0:
                for port in PORT_SITES:
                    j = positions.get(f"h2_allocation_weight_{port}")
                    if j is not None:
                        population[k, j] = np.clip(energy[port]/total, *bnds[j])
    population[1] = resource_seed
    return population


# Capacity over-provision fractions tried when closing a site (see _prune_sites).
_CLOSURE_MARGINS = (0.0, 0.1, 0.25)
# Consecutive non-improving rounds before a refinement stage is retired for the
# rest of that start.
_STAGE_PATIENCE = 3


def _close_and_redistribute(x, bnds, port, margin=0.0):
    """Close a port and move its production onto the survivors, capacity included.

    Zeroing a port on its own renormalises its share onto sites whose capacity
    does not grow, so the hydrogen it was carrying is stranded and the move is
    rejected however good the closure would have been. Each survivor takes a
    slice of the released share in proportion to what it already carries, and
    scales its capacities with it. margin over-provisions those capacities:
    proportional growth alone usually lands a fraction of a percent short, and
    _reduce_pem trims the excess back afterwards. Returns None when the move is
    unavailable or the survivors cannot absorb the share.
    """
    names = [b[0] for b in DEFAULT_BOUNDS]
    positions = {name: i for i, name in enumerate(names)}
    donor_weight = f"h2_allocation_weight_{port}"
    if donor_weight not in positions:
        return None
    values = dict(zip(names, np.asarray(x, dtype=float))) | FROZEN_DECISIONS
    released = values[donor_weight]
    floor = max(_base_params()["pem"].min_site_renewables_mw, 1e-9)
    renewables = lambda site: values[f"wind_mw_{site}"] + values[f"solar_mw_{site}"]
    if released <= 0 or renewables(port) < floor:
        return None
    upper = lambda name: bnds[positions[name]][1] if name in positions else values[name]

    room = {}
    for site in PORT_SITES:
        weight = f"h2_allocation_weight_{site}"
        if site == port or weight not in positions:
            continue
        if values[weight] <= 0 or renewables(site) < floor:
            continue  # an unused site needs a donor's profile, not a share slice
        room[site] = max(0.0, upper(weight)-values[weight])
    total = sum(room.values())
    if total < released:
        return None

    # Share out by how much production each survivor already carries, and grow
    # its capacities with it. A site at its wind or solar cap simply keeps what
    # it has: this build spills a third of its wind, so extra electrolyser and
    # storage alone can carry more hydrogen there. Whether that works is for
    # the evaluation to say, not for a conservative rule here to pre-judge.
    weights = {site: values[f"h2_allocation_weight_{site}"] for site in room}
    carried = sum(weights.values())
    new = dict(values)
    for site, capacity in room.items():
        take = min(capacity, released*weights[site]/carried) if carried > 0 else 0.0
        if take <= 0:
            continue
        weight = f"h2_allocation_weight_{site}"
        scale = (values[weight]+take) / values[weight]
        new[weight] = values[weight] + take
        for field in CAPACITY_DECISIONS:
            key = f"{field}_{site}"
            if key not in positions:
                continue  # frozen at a single value (e.g. a port with no wind)
            grown = values[key] * scale * (1+margin)
            new[key] = math.ceil(grown) if DEFAULT_BOUNDS[positions[key]][3] else grown
        # The closed port's trains follow its production (see _transfer_share).
        trains = f"synth_modules_{site}"
        new[trains] = math.ceil((values[trains] + values[f"synth_modules_{port}"] * take/released)
                                * (1+margin) - 1e-9)
    for field in SITE_DECISIONS:
        new[f"{field}_{port}"] = 0.0

    trial = np.asarray(x, dtype=float).copy()
    for name, i in positions.items():
        value = min(bnds[i][1], max(bnds[i][0], new[name]))
        trial[i] = round(value) if DEFAULT_BOUNDS[i][3] else value
    return None if np.array_equal(trial, x) else trial


def _prune_sites(x, bnds, ei_weight, parameter_overrides=None, evaluate=None):
    """Try exact site closures, which continuous DE almost never lands on."""
    evaluate = evaluate or _evaluate
    x = np.asarray(x, dtype=float).copy()
    objective, evaluations = evaluate(x, ei_weight, parameter_overrides), 1
    names = [name for name, _, _, _ in DEFAULT_BOUNDS]
    for port in PORT_SITES:
        indices = [i for i, name in enumerate(names)
                   if name in {f'{field}_{port}' for field in SITE_DECISIONS}]
        if not any(x[i] for i in indices) or any(bnds[i][0] > 0 for i in indices):
            continue
        bare = x.copy()
        bare[indices] = 0
        # The bare closure only lands when the survivors already had slack; the
        # redistributing one is what finds a cheaper footprint. Proportional
        # growth alone is typically just short, so try it over-provisioned too.
        trials = [bare] + [_close_and_redistribute(x, bnds, port, margin)
                           for margin in _CLOSURE_MARGINS]
        closed, closed_value = None, math.inf
        for trial in trials:
            if trial is None:
                continue
            value = evaluate(trial, ei_weight, parameter_overrides)
            evaluations += 1
            if value <= objective:
                x, objective = trial, value
                break
            if value < closed_value:
                closed, closed_value = trial, value
        else:
            # Judged straight after redistribution, a closure carries the
            # survivors' proportional, untuned growth and almost always loses,
            # though it may win once PEM, storage and shares are re-tuned (a
            # five-site build beat the eight-site one it was rejected from by
            # 7%). Re-tune the best closure briefly, its port held shut, before
            # judging it. A hard failure has no gradient to tune along.
            if closed is not None and closed_value < 1e9:
                pinned = [(0.0, 0.0) if i in indices else b for i, b in enumerate(bnds)]
                for stage in _CLOSURE_POLISH:
                    closed, spent = stage(closed, pinned, ei_weight,
                                          parameter_overrides=parameter_overrides, evaluate=evaluate)
                    evaluations += spent
                value = evaluate(closed, ei_weight, parameter_overrides)
                if value < objective:
                    x, objective = closed, value
    return x, evaluations

def _refine_feasible(x, bnds, ei_weight, parameter_overrides=None, evaluate=None):
    """Small coordinate steps across discrete fleet and dispatch boundaries."""
    evaluate = evaluate or _evaluate
    x = np.asarray(x, dtype=float).copy()
    objective, evaluations = evaluate(x, ei_weight, parameter_overrides), 1
    for step in (0.1, 0.03):
        for i, (name, _, _, integer) in enumerate(DEFAULT_BOUNDS):
            lo, hi = bnds[i]
            if hi <= lo:
                continue
            delta = 1 if integer else step*(x[i] if x[i] else hi-lo)
            for direction in (-1, 1):
                trial = x.copy()
                trial[i] = min(hi, max(lo, x[i]+direction*delta))
                if trial[i] == x[i]:
                    continue
                value = evaluate(trial, ei_weight, parameter_overrides)
                evaluations += 1
                if value < objective:
                    x, objective = trial, value
    return x, evaluations


def _refine_shares(x, bnds, ei_weight, fractions=(1.0, 0.3, 0.1), parameter_overrides=None, evaluate=None):
    """Greedy production transfers between every pair of sites.

    A full transfer closes the donor; a transfer to an unused site opens it.
    """
    evaluate = evaluate or _evaluate
    x = np.asarray(x, dtype=float).copy()
    objective, evaluations = evaluate(x, ei_weight, parameter_overrides), 1
    for fraction in fractions:
        for donor in PORT_SITES:
            for receiver in PORT_SITES:
                trial = _transfer_share(x, bnds, donor, receiver, fraction)
                if trial is None:
                    continue
                value = evaluate(trial, ei_weight, parameter_overrides)
                evaluations += 1
                if value < objective:
                    x, objective = trial, value
    return x, evaluations


def _reduce_pem(x, bnds, ei_weight, parameter_overrides=None, evaluate=None):
    """Try coarse-to-fine integer reductions without assuming monotone dispatch.

    Covers synthesis trains as well as PEM modules: their workbook ceilings run
    to thousands, so a search that starts far from feasibility can leave a site
    with hundreds of idle trains that one-train coordinate steps never trim
    (a 10% margin build kept 237 at a site needing 5, +2.9 AUD/kg).
    """
    evaluate = evaluate or _evaluate
    x = np.asarray(x, dtype=float).copy()
    objective, evaluations = evaluate(x, ei_weight, parameter_overrides), 1
    for i, (name, _, _, _) in enumerate(DEFAULT_BOUNDS):
        if not name.startswith(("pem_modules_", "synth_modules_")):
            continue
        floor = math.ceil(bnds[i][0])
        step = max(1, int(x[i] - floor) // 2)
        while step:
            while x[i] - step >= floor:
                trial = x.copy()
                trial[i] -= step
                value = evaluate(trial, ei_weight, parameter_overrides)
                evaluations += 1
                if value >= objective:
                    break
                x, objective = trial, value
            step //= 2
    return x, evaluations


# The short re-tune a rejected closure gets before it is judged (_prune_sites):
# trim PEM, rebalance shares among the survivors, then one coordinate pass.
_CLOSURE_POLISH = (
    _reduce_pem,
    lambda x, bnds, w, parameter_overrides=None, evaluate=None: _refine_shares(
        x, bnds, w, fractions=(0.3, 0.1), parameter_overrides=parameter_overrides, evaluate=evaluate),
    _refine_feasible,
)


class _BudgetExhausted(Exception):
    pass


def _refinement_stages(refine):
    if not refine:
        return [("site closures", _prune_sites)]
    # Closures run first: the stage is the cheapest and the only one that can
    # change the site footprint, and running it last meant an exhausted budget
    # starved it (2000 evaluations bought it 27 of ~2600).
    return [("site closures", _prune_sites), ("PEM reduction", _reduce_pem),
            ("production transfers", _refine_shares),
            ("coordinate refinement", _refine_feasible)]


def _refine_start(start_x, start_value, label, all_bounds, ei_weight, parameter_overrides,
                  refine, budget, rounds, min_improvement, progress, memo=None, archive=None):
    """Refinement rounds from one start, within its own budget.

    Module level so a process pool can run starts side by side. A serial caller
    passes its shared memo and archive; a worker builds its own and returns the
    archive entries for the caller to merge. History rows carry seconds since
    this start began and evaluations spent by it.
    """
    bnds = [(lo, hi) for _, lo, hi, _ in all_bounds]
    stages = _refinement_stages(refine)
    own_archive = archive is None
    if own_archive:
        archive = _CandidateArchive(all_bounds)
    memo = {} if memo is None else memo
    started = time.perf_counter()
    if progress:
        print(f"  Refining {label}: objective {start_value:.6g}, "
              f"{len(_active_sites(start_x))} active sites", flush=True)
    best_x, best_value = np.asarray(start_x, dtype=float).copy(), float(start_value)
    spent = [0]
    current_stage = [label]
    trajectory, history, stage_counts = [], [], {}
    # A stage that has stopped paying keeps charging: production transfers
    # spent roughly 40% of one run's budget while improving nothing after the
    # early rounds. Retire it and leave the budget to the stages still finding moves.
    idle = {stage_label: 0 for stage_label, _ in stages}

    def evaluate_refinement(trial, weight, overrides):
        nonlocal best_x, best_value
        key = np.asarray(trial, dtype=float).tobytes()
        if key in memo:
            value = memo[key]
        else:
            if spent[0] >= budget:
                raise _BudgetExhausted
            value = _evaluate(trial, weight, overrides)
            memo[key] = value
            spent[0] += 1
            archive.offer(trial, value)
        # The stage functions' return values are discarded, so the running best
        # has to be captured here -- including on a cache hit, or an improving
        # move another start already priced would be lost.
        if value < best_value:
            best_x, best_value = np.asarray(trial, dtype=float).copy(), value
            trajectory.append((current_stage[0], float(value), best_x.copy()))
        return value

    stop_reason = "round limit"
    for round_number in range(1, rounds+1 if refine else 2):
        round_value = best_value
        for stage_label, function in stages:
            if idle[stage_label] >= _STAGE_PATIENCE:
                continue
            current_stage[0] = f"{label} round {round_number}: {stage_label}"
            previous, before = spent[0], best_value
            exhausted = False
            try:
                function(best_x, bnds, ei_weight, parameter_overrides=parameter_overrides,
                         evaluate=evaluate_refinement)
            except _BudgetExhausted:
                exhausted = True
            idle[stage_label] = 0 if best_value < before else idle[stage_label]+1
            stage_counts[stage_label] = stage_counts.get(stage_label, 0) + spent[0]-previous
            history.append((current_stage[0], spent[0], best_value, time.perf_counter()-started))
            if progress:
                print(f"  {label} round {round_number}: {stage_label}: objective {best_value:.6g}; "
                      f"{spent[0]}/{budget} evaluations", flush=True)
            if exhausted:
                stop_reason = "evaluation budget"
                break
        if stop_reason == "evaluation budget":
            break
        if all(count >= _STAGE_PATIENCE for count in idle.values()):
            stop_reason = "all stages retired"
            break
        if round_value-best_value <= min_improvement:
            stop_reason = "improvement threshold"
            break
    return dict(label=label, start_objective=float(start_value), objective=float(best_value),
                active_sites=sorted(_active_sites(best_x)),
                start_active_sites=sorted(_active_sites(start_x)),
                evaluations=spent[0], stop_reason=stop_reason, x=best_x,
                _trajectory=trajectory, _history=history, _stage_counts=stage_counts,
                _archive=archive.entries() if own_archive else None)


def _refine_start_args(args):
    return _refine_start(*args)


def optimise(
    bounds: Optional[Tuple[Tuple[str, float, float, bool], ...]] = None,
    ei_weight: float = 0.0,
    seed: Optional[int] = 0,
    maxiter: int = 60,
    popsize: int = 10,
    tol: float = 0.001,
    workers: int = 1,
    x0: Optional[np.ndarray] = None,
    polish: bool = False,
    spread: float = 0.25,
    progress: bool = False,
    refine: bool = True,
    parameter_overrides=None,
    refinement_budget: int = 8000,
    refinement_rounds: int = 40,
    refinement_starts: int = 4,
    refinement_min_improvement: float = 0.001,
    delivery_margin: float = 0.0,
    _typical_search: bool = False,
) -> OptResult:
    """Differential evolution with workbook bounds, integer trains and feasible refinement.

    workers selects processes (deferred DE updates when parallel). x0 may be an
    unverified historical vector. spread controls population variation around it.
    Coordinate refinement crosses discrete train and dispatch boundaries; optional
    scipy polish is normally unhelpful on these discontinuities.
    Refinement repeats site closures, PEM reductions, share transfers and
    coordinate moves until the objective improvement stalls or its round or
    evaluation budget is reached, and runs independently from refinement_starts
    designs: the incumbent plus the best archived design of each other site
    footprint DE saw. The best refined design wins; the spread across starts is
    recorded in search_settings as refinement_spread. refinement_budget applies
    per start. search_settings retains stage history and stop reason.

    delivery_margin m holds production headroom against technical shortfalls:
    every candidate must stay hourly feasible with the target raised by m, and
    is costed at the contracted target (see _evaluate). It sizes the decisions
    searched here -- renewables, battery, storage, PEM and synthesis -- so it is
    Australian production headroom; the cracker and carrier fleets follow the
    contracted target as before.

    result.candidates is a bounded, diversity-preserving sample of the feasible
    designs the search actually evaluated (see _CandidateArchive), plus the
    incumbent trajectory -- the vector at each point the running best strictly
    improved. Both cost nothing extra: they reuse objective values the search
    already computed and never change what DE or refinement explores next, so
    the returned optimum is identical whether or not a caller inspects them.
    Use evaluate_candidates() to attach delivered cost/EI/footprint to entries.
    """
    from scipy.optimize import differential_evolution
    if (not isinstance(refinement_budget, int) or not isinstance(refinement_rounds, int)
            or not isinstance(refinement_starts, int)
            or refinement_budget < 0 or refinement_rounds < 1 or refinement_starts < 1
            or not math.isfinite(refinement_min_improvement) or refinement_min_improvement < 0):
        raise ValueError("Invalid refinement budget, rounds, starts or improvement threshold")
    if not (math.isfinite(delivery_margin) and 0 <= delivery_margin < 1):
        raise ValueError("delivery_margin must be a fraction in [0, 1)")
    if delivery_margin:
        parameter_overrides = dict(parameter_overrides or {}, _delivery_margin=delivery_margin)
    started = time.perf_counter()
    history = []
    require_current_inputs()
    fingerprint = model_fingerprint()

    all_bounds = list(DEFAULT_BOUNDS if bounds is None else bounds)
    if [b[0] for b in all_bounds] != [b[0] for b in DEFAULT_BOUNDS]:
        raise ValueError(
            "Custom bounds must retain the current decision names and order"
        )
    for (name, lo, hi, _), (_, excel_lo, excel_hi, _) in zip(
        all_bounds, DEFAULT_BOUNDS
    ):
        if not (
            math.isfinite(lo) and math.isfinite(hi) and excel_lo <= lo <= hi <= excel_hi
        ):
            raise ValueError(
                f"{name}: custom bounds must stay within the Excel Min and Max"
            )

    bnds = [(lo, hi) for _, lo, hi, _ in all_bounds]
    names = [nm for nm, _, _, _ in all_bounds]
    integrality = [is_int for _, _, _, is_int in all_bounds]

    archive = _CandidateArchive(all_bounds)
    # Every time the running best strictly improves, keep its vector too -- not
    # just its objective (history already has that) -- so a caller can tell
    # whether successive incumbents crept within one site footprint or jumped
    # between different ones.
    incumbent_trajectory = []

    def record_improvement(stage, x, value):
        incumbent_trajectory.append((stage, float(value), np.asarray(x, dtype=float).copy()))

    if x0 is None:
        x0 = _default_x0()
    x0 = _clip_x0_to_bounds(x0, bnds)
    x0[integrality] = np.rint(x0[integrality])
    # A known feasible incumbent survives failed searches and discrete polishing.
    incumbent, incumbent_value = x0.copy(), _evaluate(x0, ei_weight, parameter_overrides)
    archive.offer(x0, incumbent_value)
    record_improvement("initial", incumbent, incumbent_value)
    typical = _clip_x0_to_bounds(x0 if _typical_search else _default_x0(), bnds)
    typical[integrality] = np.rint(typical[integrality])
    typical_value = _evaluate(typical, ei_weight, parameter_overrides)
    archive.offer(typical, typical_value)
    if typical_value < incumbent_value:
        incumbent, incumbent_value = typical, typical_value
        record_improvement("typical seed", incumbent, incumbent_value)
    population = _search_population(incumbent, bnds, popsize, seed, spread)
    resource_value = _evaluate(population[1], ei_weight, parameter_overrides)
    archive.offer(population[1], resource_value)
    if resource_value < incumbent_value:
        incumbent, incumbent_value = population[1].copy(), resource_value
        record_improvement("resource seed", incumbent, incumbent_value)
    population[0], population[1] = x0, incumbent

    def record(stage, evaluations, value, feasible_members=None, population_size=None,
               elapsed=None):
        econ = (parameter_overrides or {}).get("econ", _base_params()["econ"])
        history.append(dict(stage=stage, elapsed_seconds=time.perf_counter()-started
                            if elapsed is None else elapsed,
                            evaluations=int(evaluations), objective=float(value),
                            cost_aud_per_kg_h2=float(value/econ.aud_usd_exchange_rate)
                            if ei_weight == 0 and value < 1e6 else None,
                            feasible=value < 1e6, feasible_members=feasible_members,
                            population_size=population_size))

    record("initial", 3, incumbent_value)
    generation = 0
    def report(intermediate_result):
        nonlocal generation, incumbent, incumbent_value
        generation += 1
        if intermediate_result.fun < incumbent_value:
            incumbent, incumbent_value = intermediate_result.x.copy(), float(intermediate_result.fun)
            record_improvement(f"DE generation {generation}", incumbent, incumbent_value)
        scores = getattr(intermediate_result, "population_energies", [])
        # scipy hands back the live population in real parameter space alongside
        # its energies; archiving it costs no extra objective evaluations.
        for member, value in zip(getattr(intermediate_result, "population", ()), scores):
            archive.offer(member, value)
        feasible = int(np.count_nonzero(np.asarray(scores) < 1e6))
        record(f"DE generation {generation}", 3 + (generation+1)*len(population),
               incumbent_value, feasible, len(scores))
        if progress:
            print(f"  Generation {generation}: objective {intermediate_result.fun:.6g}; "
                  f"{feasible}/{len(scores)} feasible members", flush=True)

    res = differential_evolution(
        _evaluate,
        bounds=bnds,
        args=(ei_weight, parameter_overrides),
        seed=seed,
        maxiter=maxiter,
        popsize=popsize,
        tol=tol,
        workers=workers,
        polish=polish,
        # 'deferred' is required when workers != 1; also the only mode that
        # respects integrality without breaking the multiprocessing pool.
        updating="deferred" if workers != 1 else "immediate",
        integrality=integrality,
        init=population,
        x0=x0,
        callback=report,
    )

    archive.offer(res.x, res.fun)
    if res.fun < incumbent_value:
        incumbent, incumbent_value = res.x.copy(), float(res.fun)
        record_improvement("DE complete", incumbent, incumbent_value)
    if incumbent_value >= 1e6:
        error = ValueError("No hourly feasible design found within this search budget; no optimum saved")
        error.diagnostic_x = incumbent.copy()
        raise error
    record("DE complete", 3 + res.nfev, incumbent_value)
    refinement_evaluations = 0
    stage_counts = {}

    seed_evaluations = 0
    if refine and refinement_starts > 1:
        seed_evaluations = _seed_footprints(
            incumbent, bnds, ei_weight, archive,
            lambda trial, weight, overrides: _evaluate(trial, weight, parameter_overrides))
        refinement_evaluations += seed_evaluations
    starts = _refinement_starts(archive, incumbent, incumbent_value, refinement_starts)
    distinct_footprints = len({_active_sites(sx) for sx, _, _ in starts})
    top_ups = sum(origin == "top-up" for _, _, origin in starts)
    if progress:
        print(f"  Refinement starts: {len(starts)} of {refinement_starts} requested, "
              f"{distinct_footprints} distinct footprints, {top_ups} same-footprint top-ups "
              f"(seeded with {seed_evaluations} closures)", flush=True)
        if len(starts) < refinement_starts:
            print(f"  Archive ran out of non-duplicate designs: "
                  f"{refinement_starts-len(starts)} start(s) dropped", flush=True)
    labels, counts = [], {}
    for _, _, origin in starts:
        counts[origin] = counts.get(origin, 0) + 1
        labels.append("Incumbent (DE best)" if origin == "incumbent"
                      else f"Distinct footprint {counts[origin]}" if origin == "distinct footprint"
                      else f"Top-up {counts[origin]} (same footprint)")
    origins = [origin for _, _, origin in starts]
    common = (all_bounds, ei_weight, parameter_overrides, refine, refinement_budget,
              refinement_rounds, refinement_min_improvement, progress)
    pool_size = min(len(starts), (os.cpu_count() or 1) if workers < 0 else workers)
    # Starts are independent, so they run side by side when DE had workers. Each
    # worker keeps its own memo, so a move another start already priced costs
    # budget there; serially one memo spans every start, as before.
    start_results, start_offsets = [], []
    if pool_size > 1:
        if progress:
            print(f"  Refining {len(starts)} starts on {pool_size} processes", flush=True)
        pool_started = time.perf_counter() - started
        with ProcessPoolExecutor(max_workers=pool_size) as executor:
            start_results = list(executor.map(_refine_start_args, [
                (sx, sv, label) + common for (sx, sv, _), label in zip(starts, labels)]))
        start_offsets = [pool_started] * len(start_results)
    else:
        # One memo across every start: the starts sit in overlapping regions and
        # each stage re-evaluates the vector the previous stage ended on. A repeat
        # costs nothing, so it must not consume budget either.
        memo = {}
        for (start_x, start_value, _), label in zip(starts, labels):
            start_offsets.append(time.perf_counter() - started)
            start_results.append(_refine_start(start_x, start_value, label, *common,
                                               memo=memo, archive=archive))
    for result, offset, origin in zip(start_results, start_offsets, origins):
        result["origin"] = origin
        for stage, spent, value, seconds in result.pop("_history"):
            record(stage, 3 + res.nfev + refinement_evaluations + spent, value,
                   elapsed=offset + seconds)
        refinement_evaluations += result["evaluations"]
        incumbent_trajectory.extend(result.pop("_trajectory"))
        for stage, count in result.pop("_stage_counts").items():
            stage_counts[stage] = stage_counts.get(stage, 0) + count
        for value, x, _ in result.pop("_archive") or ():
            archive.offer(x, value)
    best_start = min(start_results, key=lambda r: r["objective"])
    final_x, best_value = best_start["x"].copy(), best_start["objective"]
    stop_reason = best_start["stop_reason"]
    refinement_spread = [{k: v for k, v in r.items() if k != "x"} for r in start_results]
    refined_starts = [dict(r, x=[float(v) for v in r["x"]]) for r in start_results]
    archive.offer(final_x, best_value)
    # final_x/best_value already equal whichever trajectory entry produced them
    # (the last DE improvement if refinement never bettered it, else the last
    # refinement improvement), so no separate "final" entry is needed here.
    x_named = dict(zip(names, [float(v) for v in final_x]))
    chain = build_supply_chain(list(final_x), parameter_overrides)
    sc_res = chain.run()
    from .validation import validate_traces
    validate_traces(sc_res, chain.dispatch_traces())
    if delivery_margin:
        # Strict run: raises if the saved build cannot carry its headroom.
        build_supply_chain(list(final_x), _margin_overrides(parameter_overrides, delivery_margin)).run()
    record("final verified", 3 + res.nfev + refinement_evaluations, best_value)
    if model_fingerprint() != fingerprint:
        raise ValueError("Model or inputs changed during optimisation; restart before saving a build")

    candidates = [dict(objective=value, active_sites=sorted(sig), x=list(x), source="archive")
                  for value, x, sig in archive.entries()]
    candidates += [dict(objective=value, active_sites=sorted(_active_sites(x)), x=list(x),
                        source="trajectory", stage=stage, step=i)
                   for i, (stage, value, x) in enumerate(incumbent_trajectory)]

    return OptResult(
        success=bool(res.success),
        message=str(res.message),
        x=x_named,
        wind_region=sc_res.wind_region,
        objective=sc_res.delivered_cost_usd_per_kg_h2 + ei_weight*sc_res.delivered_ei_kg_co2_per_kg_h2,
        delivered_cost_aud_per_kg_h2=sc_res.delivered_cost_aud_per_kg_h2,
        delivered_cost_usd_per_kg_h2=sc_res.delivered_cost_usd_per_kg_h2,
        delivered_ei_kg_co2_per_kg_h2=sc_res.delivered_ei_kg_co2_per_kg_h2,
        pem_lcoh_aud_per_kg_h2=sc_res.pem_lcoh_aud_per_kg_h2,
        n_pem=sc_res.n_pem,
        n_synth=sc_res.n_synth,
        n_cracker=sc_res.n_cracker,
        solar_capacity_mw=sc_res.solar_capacity_mw,
        wind_capacity_mw=sc_res.wind_capacity_mw,
        battery_energy_capacity_mwh=sc_res.battery_energy_capacity_mwh,
        battery_power_capacity_mw=sc_res.battery_power_capacity_mw,
        wind_curtailment_fraction=sc_res.wind_curtailment_fraction,
        h2_storage_capacity_kg=sc_res.h2_storage_capacity_kg,
        region_renewables_capex_multiplier=sc_res.region_renewables_capex_multiplier,
        region_battery_capex_multiplier=sc_res.region_battery_capex_multiplier,
        site_wind_mw=sc_res.site_wind_mw,
        site_solar_mw=sc_res.site_solar_mw,
        site_n_pem=sc_res.site_n_pem,
        site_n_synth=sc_res.site_n_synth,
        site_h2_storage_kg=sc_res.site_h2_storage_kg,
        site_battery_mwh=sc_res.site_battery_mwh,
        site_pem_operating_hours=sc_res.site_pem_operating_hours,
        nfev=int(res.nfev) + refinement_evaluations + 3,
        full_result=sc_res,
        candidates=candidates,
        refined_starts=refined_starts,
        search_settings=dict(ei_weight=ei_weight, seed=seed, maxiter=maxiter, popsize=popsize,
                             tol=tol, workers=workers, polish=polish, spread=spread,
                             site_closure_evaluations=stage_counts.get("site closures", 0),
                             share_transfer_evaluations=stage_counts.get("production transfers", 0),
                             refinement_evaluations=refinement_evaluations,
                             refinement_budget=refinement_budget, refinement_rounds=refinement_rounds,
                             refinement_starts=refinement_starts,
                             refinement_min_improvement=refinement_min_improvement,
                             refinement_stop_reason=stop_reason, refinement_stage_counts=stage_counts,
                             refinement_spread=refinement_spread,
                             refinement_start_footprints=distinct_footprints,
                             refinement_start_top_ups=top_ups,
                             refinement_starts_run=len(starts),
                             refinement_seed_evaluations=seed_evaluations,
                             delivery_margin=delivery_margin,
                             delivery_margin_basis=("build hourly feasible at (1 + margin) x target, "
                                                    "costed at the contracted target; Australian "
                                                    "production headroom only") if delivery_margin else "",
                             search_history=history,
                             co_location_basis="Ten-port production allocation, independent site assets and regional/local storage; fixed workbook route topology",
                             initial_population_policy="sparse feasible-seed perturbations, share transfers with capacity and resource-proportional broad seeds",
                             model_fingerprint=fingerprint,
                             bounds=[list(b) for b in all_bounds]),
    )


def pareto_sweep(
    ei_weights: List[float],
    bounds: Optional[Tuple[Tuple[str, float, float, bool], ...]] = None,
    **opt_kwargs,
) -> List[OptResult]:
    """Repeat the optimisation across a list of EI weights for a Pareto frontier."""
    return [optimise(bounds=bounds, ei_weight=w, **opt_kwargs) for w in ei_weights]


def refinement_table(spread) -> str:
    """Aligned start -> final table, best first, objective in USD/kg H2 (+ EI weight)."""
    rows = sorted(spread, key=lambda e: e["objective"])
    best = rows[0]["objective"]
    width = max(len(e["label"]) for e in rows)
    lines = [f"  {'Origin':<{width}}  {'Sites':>7}  {'Start':>8}  {'Final':>8}  {'Change':>8}  "
             f"{'vs best':>8}  {'Evals':>6}"]
    for e in rows:
        sites = f"{len(e.get('start_active_sites') or [])}->{len(e.get('active_sites') or [])}"
        lines.append(f"  {e['label']:<{width}}  {sites:>7}  {e['start_objective']:>8.4f}  "
                     f"{e['objective']:>8.4f}  {e['objective']-e['start_objective']:>+8.4f}  "
                     f"{e['objective']-best:>+8.4f}  {e.get('evaluations', 0):>6}"
                     + ("  <- best" if e is rows[0] else ""))
    return "\n".join(lines)


def print_result(r: OptResult) -> None:
    """Pretty-print an OptResult to stdout."""
    print(f"Optimiser status: {r.message} (nfev={r.nfev})")
    print(f"Objective:        {r.objective:.4f}")
    print(
        f"Delivered cost:   {r.delivered_cost_aud_per_kg_h2:.4f} AUD/kg H2"
        f"   ({r.delivered_cost_usd_per_kg_h2:.4f} USD/kg H2)"
    )
    print(f"Delivered EI:     {r.delivered_ei_kg_co2_per_kg_h2:.4f} kg CO2/kg H2")
    margin = (r.search_settings or {}).get("delivery_margin", 0.0)
    if margin:
        print(f"Delivery margin:  {margin:.0%} production headroom (feasible at {1+margin:.2f} x target; "
              f"cost at the contracted target)")
    spread = (r.search_settings or {}).get("refinement_spread") or []
    if len(spread) > 1:
        values = sorted(e["objective"] for e in spread)
        print(f"Refinement spread:{values[0]:.4f} to {values[-1]:.4f} across {len(values)} starts "
              f"(gap {values[-1]-values[0]:.4f}); the search is only as converged as this is small")
        print(refinement_table(spread))
    print(f"PEM LCOH:         {r.pem_lcoh_aud_per_kg_h2:.4f} AUD/kg H2")
    print(
        f"Fleet:            PEM {r.n_pem * r.full_result.pem.small_module_mw:,.0f} MW, "
        f"n_synth={r.n_synth}, n_cracker={r.n_cracker}"
    )
    print(
        f"Port:             {r.wind_region}  "
        f"(renewables x{r.region_renewables_capex_multiplier:.2f}, "
        f"battery x{r.region_battery_capex_multiplier:.2f})"
    )
    if r.site_wind_mw or r.site_solar_mw:
        print()
        print("Co-location: independent site builds; routes fixed by workbook.")
        print("Delivered H2 allocation: " + ", ".join(
            f"{port} {share:.1%}" for port, share in r.full_result.site_h2_frac.items()))
        print("Per-site allocation:")
        print(
            f"  {'Site':<20} {'Wind MW':>10} {'Solar MW':>10} {'H2 stor Mkg':>12} {'Batt GWh':>10} {'PEM modules':>22} {'PEM hrs/yr':>11} {'n_synth':>8}"
        )
        for port in PORT_SITES:
            wm = r.site_wind_mw.get(port, 0.0)
            sm = r.site_solar_mw.get(port, 0.0)
            hm = r.site_h2_storage_kg.get(port, 0.0) / 1e6
            bm = r.site_battery_mwh.get(port, 0.0) / 1e3
            np_ = r.full_result.pem_modules(port)
            oh_ = r.site_pem_operating_hours.get(port, 0.0)
            ns_ = r.site_n_synth.get(port, 0)
            if wm > 0 or sm > 0:
                print(
                    f"  {port:<20} {wm:>10,.0f} {sm:>10,.0f} {hm:>12.1f} {bm:>10.1f} {np_:>22} {oh_:>11,.0f} {ns_:>8}"
                )
            else:
                print(
                    f"  {port:<20} {'0':>10} {'0':>10} {'0.0':>12} {'0.0':>10} {'-':>22} {'0':>11} {'0':>8}   (not used)"
                )
    print()
    print("Decision variables at optimum:")
    # Sites snapped below the min-renewables threshold contribute nothing.
    # Show them as 0 across all capacity decisions so the printed vector
    # matches what the model actually built.
    off = {
        p for p in PORT_SITES if p not in r.site_wind_mw and p not in r.site_solar_mw
    }
    for name, val in r.x.items():
        display = val
        for prefix in (
            "wind_mw_",
            "solar_mw_",
            "battery_mwh_",
            "h2_storage_kg_",
            "local_h2_storage_kg_",
        ):
            if name.startswith(prefix) and name[len(prefix) :] in off:
                display = 0.0
                break
        print(f"  {name:<38}{display:>12.2f}")
    print()
    print("Renewable / storage scale (fleet):")
    print(f"  Solar capacity                        {r.solar_capacity_mw:>12.1f} MW")
    print(f"  Wind capacity                         {r.wind_capacity_mw:>12.1f} MW")
    print(
        f"  Battery energy capacity               {r.battery_energy_capacity_mwh:>12.1f} MWh"
    )
    print(
        f"  Battery power capacity                {r.battery_power_capacity_mw:>12.1f} MW"
    )
    print(
        f"  Wind curtailment                      {r.wind_curtailment_fraction * 100:>12.1f} %"
    )
    print(
        f"  H2 storage capacity                   {r.h2_storage_capacity_kg:>12.0f} kg"
    )


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------


def save_optimum(r: OptResult, path: Path) -> None:
    """Serialize the decision vector + minimal metadata to JSON.

    Only the decision vector and a few headline numbers are stored --
    everything else can be recomputed by feeding x back into
    build_supply_chain(x).run(). Fast: one SC eval (~few seconds) vs
    a full DE run (~minutes to hours).
    """
    names = [name for name, _, _, _ in DEFAULT_BOUNDS]
    require_current_inputs()
    if r.search_settings.get("model_fingerprint", model_fingerprint()) != model_fingerprint():
        raise ValueError("Model changed after the search; rerun before saving a build")
    x_ordered = [float(r.x[n]) for n in names]
    if r.full_result is None or not r.full_result.hourly_feasible:
        raise ValueError("Cannot save an infeasible deployment as an optimum")
    # Recompute against current inputs rather than trusting a caller's old flag.
    verified = build_supply_chain(x_ordered).run()
    frozen_chain = SupplyChain(**SupplyChain.fixed_deployment(verified))
    frozen = frozen_chain.run()
    from .validation import close, validate_traces
    validate_traces(frozen, frozen_chain.dispatch_traces())
    close(frozen.delivered_cost_aud_per_kg_h2, verified.delivered_cost_aud_per_kg_h2,
          "Frozen nominal build matches design cost")
    close(verified.delivered_cost_aud_per_kg_h2, r.delivered_cost_aud_per_kg_h2,
          "Saved candidate matches current model")
    environment = {"python": platform.python_version()}
    for package in ("numpy", "scipy", "numba", "pandas", "openpyxl"):
        try:
            environment[package] = version(package)
        except PackageNotFoundError:
            environment[package] = "not installed"
    payload = {
        "model_fingerprint": model_fingerprint(),
        "bounds_names": names,
        "x": x_ordered,
        "deployment": {key: getattr(verified, key) for key in
                       ("site_h2_frac", "site_n_synth", "site_annual_meoh_tpa")},
        "search_settings": r.search_settings,
        "environment": environment,
        "hourly_feasible": True,
        "status": "feasible candidate; global optimality not established",
        "objective": float(r.objective),
        "nfev": int(r.nfev),
        "success": bool(r.success),
        "message": str(r.message),
        "delivered_cost_aud_per_kg_h2": float(r.delivered_cost_aud_per_kg_h2),
        "delivered_cost_usd_per_kg_h2": float(r.delivered_cost_usd_per_kg_h2),
        "delivered_ei_kg_co2_per_kg_h2": float(r.delivered_ei_kg_co2_per_kg_h2),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def load_optimum(path: Path, allow_stale=False) -> Dict:
    """Read a saved optimum JSON. Returns the payload dict; use payload['x']
    with build_supply_chain(x).run() or result_from_saved(path)."""
    payload = json.loads(Path(path).read_text())
    if payload.get("model_fingerprint") != model_fingerprint() and not allow_stale:
        raise ValueError(
            "Saved deployment predates current model, profiles or inputs. Re-run optimisation."
        )
    if payload.get("model_fingerprint") != model_fingerprint() and allow_stale:
        print(f"Warning: loading stale deployment from {path}", flush=True)
    return payload


def result_from_saved(path: Path, allow_stale=False) -> OptResult:
    """Load a saved optimum, re-run the supply chain, and materialise an OptResult.

    Raises ValueError if the saved bounds_names don't match current
    DEFAULT_BOUNDS (the vector layout has changed since the save). allow_stale
    forwards to load_optimum() for callers that only need bounds_names to
    match (e.g. a golden-master harness comparing behaviour across commits
    that legitimately change model_fingerprint by editing .py files).
    """
    payload = load_optimum(path, allow_stale=allow_stale)
    current_names = [name for name, _, _, _ in DEFAULT_BOUNDS]
    saved_names = payload.get("bounds_names", [])
    if saved_names != current_names:
        raise ValueError(
            f"Saved bounds do not match current DEFAULT_BOUNDS. "
            f"Saved {len(saved_names)} vars: {saved_names[:3]}... "
            f"Current {len(current_names)} vars: {current_names[:3]}... "
            f"Re-run the optimiser to regenerate."
        )
    x = payload["x"]
    sc_res = build_supply_chain(x).run()
    return OptResult(
        success=bool(payload["success"]),
        message=str(payload["message"]),
        x=dict(zip(current_names, x)),
        wind_region=sc_res.wind_region,
        objective=float(payload["objective"]),
        delivered_cost_aud_per_kg_h2=sc_res.delivered_cost_aud_per_kg_h2,
        delivered_cost_usd_per_kg_h2=sc_res.delivered_cost_usd_per_kg_h2,
        delivered_ei_kg_co2_per_kg_h2=sc_res.delivered_ei_kg_co2_per_kg_h2,
        pem_lcoh_aud_per_kg_h2=sc_res.pem_lcoh_aud_per_kg_h2,
        n_pem=sc_res.n_pem,
        n_synth=sc_res.n_synth,
        n_cracker=sc_res.n_cracker,
        solar_capacity_mw=sc_res.solar_capacity_mw,
        wind_capacity_mw=sc_res.wind_capacity_mw,
        battery_energy_capacity_mwh=sc_res.battery_energy_capacity_mwh,
        battery_power_capacity_mw=sc_res.battery_power_capacity_mw,
        wind_curtailment_fraction=sc_res.wind_curtailment_fraction,
        h2_storage_capacity_kg=sc_res.h2_storage_capacity_kg,
        region_renewables_capex_multiplier=sc_res.region_renewables_capex_multiplier,
        region_battery_capex_multiplier=sc_res.region_battery_capex_multiplier,
        site_wind_mw=sc_res.site_wind_mw,
        site_solar_mw=sc_res.site_solar_mw,
        site_n_pem=sc_res.site_n_pem,
        site_n_synth=sc_res.site_n_synth,
        site_h2_storage_kg=sc_res.site_h2_storage_kg,
        site_battery_mwh=sc_res.site_battery_mwh,
        site_pem_operating_hours=sc_res.site_pem_operating_hours,
        nfev=int(payload["nfev"]),
        full_result=sc_res,
        search_settings=payload.get("search_settings", {}),
    )


def _replay_candidate(args):
    """Re-run one archived vector; module level so it pickles for a worker pool."""
    x, parameter_overrides = args
    try:
        r = build_supply_chain(x, parameter_overrides).run(allow_infeasible=True)
        return dict(delivered_cost_aud_per_kg_h2=r.delivered_cost_aud_per_kg_h2,
                    delivered_cost_usd_per_kg_h2=r.delivered_cost_usd_per_kg_h2,
                    delivered_ei_kg_co2_per_kg_h2=r.delivered_ei_kg_co2_per_kg_h2,
                    n_pem=r.n_pem, n_synth=r.n_synth, n_cracker=r.n_cracker,
                    wind_capacity_mw=r.wind_capacity_mw, solar_capacity_mw=r.solar_capacity_mw,
                    hourly_feasible=bool(r.hourly_feasible), error="")
    except (ValueError, ArithmeticError, RuntimeError) as exc:
        return dict(delivered_cost_aud_per_kg_h2=float("nan"), delivered_cost_usd_per_kg_h2=float("nan"),
                    delivered_ei_kg_co2_per_kg_h2=float("nan"), n_pem=0, n_synth=0, n_cracker=0,
                    wind_capacity_mw=float("nan"), solar_capacity_mw=float("nan"),
                    hourly_feasible=False, error=str(exc) or type(exc).__name__)


def evaluate_candidates(candidates, parameter_overrides=None, workers=1):
    """Re-run each archived candidate's own vector once, to attach delivered
    cost/EI/footprint that the raw search objective does not carry on its own
    (an ei_weight-weighted objective mixes them; ei_weight=0 keeps only cost).
    One evaluation per candidate -- cheap relative to the search itself -- and
    it never changes the saved optimum; purely descriptive of the archived set.
    """
    require_current_inputs()
    args = [(c["x"], parameter_overrides) for c in candidates]
    if workers > 1 and len(args) > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            details = list(executor.map(_replay_candidate, args))
    else:
        details = [_replay_candidate(a) for a in args]
    return [dict(**c, **d) for c, d in zip(candidates, details)]


def model_fingerprint():
    """Fingerprint of the model inputs: workbook values and hourly profiles.

    Code is not hashed. A code change must reproduce the saved results, which
    the golden-master tests check; an input change makes them stale.
    """
    root = Path(__file__).resolve().parent.parent
    digest = hashlib.sha256(inputs_digest().encode())
    for path in sorted((root / "inputs" / "renewables_ninja").glob("*profile_hourly_ports.csv")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()
