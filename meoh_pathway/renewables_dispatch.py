"""Independent hourly PEM, process-load, hydrogen and battery dispatch.

Negative residual generation is unmet process electricity unless supplied by
the battery. The same kernel produces aggregate results and hourly traces.
Inventories converge to a periodic year; numba is optional acceleration.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from constants import HOURS_PER_YEAR

# Fixed port -> profile-column mapping. Values are (wind_column, solar_column)
# in the renewables_ninja CSVs. `None` marks that a resource is not modelled
# at that port and the corresponding CF profile falls back to zeros.
# Kept in code rather than the Excel because CSV column names are provenance
# metadata, not tunable inputs.
PORT_PROFILE_COLUMNS = {
    "PortHastings": ("PortHastings_StarOfTheSouth", "PortHastings_Gippsland"),
    "PortBonython": ("PortBonython_Nonowie", "PortBonython_Cultana"),
    "Gladstone": ("Gladstone_BoulderCreek", "Gladstone_Aldoga"),
    "HayPoint": ("HayPoint_Collinsville", "AbbotPoint_Collinsville"),
    "Ashburton": ("Ashburton_AshburtonNorthSIA", "Ashburton_AshburtonNorthSIA"),
    "Dampier": ("Dampier_Woodside", "Dampier_RioTintoNgarluma"),
    "PortDarwin": (None, "PortDarwin_DarwinREH"),
    "PortAbbotPoint": ("HayPoint_Collinsville", "AbbotPoint_Collinsville"),
    "PortHedland": ("PortHedland_AREH", "PortHedland_NorthStarJunction"),
    "Newcastle": ("Newcastle_LiverpoolRange", "Newcastle_Liddell"),
}

# Numba-jit the hot inner loop. Falls back to a no-op decorator if numba isn't
# installed (correctness unchanged; ~50x slower). First call pays a ~1-2s
# compile cost, then the jitted function runs in ~200 us for a full year.
try:
    from numba import njit

    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        # Support both @njit and @njit(cache=True) usage patterns.
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def deco(f):
            return f

        return deco


_RENEWABLES_NINJA_DIR = (
    Path(__file__).resolve().parent.parent / "inputs" / "renewables_ninja"
)
_WIND_PROFILE_PATH = _RENEWABLES_NINJA_DIR / "wind_profile_hourly_ports.csv"
_SOLAR_PROFILE_PATH = _RENEWABLES_NINJA_DIR / "solar_profile_hourly_ports.csv"


def _load_profile(path: Path, column: str, kind: str) -> np.ndarray:
    df = pd.read_csv(path, index_col=0)
    if column not in df.columns:
        raise KeyError(
            f"{kind} column '{column}' not in {path.name}. "
            f"Available: {list(df.columns)}"
        )
    arr = df[column].to_numpy(dtype=float)
    if arr.shape[0] != int(HOURS_PER_YEAR):
        raise ValueError(f"Expected {int(HOURS_PER_YEAR)} hours, got {arr.shape[0]}")
    if not np.isfinite(arr).all() or np.any((arr < 0) | (arr > 1)):
        raise ValueError(f"Invalid capacity factors in {path.name}: {column}")
    arr.setflags(write=False)
    return arr


def _port_column(port: str, kind: str) -> Optional[str]:
    if port not in PORT_PROFILE_COLUMNS:
        raise KeyError(
            f"Port '{port}' has no profile mapping. "
            f"Available: {list(PORT_PROFILE_COLUMNS)}"
        )
    return PORT_PROFILE_COLUMNS[port][0 if kind == "wind" else 1]


@lru_cache(maxsize=None)
def load_wind_profile(port: str) -> np.ndarray:
    """Hourly wind CF (fraction) for one port site. Shape (8760,)."""
    column = _port_column(port, "wind")
    if column is None:
        return np.zeros(int(HOURS_PER_YEAR))
    return _load_profile(_WIND_PROFILE_PATH, column, "Wind")


@lru_cache(maxsize=None)
def load_solar_profile(port: str) -> np.ndarray:
    """Hourly solar CF (fraction) for one port site. Shape (8760,)."""
    column = _port_column(port, "solar")
    if column is None:
        return np.zeros(int(HOURS_PER_YEAR))
    return _load_profile(_SOLAR_PROFILE_PATH, column, "Solar")


@dataclass
class FlexDispatchResult:
    pem_electricity_mwh: float
    synth_h2_delivered_kg: float
    unmet_h2_kg: float
    curtailed_elec_mwh: float
    pem_operating_hours: float
    battery_throughput_mwh: float
    battery_cycles: float
    converged_soc_h2: float = 0.0
    converged_soc_e: float = 0.0
    unmet_electricity_mwh: float = 0.0
    pem_full_load_hours: float = 0.0
    h2_produced_kg: float = 0.0
    battery_loss_mwh: float = 0.0
    end_soc_h2: float = 0.0
    end_soc_e: float = 0.0
    # Local-first inventory split: hourly flows on the trunk and regional line.
    trunk_peak_kg_per_hour: float = 0.0
    regional_peak_kg_per_hour: float = 0.0
    regional_charged_kg: float = 0.0
    transport_electricity_mwh: float = 0.0
    regional_compression_mwh: float = 0.0
    local_compression_mwh: float = 0.0
    compression_peak_mw: float = 0.0
    regional_compression_peak_mw: float = 0.0
    local_charge_peak_kg_per_hour: float = 0.0
    regional_charge_peak_kg_per_hour: float = 0.0
    pipeline_compression_peak_mw: float = 0.0
    transmission_peak_mw: float = 0.0


@njit(cache=True)
def _aux_reserve(net, power, eta_c, eta_d):
    # Backward storage requirement protects process loads before discretionary PEM.
    reserve = np.zeros(len(net))
    need = 0.0
    for k in range(2 * len(net) - 1, -1, -1):
        h = k % len(net)
        if k < len(net):
            reserve[h] = need
        if net[h] < 0:
            need += min(-net[h], power) / eta_d
        else:
            need = max(0.0, need - min(net[h], power) * eta_c)
    return reserve


@njit(cache=True)
def _hydrogen_power(made, served, inventory, local, sec, trunk_sec, regional_sec,
                    local_charge_sec, regional_charge_sec):
    end = max(0.0, inventory + made - served)
    before_r = max(0.0, inventory - local)
    change_r = max(0.0, end - local) - before_r
    change_l = min(end, local) - min(inventory, local)
    trunk = max(0.0, served + change_r)
    regional = (abs(change_r) * regional_sec
                + max(0.0, change_r) * regional_charge_sec) / 1000
    local_power = max(0.0, change_l) * local_charge_sec / 1000
    transport = trunk * trunk_sec / 1000 + regional + local_power
    return made * sec / 1000 + transport, transport, regional, local_power, trunk, change_r, change_l


@njit(cache=True)
def _production_for_power(budget, low, high, served, inventory, local, sec,
                          trunk_sec, regional_sec, local_charge_sec, regional_charge_sec):
    # The load is continuous and piecewise linear in production. Interpolate on
    # its exact breakpoints rather than iterating a compressor/dispatch loop.
    left = low
    left_power = _hydrogen_power(left, served, inventory, local, sec, trunk_sec,
                                regional_sec, local_charge_sec, regional_charge_sec)[0]
    for _ in range(5):
        right = high
        for point in (served - inventory + min(inventory, local),
                      served - inventory + local, served):
            if left < point < right:
                right = point
        right_power = _hydrogen_power(right, served, inventory, local, sec, trunk_sec,
                                     regional_sec, local_charge_sec, regional_charge_sec)[0]
        if right_power > budget:
            return left + (right-left) * max(0.0, budget-left_power) / (right_power-left_power)
        if right == high:
            return high
        left, left_power = right, right_power
    return left


@njit(cache=True)
def _production_limits(budget, low, high, served, inventory, local, sec,
                       trunk_sec, regional_sec, local_sec, injection_sec,
                       limits, port_load, delivery_factor, min_pem_mw=0.0):
    """Intersect the piecewise-linear power and equipment constraints."""
    first, last = -1.0, -1.0
    left = low
    for _ in range(4):
        right = high
        for point in (served - inventory + local, served):
            if left < point < right:
                right = point
        a = _hydrogen_power(left, served, inventory, local, sec, trunk_sec,
                            regional_sec, local_sec, injection_sec)
        b = _hydrogen_power(right, served, inventory, local, sec, trunk_sec,
                            regional_sec, local_sec, injection_sec)
        va = (a[0], a[4], abs(a[5]), a[6], a[5],
              (a[4]*trunk_sec + abs(a[5])*regional_sec*delivery_factor)/1000,
              a[2] + port_load)
        vb = (b[0], b[4], abs(b[5]), b[6], b[5],
              (b[4]*trunk_sec + abs(b[5])*regional_sec*delivery_factor)/1000,
              b[2] + port_load)
        lower, upper = 0.0, 1.0
        for j in range(7):
            cap = budget if j == 0 else limits[j-1]
            slope = vb[j] - va[j]
            if slope > 0:
                upper = min(upper, (cap-va[j])/slope)
            elif slope < 0:
                lower = max(lower, (cap-va[j])/slope)
            elif va[j] > cap + 1e-9:
                upper = -1.0
        if upper >= lower:
            if first < 0:
                first = left + lower*(right-left)
            last = left + upper*(right-left)
        if right == high:
            break
        left = right
    if 0 < last*sec/1000 < min_pem_mw-1e-9:
        return (0.0, 0.0) if first <= 0 else (-1.0, -1.0)
    return first, last


@njit(cache=True)
def _run_year_flex(net, nameplate, sec, demand, hcap, ecap, power, eta_c, eta_d,
                   initial_e, initial_h, reserve, local, trunk_sec, regional_sec,
                   local_charge_sec, regional_charge_sec, limits, port_load, delivery_factor,
                   min_pem_mw=0.0):
    trace = np.zeros((len(net), 17))
    _fill_year_flex(trace, net, nameplate, sec, demand, hcap, ecap, power, eta_c,
                    eta_d, initial_e, initial_h, reserve, local, trunk_sec,
                    regional_sec, local_charge_sec, regional_charge_sec,
                    limits, port_load, delivery_factor, min_pem_mw)
    return trace


@njit(cache=True)
def _fill_year_flex(trace, net, nameplate, sec, demand, hcap, ecap, power, eta_c,
                    eta_d, soc_e, soc_h, reserve, local, trunk_sec, regional_sec,
                    local_charge_sec, regional_charge_sec, limits, port_load, delivery_factor,
                    min_pem_mw=0.0):
    for h in range(len(net)):
        gen = max(0.0, net[h])
        aux = max(0.0, -net[h])
        aux_batt = min(aux, power, soc_e * eta_d)
        soc_e -= aux_batt / eta_d
        unmet_e = aux - aux_batt
        remaining_power = max(0.0, power - aux_batt)
        charge = 0.0
        if aux_batt <= 0:
            charge = min(gen, power, max(0.0, min(ecap, reserve[h])-soc_e)/eta_c)
            soc_e += charge * eta_c
            gen -= charge
        max_made = nameplate * 1000 / sec
        served = min(demand[h], soc_h + max_made)
        minimum = max(0.0, served - soc_h)
        budget = gen + (min(remaining_power, max(0.0, soc_e-reserve[h])*eta_d)
                        if charge <= 0 else 0.0)
        if np.isfinite(limits).any() or min_pem_mw > 0:
            maximum = min(max_made, max(0.0, hcap-soc_h+served))
            _, made = _production_limits(budget, minimum, maximum, served, soc_h,
                local, sec, trunk_sec, regional_sec, local_charge_sec,
                regional_charge_sec, limits, port_load[h], delivery_factor, min_pem_mw)
            if made < 0:
                lo, hi = 0.0, served
                for _ in range(35):
                    mid = (lo+hi)/2
                    _, trial = _production_limits(budget, max(0.0, mid-soc_h),
                        min(max_made, hcap-soc_h+mid), mid, soc_h, local, sec,
                        trunk_sec, regional_sec, local_charge_sec, regional_charge_sec,
                        limits, port_load[h], delivery_factor, min_pem_mw)
                    if trial >= 0:
                        lo = mid
                    else:
                        hi = mid
                served = lo
                _, made = _production_limits(budget, max(0.0, served-soc_h),
                    min(max_made, hcap-soc_h+served), served, soc_h, local, sec,
                    trunk_sec, regional_sec, local_charge_sec, regional_charge_sec,
                    limits, port_load[h], delivery_factor, min_pem_mw)
                made = max(0.0, made)
            used, transport, regional, local_power, trunk, change_r, change_l = _hydrogen_power(
                made, served, soc_h, local, sec, trunk_sec, regional_sec,
                local_charge_sec, regional_charge_sec)
            discharge = aux_batt + max(0.0, used-gen)
            soc_e -= max(0.0, used-gen)/eta_d
            gen = max(0.0, gen-used)
            if discharge <= 0:
                extra_charge = min(gen, max(0.0, power-charge), max(0.0, ecap-soc_e)/eta_c)
                charge += extra_charge
                soc_e += extra_charge*eta_c
                gen -= extra_charge
            soc_h = max(0.0, soc_h+made-served)
            # Remove subtraction roundoff before this SOC becomes a replay warm start.
            if -1e-9 < soc_e < 0:
                soc_e = 0.0
            loss = charge*(1-eta_c) + discharge*(1/eta_d-1)
            trace[h] = (made*sec/1000, made, served, soc_h, soc_e, gen, discharge,
                        demand[h]-served, unmet_e, loss, transport, regional,
                        local_power, trunk, change_r, change_l, charge)
            continue
        required = _hydrogen_power(minimum, served, soc_h, local, sec, trunk_sec,
                                   regional_sec, local_charge_sec, regional_charge_sec)[0]
        if required > budget:
            lo, hi = 0.0, served
            for _ in range(40):
                mid = (lo + hi) / 2
                trial = _hydrogen_power(max(0.0, mid-soc_h), mid, soc_h, local,
                                        sec, trunk_sec, regional_sec,
                                        local_charge_sec, regional_charge_sec)[0]
                if trial <= budget:
                    lo = mid
                else:
                    hi = mid
            served = lo
            minimum = max(0.0, served-soc_h)
            required = _hydrogen_power(minimum, served, soc_h, local, sec, trunk_sec,
                                       regional_sec, local_charge_sec, regional_charge_sec)[0]
        mandatory_batt = max(0.0, required-gen)
        soc_e -= mandatory_batt / eta_d
        gen = max(0.0, gen-required)
        discharge = aux_batt + mandatory_batt
        discretionary_batt = 0.0
        if charge <= 0:
            discretionary_batt = min(max(0.0, power-discharge),
                                      max(0.0, soc_e-reserve[h])*eta_d)
        maximum = min(max_made, max(0.0, hcap-soc_h+served))
        made = _production_for_power(required+gen+discretionary_batt, minimum,
                                     maximum, served, soc_h, local, sec, trunk_sec,
                                     regional_sec, local_charge_sec, regional_charge_sec)
        used, transport, regional, local_power, trunk, change_r, change_l = _hydrogen_power(
            made, served, soc_h, local, sec, trunk_sec, regional_sec,
            local_charge_sec, regional_charge_sec)
        extra = max(0.0, used-required)
        extra_batt = max(0.0, extra-gen)
        soc_e -= extra_batt / eta_d
        discharge += extra_batt
        gen = max(0.0, gen-extra)
        if discharge <= 0:
            extra_charge = min(gen, max(0.0, power-charge), max(0.0, ecap-soc_e)/eta_c)
            charge += extra_charge
            soc_e += extra_charge * eta_c
            gen -= extra_charge
        soc_h = max(0.0, soc_h+made-served)
        soc_e = max(0.0, soc_e)
        loss = charge*(1-eta_c) + discharge*(1/eta_d-1)
        trace[h] = (made*sec/1000, made, served, soc_h, soc_e, gen, discharge,
                    demand[h]-served, unmet_e, loss, transport, regional,
                    local_power, trunk, change_r, change_l, charge)
    return trace


@njit(cache=True)
def _periodic_year_flex(
    net, nameplate, sec, demand, hcap, ecap, power, eta, reserve, max_passes, tolerance,
    local, trunk_sec, regional_sec, local_charge_sec, regional_charge_sec,
    limits, port_load, delivery_factor, warm_start_soc_h2, warm_start_soc_e,
    min_pem_mw=0.0,
):
    # One buffer serves every pass; allocating a fresh trace each year cost
    # about as much as the dispatch itself.
    trace = np.zeros((len(net), 17))
    soc_e, soc_h = warm_start_soc_e, warm_start_soc_h2
    for _ in range(max_passes):
        start_e, start_h = soc_e, soc_h
        _fill_year_flex(
            trace,
            net,
            nameplate,
            sec,
            demand,
            hcap,
            ecap,
            power,
            eta,
            eta,
            start_e,
            start_h,
            reserve, local, trunk_sec, regional_sec, local_charge_sec, regional_charge_sec,
            limits, port_load, delivery_factor, min_pem_mw,
        )
        soc_h, soc_e = trace[-1, 3], trace[-1, 4]
        battery_settled = abs(soc_e - start_e) <= max(1.0, ecap) * tolerance
        if battery_settled and abs(soc_h - start_h) <= max(1.0, hcap) * tolerance:
            return trace, start_e, start_h, soc_e, soc_h
        drift = soc_h-start_h
        low, high = min(start_h, trace[:, 3].min()), max(start_h, trace[:, 3].max())
        # Skip repeated drift only within one side of the local/regional split.
        # Crossing that split changes both compression and dispatch decisions.
        floor, ceiling = (local, hcap) if low > local else (0.0, local)
        if battery_settled and drift != 0 and low > floor and high < ceiling:
            room = ceiling-high if drift > 0 else low-floor
            years = room // abs(drift)-2
            if years > 0:
                soc_h += years*drift
    raise ValueError("Annual dispatch did not converge to periodic storage inventory")


def _dispatch_inputs(net, nameplate, sec, demand, hcap, ecap, power, rte):
    net = np.asarray(net, dtype=float)
    demand = np.broadcast_to(np.asarray(demand, dtype=float), net.shape).copy()
    if net.ndim != 1 or len(net) == 0 or not np.isfinite(net).all():
        raise ValueError("Generation must be a finite hourly vector")
    if not np.isfinite(demand).all() or np.any(demand < 0):
        raise ValueError("Hydrogen demand must be finite and non-negative")
    if not all(np.isfinite(v) for v in (nameplate, sec, hcap, ecap, power, rte)):
        raise ValueError("Dispatch inputs must be finite")
    if min(nameplate, hcap, ecap, power) < 0 or sec <= 0 or not 0 < rte <= 1:
        raise ValueError("Invalid dispatch capacity, SEC or efficiency")
    return net, demand


def _ratings_inputs(limits, port_load, net, delivery_factor):
    # Trunk flow, regional flow, local injection, regional injection, pipeline MW,
    # and transmission MW, respectively. Unbuilt cases size these after dispatch.
    limits = np.full(6, np.inf) if limits is None else np.asarray(limits, dtype=float)
    load = np.broadcast_to(np.asarray(0.0 if port_load is None else port_load, dtype=float), net.shape).copy()
    if limits.shape != (6,) or np.isnan(limits).any() or (limits < 0).any():
        raise ValueError("Invalid installed dispatch ratings")
    if not np.isfinite(load).all() or (load < 0).any() or not 0 < delivery_factor <= 1:
        raise ValueError("Invalid port electricity load or delivery factor")
    if np.any(load > limits[5] + 1e-8):
        raise ValueError("Installed transmission cannot supply synthesis electricity")
    return limits, load


def simulate_flex_dispatch(
    gen_at_pem_mw,
    pem_nameplate_mw,
    sec_kwh_per_kg_h2,
    synth_h2_kg_per_hr,
    h2_storage_capacity_kg,
    battery_energy_mwh,
    battery_power_mw,
    round_trip_efficiency,
    max_passes=200,
    soc_tolerance_frac=1e-9,
    local_h2_storage_kg=None,
    trunk_sec=0.0,
    regional_sec=0.0,
    local_charge_sec=0.0,
    regional_charge_sec=0.0,
    limits=None,
    port_load_mw=None,
    regional_delivery_factor=1.0,
    warm_start_soc_h2=0.0,
    warm_start_soc_e=0.0,
    min_pem_power_mw=0.0,
):
    net, demand = _dispatch_inputs(
        gen_at_pem_mw,
        pem_nameplate_mw,
        sec_kwh_per_kg_h2,
        synth_h2_kg_per_hr,
        h2_storage_capacity_kg,
        battery_energy_mwh,
        battery_power_mw,
        round_trip_efficiency,
    )
    local = h2_storage_capacity_kg if local_h2_storage_kg is None else local_h2_storage_kg
    if not np.isfinite(min_pem_power_mw) or not 0 <= min_pem_power_mw <= pem_nameplate_mw:
        raise ValueError("Minimum PEM power must be within installed nameplate")
    if not 0 <= local <= h2_storage_capacity_kg:
        raise ValueError("Local storage must be within total H2 capacity")
    if not 0 <= warm_start_soc_h2 <= h2_storage_capacity_kg:
        raise ValueError("H2 warm start must be within capacity")
    if not 0 <= warm_start_soc_e <= battery_energy_mwh:
        raise ValueError("Battery warm start must be within capacity")
    if any(not np.isfinite(v) or v < 0 for v in
           (trunk_sec, regional_sec, local_charge_sec, regional_charge_sec)):
        raise ValueError("Compression SEC must be finite and non-negative")
    if sec_kwh_per_kg_h2 <= regional_sec:
        raise ValueError("PEM SEC must exceed regional withdrawal SEC")
    eta = np.sqrt(round_trip_efficiency)
    limits, port_load = _ratings_inputs(limits, port_load_mw, net, regional_delivery_factor)
    withdrawal_sec = max(trunk_sec if local > 0 else 0.0,
                         regional_sec if h2_storage_capacity_kg > local else 0.0)
    reserve = _aux_reserve(net-demand*withdrawal_sec/1000, battery_power_mw, eta, eta)
    trace, start_e, start_h, soc_e, soc_h = _periodic_year_flex(
        net,
        pem_nameplate_mw,
        sec_kwh_per_kg_h2,
        demand,
        h2_storage_capacity_kg,
        battery_energy_mwh,
        battery_power_mw,
        eta,
        reserve,
        max_passes,
        soc_tolerance_frac, local, trunk_sec, regional_sec, local_charge_sec, regional_charge_sec,
        limits, port_load, regional_delivery_factor, warm_start_soc_h2, warm_start_soc_e,
        min_pem_power_mw,
    )
    regional_step, trunk_flow = trace[:, 14], trace[:, 13]
    energy = float(trace[:, 0].sum())
    throughput = float(trace[:, 6].sum())
    return FlexDispatchResult(
        pem_electricity_mwh=energy,
        synth_h2_delivered_kg=float(trace[:, 2].sum()),
        unmet_h2_kg=float(trace[:, 7].sum()),
        curtailed_elec_mwh=float(trace[:, 5].sum()),
        pem_operating_hours=float(np.count_nonzero(trace[:, 0] > 1e-9)),
        battery_throughput_mwh=throughput,
        battery_cycles=throughput / battery_energy_mwh if battery_energy_mwh else 0.0,
        converged_soc_h2=float(start_h),
        converged_soc_e=float(start_e),
        unmet_electricity_mwh=float(trace[:, 8].sum()),
        pem_full_load_hours=energy / pem_nameplate_mw if pem_nameplate_mw else 0.0,
        h2_produced_kg=float(trace[:, 1].sum()),
        battery_loss_mwh=float(trace[:, 9].sum()),
        end_soc_h2=float(soc_h),
        end_soc_e=float(soc_e),
        trunk_peak_kg_per_hour=float(max(0.0, trunk_flow.max())),
        regional_peak_kg_per_hour=float(np.abs(regional_step).max()),
        regional_charged_kg=float(regional_step[regional_step > 0].sum()),
        transport_electricity_mwh=float(trace[:, 10].sum()),
        regional_compression_mwh=float(trace[:, 11].sum()),
        local_compression_mwh=float(trace[:, 12].sum()),
        compression_peak_mw=float(trace[:, 10].max()),
        regional_compression_peak_mw=float(trace[:, 11].max()),
        local_charge_peak_kg_per_hour=float(np.maximum(trace[:, 15], 0).max()),
        regional_charge_peak_kg_per_hour=float(np.maximum(trace[:, 14], 0).max()),
        pipeline_compression_peak_mw=float(np.max(
            (trunk_flow*trunk_sec+np.abs(regional_step)*regional_sec*regional_delivery_factor)/1000)),
        transmission_peak_mw=float(np.max(port_load+trace[:, 11])),
    )


def run_year_flex_trace(
    gen_at_pem_mw,
    pem_nameplate_mw,
    sec_kwh_per_kg_h2,
    synth_h2_kg_per_hr,
    h2_storage_capacity_kg,
    battery_energy_mwh,
    battery_power_mw,
    round_trip_efficiency,
    initial_soc_h2=None,
    initial_soc_e=None,
    local_h2_storage_kg=None,
    trunk_sec=0.0,
    regional_sec=0.0,
    local_charge_sec=0.0,
    regional_charge_sec=0.0,
    limits=None,
    port_load_mw=None,
    regional_delivery_factor=1.0,
    warm_start_soc_h2=0.0,
    warm_start_soc_e=0.0,
    min_pem_power_mw=0.0,
):
    net, demand = _dispatch_inputs(
        gen_at_pem_mw,
        pem_nameplate_mw,
        sec_kwh_per_kg_h2,
        synth_h2_kg_per_hr,
        h2_storage_capacity_kg,
        battery_energy_mwh,
        battery_power_mw,
        round_trip_efficiency,
    )
    local = h2_storage_capacity_kg if local_h2_storage_kg is None else local_h2_storage_kg
    if not np.isfinite(min_pem_power_mw) or not 0 <= min_pem_power_mw <= pem_nameplate_mw:
        raise ValueError("Minimum PEM power must be within installed nameplate")
    if not 0 <= local <= h2_storage_capacity_kg:
        raise ValueError("Local storage must be within total H2 capacity")
    if any(not np.isfinite(v) or v < 0 for v in
           (trunk_sec, regional_sec, local_charge_sec, regional_charge_sec)):
        raise ValueError("Compression SEC must be finite and non-negative")
    if sec_kwh_per_kg_h2 <= regional_sec:
        raise ValueError("PEM SEC must exceed regional withdrawal SEC")
    if initial_soc_h2 is not None and not 0 <= initial_soc_h2 <= h2_storage_capacity_kg:
        raise ValueError("Initial H2 inventory must be within capacity")
    if initial_soc_e is not None and not 0 <= initial_soc_e <= battery_energy_mwh:
        raise ValueError("Initial battery inventory must be within capacity")
    if initial_soc_h2 is None or initial_soc_e is None:
        r = simulate_flex_dispatch(
            net,
            pem_nameplate_mw,
            sec_kwh_per_kg_h2,
            demand,
            h2_storage_capacity_kg,
            battery_energy_mwh,
            battery_power_mw,
            round_trip_efficiency,
            local_h2_storage_kg=local_h2_storage_kg, trunk_sec=trunk_sec,
            regional_sec=regional_sec, local_charge_sec=local_charge_sec,
            regional_charge_sec=regional_charge_sec,
            limits=limits, port_load_mw=port_load_mw,
            regional_delivery_factor=regional_delivery_factor,
            warm_start_soc_h2=warm_start_soc_h2, warm_start_soc_e=warm_start_soc_e,
            min_pem_power_mw=min_pem_power_mw,
        )
        initial_soc_h2, initial_soc_e = r.converged_soc_h2, r.converged_soc_e
    eta = np.sqrt(round_trip_efficiency)
    limits, port_load = _ratings_inputs(limits, port_load_mw, net, regional_delivery_factor)
    a = _run_year_flex(
        net,
        pem_nameplate_mw,
        sec_kwh_per_kg_h2,
        demand,
        h2_storage_capacity_kg,
        battery_energy_mwh,
        battery_power_mw,
        eta,
        eta,
        initial_soc_e,
        initial_soc_h2,
        _aux_reserve(net-demand*max(
            trunk_sec if local_h2_storage_kg is None or local_h2_storage_kg > 0 else 0.0,
            regional_sec if local_h2_storage_kg is not None and h2_storage_capacity_kg > local_h2_storage_kg else 0.0
        )/1000, battery_power_mw, eta, eta),
        h2_storage_capacity_kg if local_h2_storage_kg is None else local_h2_storage_kg,
        trunk_sec, regional_sec, local_charge_sec, regional_charge_sec,
        limits, port_load, regional_delivery_factor, min_pem_power_mw,
    )
    return dict(
        min_pem_power_mw=min_pem_power_mw,
        pem_mw=a[:, 0],
        h2_produced_kg=a[:, 1],
        h2_delivered_kg=a[:, 2],
        soc_h2_kg=a[:, 3],
        soc_e_mwh=a[:, 4],
        curtailed_mw=a[:, 5],
        battery_discharge_mw=a[:, 6],
        unmet_h2_kg=a[:, 7],
        unmet_electricity_mwh=a[:, 8],
        battery_loss_mwh=a[:, 9],
        demand_h2_kg=demand,
        transport_mw=a[:, 10], regional_compression_mw=a[:, 11],
        local_compression_mw=a[:, 12], trunk_h2_kg=a[:, 13],
        regional_h2_change_kg=a[:, 14], local_h2_change_kg=a[:, 15],
        battery_charge_mw=a[:, 16],
        initial_soc_h2=initial_soc_h2, initial_soc_e=initial_soc_e,
        net_generation_mw=net,
        round_trip_efficiency=round_trip_efficiency,
        port_load_mw=port_load,
        regional_delivery_factor=regional_delivery_factor,
        pipeline_mw=(a[:, 13]*trunk_sec+np.abs(a[:, 14])*regional_sec*regional_delivery_factor)/1000,
        transmission_mw=port_load+a[:, 11],
        regional_injection_mw=np.maximum(a[:, 14], 0)*regional_charge_sec*regional_delivery_factor/1000,

    )
