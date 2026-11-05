"""Independent port hubs with closed methanol fuel, carbon and hourly energy balances."""

import math
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Dict

import numpy as np

from .parameters import (
    Economics,
    default_target,
    load_economics,
    load_pem,
    load_synth,
    load_ship,
    load_cracker,
    load_co2_capture,
    load_dac,
    PORT_SITE_INFO,
    PORT_SITES,
)
from .green_h2_pem import (
    PemElectrolyser,
    PemResult,
    life_average_sec,
    module_label,
    module_layout,
    pem_capex_aud,
)
from .costing import scaled_unit_cost
from .parameters import load_routes, load_h2_transport
from .meoh_synthesis import MeohSynthesis, MeohSynthResult, synth_feed_ratios
from .shipping import Shipping, ShipResult
from .meoh_cracking import (
    MeohCracker,
    CrackerResult,
    net_h2_per_t_feed,
    supplemental_meoh_per_t_feed,
    captured_co2_per_t_feed,
    converted_fraction,
)
from .co2_capture import Co2Capture, Co2CaptureResult
from .renewables_dispatch import (
    load_wind_profile,
    load_solar_profile,
    simulate_flex_dispatch,
)
from constants import (
    HOURS_PER_YEAR,
    DAYS_PER_YEAR,
    H2_STOICH_KG_PER_KG_MEOH,
    CO2_STOICH_KG_PER_KG_MEOH,
)

# Fleet-level storage defaults. These are ABSOLUTE capacities for the whole
# integrated site (not per-PEM-unit), since PEM/synth/cracker are independently
# scaled. Sourced from Excel ('battery' and 'pem' groups) so the baseline lives
# next to the rest of the renewable inputs; still overridable via the SupplyChain
# constructor for sweeps.


@dataclass
class SupplyChainResult:
    electricity_supply: str = "dedicated_renewables"
    pem_electricity_ei: float = 0.0
    synth_available_hours_per_year: float = 0.0
    cracker_available_hours_per_year: float = 0.0
    installed_design: dict = field(default_factory=dict)
    equipment_shortfalls: list = field(default_factory=list)
    site_shortfalls: dict = field(default_factory=dict)
    site_storage_annual_aud: dict = field(default_factory=dict)
    annual_balanced: bool = False
    # Per-unit module results. pem and synth start from the first active site's
    # result with fleet-average throughput, energy and cost fields written over;
    # synth capex/FOM/VOM stay that site's unit values. ship_fleet and
    # co2_capture are the first route's result only. Representative, not
    # fleet totals -- use the fleet_*/annual_* fields for those.
    # pem.annual_electricity_cost_aud in an hourly run prices the legacy
    # economics.power_price memo, not the delivered cost.
    pem: PemResult = None
    synth: MeohSynthResult = None
    cracker: CrackerResult = None
    ship_fleet: ShipResult = None
    co2_capture: Co2CaptureResult = None

    # Independent fleet counts (one per stage)
    n_pem: int = 0
    n_synth: int = 0
    n_cracker: int = 0

    # Closure capacity factors (actual_op_hrs / 8760 per stage; same nameplate
    # convention as MeohSynthesis -- nameplate annual is the 8760-basis rate).
    pem_capacity_factor: float = 0.0
    synth_capacity_factor: float = 0.0
    cracker_capacity_factor: float = 0.0

    # Actual operating hours per unit after closure
    pem_actual_op_hrs: float = 0.0
    synth_actual_op_hrs: float = 0.0
    cracker_actual_op_hrs: float = 0.0

    # PEM LCOH
    pem_lcoh_aud_per_kg_h2: float = 0.0

    # LCOMeOH bottom-up breakdown (AUD/kg MeOH)
    lcomeoh_aud_per_kg_meoh: float = 0.0
    h2_feed_aud_per_kg_meoh: float = 0.0
    co2_feed_aud_per_kg_meoh: float = 0.0
    synth_plant_aud_per_kg_meoh: float = 0.0

    lcomeoh_component_aud_per_kg_h2: float = 0.0
    ship_component_aud_per_kg_h2: float = 0.0
    cracker_component_aud_per_kg_h2: float = 0.0
    co2_capture_component_aud_per_kg_h2: float = 0.0
    delivered_cost_aud_per_kg_h2: float = 0.0
    delivered_cost_aud_per_t_h2: float = 0.0

    lcomeoh_component_usd_per_kg_h2: float = 0.0
    ship_component_usd_per_kg_h2: float = 0.0
    cracker_component_usd_per_kg_h2: float = 0.0
    co2_capture_component_usd_per_kg_h2: float = 0.0
    delivered_cost_usd_per_kg_h2: float = 0.0
    delivered_cost_usd_per_t_h2: float = 0.0

    ship_fuel_ei: float = 0.0
    ship_fossil_fuel_ei: float = 0.0
    synth_electricity_ei: float = 0.0
    cracker_vent_ei: float = 0.0
    cracker_electricity_ei: float = 0.0
    cracker_heat_ei: float = 0.0
    co2_capture_ei: float = 0.0
    delivered_ei_kg_co2_per_kg_h2: float = 0.0

    circularity_rate: float = 0.0

    target_h2_tpa: float = 0.0
    required_cracker_meoh_tpa: float = 0.0  # MeOH the cracker fleet must consume
    fleet_h2_delivered_tpa: float = 0.0
    fleet_annual_meoh_tpa: float = 0.0
    fleet_annual_synth_co2_feed_tpa: float = 0.0
    fleet_annual_co2_returned_tpa: float = 0.0
    fleet_annual_co2_makeup_tpa: float = 0.0

    fleet_ship_total: int = 0
    fleet_meoh_synth_capex_aud: float = 0.0
    fleet_cracker_capex_aud: float = 0.0
    fleet_pem_capex_aud: float = 0.0

    # Fleet electricity flows (kWh/yr) -- the demand the renewables must meet
    fleet_pem_electricity_kwh_per_year: float = 0.0
    fleet_synth_electricity_kwh_per_year: float = 0.0
    fleet_total_electricity_kwh_per_year: float = 0.0

    energy_intensity_gj_per_t_h2: float = 0.0

    cost_breakdown_aud_per_kg_h2: Dict[str, float] = field(default_factory=dict)
    cost_breakdown_usd_per_kg_h2: Dict[str, float] = field(default_factory=dict)

    # CO2 balance diagnostics (fleet basis)
    fleet_co2_required_tpa: float = 0.0
    fleet_co2_captured_tpa: float = 0.0
    co2_balance_closure_ratio: float = 0.0
    co2_balance_warning: str = ""

    # Renewable supply (single fleet system; shared between PEM and synth electricity)
    solar_capacity_mw: float = 0.0
    wind_capacity_mw: float = 0.0
    solar_annual_energy_mwh: float = 0.0
    wind_annual_energy_mwh: float = 0.0
    solar_capex_aud: float = 0.0
    wind_capex_aud: float = 0.0
    solar_capex_aud_per_kw_effective: float = 0.0
    wind_capex_aud_per_kw_effective: float = 0.0

    # Synth electricity supplied by the shared renewable system (no separate steam EI)
    annual_synth_electricity_renewable_aud: float = 0.0

    annual_solar_capex_aud: float = 0.0
    annual_wind_capex_aud: float = 0.0
    annual_solar_opex_aud: float = 0.0
    annual_wind_opex_aud: float = 0.0
    annual_renewables_total_aud: float = 0.0

    integrated_renewables_aud_per_kwh: float = 0.0
    # Memo only, from the legacy economics.power_price_usd_per_kwh row; never
    # enters delivered cost.
    ppa_aud_per_kwh: float = 0.0
    annual_ppa_equivalent_aud: float = 0.0

    transmission_loss_fraction: float = 0.0
    transmission_loss_mwh: float = 0.0
    gross_renewable_energy_mwh: float = 0.0
    transmission_capex_aud: float = 0.0
    annual_transmission_capex_aud: float = 0.0
    annual_transmission_opex_aud: float = 0.0

    # Battery balancing (fleet absolute, not per-PEM). Auto-sized by the
    # hourly dispatch so the year is feasible with zero unmet demand.
    battery_energy_capacity_mwh: float = 0.0
    battery_power_capacity_mw: float = 0.0
    battery_capex_aud: float = 0.0
    annual_battery_capex_aud: float = 0.0
    annual_battery_opex_aud: float = 0.0

    # Dispatch diagnostics (hourly wind + solar profiles, port-selected, + battery buffer)
    wind_region: str = ""
    site_wind_mw: Dict[str, float] = field(default_factory=dict)
    site_solar_mw: Dict[str, float] = field(default_factory=dict)
    site_n_pem: Dict[str, int] = field(default_factory=dict)
    site_n_synth: Dict[str, int] = field(default_factory=dict)
    site_h2_storage_kg: Dict[str, float] = field(default_factory=dict)
    site_battery_mwh: Dict[str, float] = field(default_factory=dict)
    # Share of the total H2 demand each site carries: its energy share scaled by
    # what its hourly dispatch can deliver.
    site_h2_frac: Dict[str, float] = field(default_factory=dict)
    idle_site_renewable_mwh: Dict[str, float] = field(default_factory=dict)
    # Full-load-equivalent operating hours per PEM module at each site.
    site_pem_operating_hours: Dict[str, float] = field(default_factory=dict)
    wind_profile_mean_cf: float = 0.0
    solar_profile_mean_cf: float = 0.0
    wind_curtailment_mwh: float = 0.0
    wind_curtailment_fraction: float = 0.0
    battery_throughput_mwh: float = 0.0
    battery_cycles_per_year: float = 0.0
    unmet_demand_mwh: float = 0.0
    pem_flex_op_hrs: float = 0.0
    unmet_h2_kg: float = 0.0
    # NZAu Table 25 CAPEX multipliers actually applied (region-dependent).
    region_renewables_capex_multiplier: float = 1.0
    region_battery_capex_multiplier: float = 1.0

    # H2 storage (fleet absolute, mass buffer between PEM and synth)
    h2_storage_capacity_kg: float = 0.0
    h2_storage_capex_aud: float = 0.0
    annual_h2_storage_capex_aud: float = 0.0

    fleet_pem_h2_tpa: float = 0.0
    fleet_heat_meoh_tpa: float = 0.0
    fleet_h2_fuel_tpa: float = 0.0
    fleet_ship_fuel_meoh_tpa: float = 0.0
    fleet_meoh_shipped_tpa: float = 0.0
    fleet_meoh_loss_tpa: float = 0.0
    fleet_transport_electricity_kwh_per_year: float = 0.0
    h2_transport_capex_aud: float = 0.0
    annual_h2_transport_aud: float = 0.0
    h2_transport_component_aud_per_kg_h2: float = 0.0
    synth_process_ei: float = 0.0
    meoh_loss_ei: float = 0.0
    battery_loss_mwh: float = 0.0
    site_synth_operating_hours: dict = field(default_factory=dict)
    site_annual_meoh_tpa: dict = field(default_factory=dict)
    site_routes: dict = field(default_factory=dict)
    site_h2_transport_annual_aud: dict = field(default_factory=dict)
    # site_h2_storage_kg is the total buffer; these split it by location.
    site_local_h2_storage_kg: dict = field(default_factory=dict)
    site_regional_h2_storage_kg: dict = field(default_factory=dict)
    site_local_h2_storage_type: dict = field(default_factory=dict)
    site_regional_compression_mwh: dict = field(default_factory=dict)
    site_transmission_loss_fraction: dict = field(default_factory=dict)
    site_compression_kwh_per_kg: dict = field(default_factory=dict)
    hourly_feasible: bool = False
    model_basis: str = ""
    cracker_heat_production_aud_per_kg_h2: float = 0.0
    ship_fuel_production_aud_per_kg_h2: float = 0.0
    site_ship_results: dict = field(default_factory=dict)
    site_co2_capture_results: dict = field(default_factory=dict)
    # Site PEM capacity as large plants plus small plants.
    site_pem_large_modules: dict = field(default_factory=dict)
    site_pem_small_modules: dict = field(default_factory=dict)
    pem_large_module_mw: float = 0.0
    pem_small_module_mw: float = 0.0

    def pem_modules(self, port):
        return module_label(self.site_pem_large_modules.get(port, 0), self.pem_large_module_mw,
                            self.site_pem_small_modules.get(port, 0), self.pem_small_module_mw)


def _campaign_phase(schedule, resource):
    """Rotation of the campaign that sits over the most generation.

    Circular cross-correlation via FFT: score[k] is the resource captured when
    the schedule is rolled by k, so argmax is the best start hour.
    """
    n = len(schedule)
    spectrum = np.fft.rfft(np.asarray(resource, dtype=float)) * np.conj(
        np.fft.rfft(schedule)
    )
    return int(np.argmax(np.fft.irfft(spectrum, n)))


def synthesis_schedule(
    annual_meoh_t, units, capacity_tpy, resource=None, mwh_per_t=0.0, min_load=1.0,
    available_hours=HOURS_PER_YEAR, max_rate=None, inventory_limit=None,
):
    """Stagger train shutdowns; fractional hours preserve the exact annual mass.

    A fleet that cannot fill the year leaves idle hours. Where those sit is a
    real choice, and leaving them at calendar hour zero is an arbitrary one that
    strands a single-train site in whichever season the year happens to start
    in. Given a resource profile, rotate the campaign onto the site's strongest
    generation instead.

    mwh_per_t is the generation one tonne of methanol draws at the renewables
    site. With it and min_load below 1 the fleet also follows generation.
    """
    rate = capacity_tpy / HOURS_PER_YEAR
    hours = annual_meoh_t / (units * rate)
    if not 0 < hours <= available_hours*(1+1e-10) or available_hours > HOURS_PER_YEAR:
        raise ValueError("Installed synthesis fleet exceeds available full-load hours")
    n = int(HOURS_PER_YEAR)
    # Every unit runs the same campaign from its own start hour.
    ramp = np.clip(hours - np.arange(n), 0, 1) * rate
    hourly = np.zeros(n)
    available = np.zeros(n)
    available_ramp = np.clip(available_hours-np.arange(n), 0, 1)*rate
    for unit in range(units):
        start = int(unit * HOURS_PER_YEAR / units)
        hourly[start:] += ramp[: n - start]
        hourly[:start] += ramp[n - start :]
        available += np.roll(available_ramp, start)
    # A schedule that already covers every hour has nothing to align.
    if resource is not None and hourly.min() <= 0:
        phase = _campaign_phase(hourly, resource)
        hourly, available = np.roll(hourly, phase), np.roll(available, phase)
    if max_rate is not None:
        available = np.minimum(available, max_rate)
        if available.sum() < annual_meoh_t*(1-1e-10):
            raise ValueError("Installed transmission cannot carry annual synthesis electricity")
        hourly = np.minimum(hourly, available)
        room = available-hourly
        if room.sum() > 0:
            hourly += room*max(0.0, annual_meoh_t-hourly.sum())/room.sum()
    if resource is not None and mwh_per_t > 0 and min_load < 1:
        hourly = _follow_generation(
            hourly, np.asarray(resource, dtype=float), available, mwh_per_t, min_load
        )
    if inventory_limit is not None:
        swing = lambda v: float(np.ptp(np.r_[0.0, np.cumsum(v-v.mean())]))
        if swing(hourly) > inventory_limit + 1e-6:
            flat = available*(annual_meoh_t/available.sum())
            if swing(flat) > inventory_limit + 1e-6:
                raise ValueError("Installed terminal storage cannot buffer synthesis shutdowns")
            lo, hi = 0.0, 1.0
            for _ in range(30):
                mid = (lo+hi)/2
                if swing(flat+mid*(hourly-flat)) <= inventory_limit:
                    lo = mid
                else:
                    hi = mid
            hourly = flat+lo*(hourly-flat)
    return hourly, hours


def _follow_generation(hourly, resource, fleet_rate, mwh_per_t, min_load):
    """Turn synthesis down where generation cannot carry it and make it up later.

    Output falls to what generation supports but not below min_load of the
    scheduled rate, so the battery only carries the gap under that floor. The
    lost tonnes return in hours with surplus generation and spare nameplate, in
    proportion to that room. Annual output is unchanged; when the room is smaller
    than the loss, turndown is scaled back to fit.
    """
    lost = hourly - np.maximum(
        np.minimum(hourly, resource / mwh_per_t), hourly * min_load
    )
    room = np.clip(
        np.minimum(fleet_rate - hourly, resource / mwh_per_t - hourly), 0, None
    )
    moved = min(lost.sum(), room.sum())
    if moved <= 0:
        return hourly
    return hourly - lost * (moved / lost.sum()) + room * (moved / room.sum())


def _line_loss(route, pem):
    loss = route["renewables_to_port_km"] * pem.transmission_losses_percent / 100
    if not 0 <= loss < 1:
        raise ValueError("Transmission losses must be in [0,1)")
    return loss


def _pipeline_capex(transport, peak_kg_per_hour, km, offshore_fraction):
    strings = math.ceil(peak_kg_per_hour / transport.pipeline_capacity_kg_per_hour)
    unit = (
        transport.pipeline_capex_aud_per_km * (1 - offshore_fraction)
        + transport.offshore_pipeline_capex_aud_per_km * offshore_fraction
    )
    return strings * km * unit


def _engineered_storage_cost(kind, working_kg, pem, econ):
    """Installed capex and annual cost of engineered underground storage."""
    if kind == "lined_rock_cavern":
        aud_per_kg = scaled_unit_cost(
            working_kg / 1000,
            pem.engineered_storage_small_t,
            pem.lrc_capex_aud_per_kg_small,
            pem.engineered_storage_large_t,
            pem.lrc_capex_aud_per_kg_large,
        )
        opex = pem.lrc_opex_percent_capex
    else:
        aud_per_kg = pem.underground_pipe_capex_aud_per_kg
        opex = pem.underground_pipe_opex_percent_capex
    capex = working_kg * aud_per_kg
    crf = Economics.crf(econ.discount_rate, pem.engineered_storage_life_years)
    return capex, capex * (crf + opex / 100)


def _meets_demand(dispatch, demand):
    return (
        dispatch.unmet_h2_kg <= max(1, float(demand.sum())) * 1e-8
        and dispatch.unmet_electricity_mwh <= 1e-3
    )


class SupplyChain:
    def __init__(
        self,
        economics=None,
        pem=None,
        synth=None,
        ship=None,
        cracker=None,
        co2_capture=None,
        target=None,
        dac=None,
        site_wind_mw=None,
        site_solar_mw=None,
        site_h2_storage_kg=None,
        site_battery_mwh=None,
        site_local_h2_storage_kg=None,
        site_pem_modules=None,
        site_h2_allocation_weight=None,
        site_synth_modules=None,
        site_storage_costs=None,
        site_routes=None,
        h2_transport=None,
        fixed_design=None,
        preserve_assets=False,
        allow_capacity_overrides=False,
        resize_capacities=False,
    ):
        self.econ = economics or load_economics()
        self.pem = pem or load_pem()
        self.synth = synth or load_synth()
        self.ship = ship or load_ship()
        self.cracker = cracker or load_cracker()
        self.co2_cap = co2_capture or load_co2_capture()
        self.dac = dac or load_dac()
        self.transport = h2_transport or load_h2_transport()
        self.fixed_design = fixed_design
        # Technical uncertainty resize: the continuous ratings (transmission,
        # trunk/regional flow, pipeline compressor, terminal storage) uprate to
        # what a draw needs instead of being capped at the frozen reference. Every equipment count -- PEM modules, synthesis
        # trains, cracker units, carriers -- stays frozen, so the mode
        # measures whether the as-built fleet copes (see _installed_capacity,
        # _size_pem_fleet, _dispatch_ratings, _fixed_annual_limit, _site_plan).
        self.resize_capacities = bool(resize_capacities)
        if self.resize_capacities and not fixed_design:
            raise ValueError("resize_capacities only applies to a frozen (fixed_design) build")
        self.preserve_assets = bool(preserve_assets)
        self.allow_capacity_overrides = bool(allow_capacity_overrides)
        self._evaluating_typical = False
        self._fixed_operation = None
        self.site_storage_costs = dict(site_storage_costs or {})
        if set(self.site_storage_costs)-set(PORT_SITES) or any(
            not np.isfinite(v) or v <= 0 for v in self.site_storage_costs.values()
        ):
            raise ValueError("Site storage unit costs must be positive and finite")
        self.target = target or default_target()
        self.routes = load_routes()
        for key, km in (site_routes or {}).items():
            port, _, field = key.partition(".")
            if (port not in self.routes or field not in ("renewables_to_port_km", "pipeline_distance_km")
                    or not np.isfinite(km) or km < 0):
                raise ValueError(f"Invalid route override {key}={km}")
            self.routes[port][field] = float(km)
        self.site_h2_allocation_weight = site_h2_allocation_weight
        self.site_pem_modules = site_pem_modules
        self.site_synth_modules = site_synth_modules
        maps = [
            site_wind_mw,
            site_solar_mw,
            site_h2_storage_kg,
            site_battery_mwh,
            site_local_h2_storage_kg,
        ]
        if all(v is None for v in maps):
            # Capacities are calculated by the Typical build (see run(), which
            # delegates to solve_typical before any of these maps is read).
            self.preserve_assets = True
        (
            self.site_wind_mw,
            self.site_solar_mw,
            self.site_h2_storage_kg,
            self.site_battery_mwh,
            self.site_local_h2_storage_kg,
        ) = maps
        for mapping in maps:
            if mapping is not None:
                if set(mapping) - set(PORT_SITES):
                    raise ValueError("Unknown site in deployment")
                if any(not np.isfinite(v) or v < 0 for v in mapping.values()):
                    raise ValueError("Site capacities must be finite and non-negative")
        self._trace_inputs = {}
        self._port_loads = {}
        for port, count in (self.site_synth_modules or {}).items():
            if port not in PORT_SITES or not np.isfinite(count) or count < 0 or int(count) != count:
                raise ValueError("Site synthesis modules must be non-negative integers at known sites")
        for port, count in (self.site_pem_modules or {}).items():
            if port not in PORT_SITES or not np.isfinite(count) or count < 0 or int(count) != count:
                raise ValueError("Site PEM modules must be non-negative integers at known sites")

    def _size_pem_fleet(
        self,
        net,
        demand,
        hcap,
        ecap,
        transport_sec,
        allow_infeasible=False,
        local_kg=None,
        port=None,
    ):
        chosen = None if self.site_pem_modules is None else self.site_pem_modules.get(port)
        if self.fixed_design:
            return self._flex_dispatch(self.fixed_design["site_n_pem"][port], net,
                                       demand, hcap, ecap, transport_sec, local_kg, port)
        if chosen is not None:
            n, dispatch = self._flex_dispatch(int(chosen), net, demand, hcap, ecap,
                                              transport_sec, local_kg, port)
            if not allow_infeasible and not _meets_demand(dispatch, demand):
                raise ValueError(
                    f"Infeasible hourly supply: {dispatch.unmet_h2_kg:,.0f} kg H2 and "
                    f"{dispatch.unmet_electricity_mwh:,.3f} MWh process electricity unmet")
            return n, dispatch
        hi, best = self._flex_dispatch(
            None, net, demand, hcap, ecap, transport_sec, local_kg, port
        )
        if not _meets_demand(best, demand):
            if allow_infeasible:
                return hi, best
            raise ValueError(
                f"Infeasible hourly supply: {best.unmet_h2_kg:,.0f} kg H2 and {best.unmet_electricity_mwh:,.3f} MWh process electricity unmet"
            )
        lo = 0
        while hi - lo > 1:
            mid = (hi + lo) // 2
            _, trial = self._flex_dispatch(
                mid, net, demand, hcap, ecap, transport_sec, local_kg, port
            )
            if _meets_demand(trial, demand):
                hi, best = mid, trial
            else:
                lo = mid
        return hi, best

    def _flex_dispatch(self, n, net, demand, hcap, ecap, transport_sec, local_kg, port):
        """Hourly dispatch with n PEM blocks; None runs the most the site can use."""
        sec = life_average_sec(self.pem)
        block_mw = self.pem.small_module_mw
        power = ecap / self.pem.battery_storage_duration_hours
        if n is None:
            n = max(1, math.ceil((max(0, float(np.max(net))) + power) / block_mw))
        key = (port, n, float(demand.sum()))
        if self.fixed_design and key in self._fixed_dispatch_cache:
            return n, self._fixed_dispatch_cache[key]
        if self._fixed_operation is not None:
            raise ValueError("Fixed operating schedule has no matching dispatch")
        dispatch = simulate_flex_dispatch(
            net,
            n * block_mw,
            sec,
            demand,
            hcap,
            ecap,
            power,
            self.pem.battery_round_trip_efficiency_percent / 100,
            local_h2_storage_kg=local_kg,
            **self._compression_inputs(port, transport_sec),
            **self._dispatch_ratings(port),
        )
        if self.fixed_design:
            self._fixed_dispatch_cache[key] = dispatch
        return n, dispatch

    def _dispatch_ratings(self, port):
        options = dict(min_pem_power_mw=self.pem.small_module_mw*self.pem.min_load_percent/100,
                       port_load_mw=self._port_loads[port],
                       regional_delivery_factor=1-_line_loss(self.routes[port], self.pem))
        if not self.fixed_design or self.resize_capacities:
            # Resize mode lets the dispatch run unconstrained on these ratings,
            # exactly like a fresh (non-frozen) build; _installed_capacity then
            # sizes and costs each one to whatever the resulting peak needs.
            return options
        c = self.fixed_design["capacities"]
        kinds = ("trunk_kg_per_hour", "regional_kg_per_hour",
                 "local_injection_kg_per_hour", "regional_injection_kg_per_hour",
                 "pipeline_compressor_mw", "transmission_circuits")
        ratings = [c[f"{port}.{kind}"] for kind in kinds]
        ratings[-1] *= self.pem.transmission_circuit_capacity_mw
        for index, kind in ((2, "local"), (3, "regional")):
            if self.pem.storage_injection_kwh_per_kg > 0:
                ratings[index] = min(ratings[index], c[f"{port}.{kind}_injection_mw"]
                    *1000/self.pem.storage_injection_kwh_per_kg)
        return dict(limits=ratings,
            warm_start_soc_h2=self.fixed_design["reference_soc_h2"][port],
            warm_start_soc_e=self.fixed_design["reference_soc_e"][port], **options)

    def _compression_inputs(self, port, trunk_sec):
        loss = _line_loss(self.routes[port], self.pem)
        return dict(
            trunk_sec=trunk_sec,
            regional_sec=self.routes[port]["pipeline_distance_km"] / 100
            * self.transport.compression_kwh_per_kg_per_100km / (1-loss),
            local_charge_sec=self.pem.storage_injection_kwh_per_kg,
            regional_charge_sec=self.pem.storage_injection_kwh_per_kg / (1-loss),
        )

    def _installed_capacity(self, result, port, kind, needed):
        key = f"{port}.{kind}"
        if self.fixed_design:
            installed = self.fixed_design["capacities"][key]
            # Technical resize mode uprates continuous ratings to whatever this
            # draw needs, rather than treating the frozen reference as a ceiling.
            capacity = max(installed, needed) if self.resize_capacities else installed
        else:
            capacity = needed
        result.installed_design.setdefault("capacities", {})[key] = capacity
        if needed > capacity + max(1, capacity)*1e-8:
            result.equipment_shortfalls.append(
                f"{port}: {kind} requires {needed:.6g}, installed {capacity:.6g}")
        return capacity

    @staticmethod
    def fixed_deployment(result):
        """Physical site capacities, allocation and equipment counts from one run."""
        return dict(
            site_wind_mw=dict(result.site_wind_mw),
            site_solar_mw=dict(result.site_solar_mw),
            site_h2_storage_kg=dict(result.site_regional_h2_storage_kg),
            site_local_h2_storage_kg=dict(result.site_local_h2_storage_kg),
            site_battery_mwh=dict(result.site_battery_mwh),
            fixed_design=deepcopy(result.installed_design),
            preserve_assets=result.installed_design.get("preserve_assets", False),
        )

    def _cost_idle_assets(self, result, amounts):
        p, e = self.pem, self.econ
        for port in PORT_SITES:
            if port in result.site_h2_frac:
                continue
            w = (self.site_wind_mw or {}).get(port, 0)
            s = (self.site_solar_mw or {}).get(port, 0)
            battery = (self.site_battery_mwh or {}).get(port, 0)
            regional = (self.site_h2_storage_kg or {}).get(port, 0)
            local = (self.site_local_h2_storage_kg or {}).get(port, 0)
            if not any((w, s, battery, regional, local)):
                continue
            info = PORT_SITE_INFO[port]
            offshore = port == "PortHastings"
            wcap = w * 1000 * (p.offshore_wind_capex_aud_per_kw if offshore else p.wind_capex_aud_per_kw) * info.renewables_capex_mult
            scap = s * 1000 * p.solar_capex_aud_per_kw * info.renewables_capex_mult
            wa = wcap * Economics.crf(e.discount_rate, p.offshore_wind_life_years if offshore else p.wind_life_years)
            sa = scap * Economics.crf(e.discount_rate, p.solar_life_years)
            wo = w * (p.offshore_wind_opex_aud_per_mw_per_yr if offshore else p.wind_opex_aud_per_mw_per_yr)
            so = s * p.solar_opex_aud_per_mw_per_yr
            amounts["renewables"] += wa + sa + wo + so
            result.wind_capex_aud += wcap
            result.solar_capex_aud += scap
            result.annual_wind_capex_aud += wa
            result.annual_solar_capex_aud += sa
            result.annual_wind_opex_aud += wo
            result.annual_solar_opex_aud += so
            bcap = battery * 1000 * p.battery_capex_aud_per_kwh * info.battery_capex_mult
            ba = bcap * Economics.crf(e.discount_rate, p.battery_life_years)
            bo = bcap * p.battery_opex_percent_capex_per_year / 100
            amounts["battery"] += ba + bo
            result.battery_capex_aud += bcap
            result.annual_battery_capex_aud += ba
            result.annual_battery_opex_aud += bo
            regional_capex = regional * self.site_storage_costs.get(port, info.h2_storage_capex_aud_per_kg)
            local_capex, local_annual = _engineered_storage_cost(info.local_h2_storage_type, local, p, e)
            storage_annual = regional_capex * (Economics.crf(e.discount_rate, p.regional_storage_life_years)
                + p.regional_storage_opex_percent_capex / 100) + local_annual
            amounts["storage"] += storage_annual
            result.h2_storage_capex_aud += regional_capex + local_capex
            result.site_storage_annual_aud[port] = storage_annual
            result.site_wind_mw[port] = w
            result.site_solar_mw[port] = s
            result.site_battery_mwh[port] = battery
            result.site_h2_storage_kg[port] = regional + local
            result.site_regional_h2_storage_kg[port] = regional
            result.site_local_h2_storage_kg[port] = local
            result.site_local_h2_storage_type[port] = info.local_h2_storage_type
            wind_energy = w * float(load_wind_profile(port).sum())
            solar_energy = s * float(load_solar_profile(port).sum())
            result.wind_annual_energy_mwh += wind_energy
            result.solar_annual_energy_mwh += solar_energy
            result.wind_curtailment_mwh += wind_energy + solar_energy
            result.idle_site_renewable_mwh[port] = wind_energy + solar_energy

    def _site_plan(self, port, frac, required_meoh, fuel_ratio, h2_ratio, w, s):
        """Methanol, shipping and hourly load for a site carrying frac of the demand."""
        e, p, t = self.econ, self.pem, self.transport
        route = self.routes[port]
        crack_meoh = required_meoh * frac
        heat_meoh = crack_meoh * fuel_ratio
        sp = replace(self.ship, shipping_distance_nm=route["route_distance_nm"])
        bog = Shipping(sp, e, 1, 0).run().bog_fraction_per_voyage
        if not 0 <= bog < 1:
            raise ValueError("Invalid methanol cargo loss")
        cargo = (crack_meoh + heat_meoh) / (1 - bog)
        captured = crack_meoh*captured_co2_per_t_feed(self.cracker)
        sr = Shipping(sp, e, cargo, 0,
                          self.fixed_design["site_ships"][port] if self.fixed_design else None,
                          annual_co2_shipped_tpa=captured).run()
        bunker = sr.ships_required_fractional * sr.annual_meoh_bunker_per_ship_t
        meoh = cargo + bunker
        nominal = self.synth.available_hours_per_year
        if not 0 < nominal <= HOURS_PER_YEAR:
            raise ValueError("Invalid site synthesis hours")
        ns = math.ceil(meoh / (self.synth.capacity_tpy * nominal / HOURS_PER_YEAR))
        # An explicit module count (site decision or frozen build) is absolute,
        # like site_pem_modules -- not an increment above the ceil-derived
        # minimum -- so a short fleet is reported as unmet, not silently resized.
        explicit_ns = None
        if self.site_synth_modules is not None:
            explicit_ns = self.site_synth_modules.get(port)
        if self.fixed_design:
            explicit_ns = self.fixed_design["site_n_synth"][port]
        if explicit_ns is not None:
            ns = explicit_ns
            site_capacity = ns * self.synth.capacity_tpy * nominal / HOURS_PER_YEAR
            if meoh > site_capacity * (1+1e-9):
                error = ValueError(f"{port}: installed synthesis fleet cannot meet required "
                                    f"throughput ({site_capacity/meoh:.4%} of this site's share)")
                error.achievable_fraction = frac * site_capacity / meoh
                raise error
        gen_w = load_wind_profile(port) * w
        gen_s = load_solar_profile(port) * s
        loss = _line_loss(route, p)
        synth_mwh_per_t = (
            self.synth.sec_electricity_kwh_per_t
            + self.synth.sec_heat_kwh_per_t / self.synth.heat_efficiency
        ) / 1000
        max_rate, inventory_limit = None, None
        if self.fixed_design:
            caps = self.fixed_design["capacities"]
            # Transmission circuits and terminal storage are re-ratable continuous
            # items in technical resize mode (see resize_capacities): they size
            # themselves to whatever this draw needs, via _installed_capacity below,
            # so the frozen caps are not a hard ceiling on the schedule here.
            if not self.resize_capacities:
                if synth_mwh_per_t > 0:
                    max_rate = (caps[f"{port}.transmission_circuits"]
                        * p.transmission_circuit_capacity_mw*(1-loss)/synth_mwh_per_t)
                jp = sr.payload_per_voyage_t
                au = sr.payload_per_voyage_t*(1+bunker/cargo)
                tank_au = caps[f"{port}.terminal_storage_au_t"]
                tank_jp = caps[f"{port}.terminal_storage_jp_t"]
                if au > tank_au*(1+1e-9) or jp > tank_jp*(1+1e-9):
                    raise ValueError(f"{port}: installed terminal storage cannot hold a sailing parcel")
                inventory_limit = max(0.0, tank_au-sr.payload_per_voyage_t*(1+bunker/cargo))
        if self._fixed_operation is not None:
            schedule = self._fixed_operation["schedules"][port]
            hours = self._fixed_operation["result"].site_synth_operating_hours[port]
        else:
            schedule, hours = synthesis_schedule(
                meoh,
                ns,
                self.synth.capacity_tpy,
                gen_w + gen_s,
                synth_mwh_per_t / (1 - loss),
                self.synth.min_load_percent / 100,
                available_hours=nominal, max_rate=max_rate, inventory_limit=inventory_limit,
            )
        aux = schedule * synth_mwh_per_t
        self._port_loads[port] = aux/(1-loss)
        demand = schedule * h2_ratio * 1000
        if self._fixed_operation is not None:
            demand = self._fixed_operation["demands"][port]
        # PEM sits at the renewables site with a trunk line to the port.
        # Local storage is beside PEM; regional storage is reached from the port.
        # Compression follows the actual hourly flows on both routes.
        comp = route["renewables_to_port_km"] / 100 * t.compression_kwh_per_kg_per_100km
        # Only synthesis power crosses the line to the port.
        net = gen_w + gen_s - aux / (1 - loss)
        return (
            crack_meoh,
            heat_meoh,
            bog,
            sr,
            cargo,
            bunker,
            meoh,
            ns,
            gen_w,
            gen_s,
            loss,
            hours,
            aux,
            demand,
            comp,
            net,
        )

    def _site_carries(self, port, frac, required_meoh, fuel_ratio, h2_ratio, w, s):
        """Whether a site's hourly dispatch delivers frac of the total demand.

        The check runs the most PEM the site's generation can use, so the limit is
        energy, storage and firming rather than electrolyser count. The year's
        surplus must also cover the regional compression that dispatch needs.
        """
        hcap = self.site_h2_storage_kg.get(port, 0)
        ecap = self.site_battery_mwh.get(port, 0)
        lcap = (self.site_local_h2_storage_kg or {}).get(port, 0)
        try:
            *_, loss, hours, aux, demand, comp, net = self._site_plan(
                port, frac, required_meoh, fuel_ratio, h2_ratio, w, s
            )
            _, dispatch = self._flex_dispatch(
                self.fixed_design["site_n_pem"][port] if self.fixed_design else None,
                net, demand, hcap + lcap, ecap, comp, lcap, port
            )
        except ValueError:
            return False
        return _meets_demand(dispatch, demand)

    def _fixed_annual_limit(self, port, required_meoh, fuel_ratio, h2_ratio, w, s):
        """Necessary throughput limits before spending time on hourly recourse."""
        d, p = self.fixed_design, self.pem
        route = self.routes[port]
        sp = replace(self.ship, shipping_distance_nm=route["route_distance_nm"])
        bog = Shipping(sp, self.econ, 1, 0).run().bog_fraction_per_voyage
        cargo = required_meoh*(1+fuel_ratio)/(1-bog)
        captured = required_meoh*captured_co2_per_t_feed(self.cracker)
        # Per tonne of cargo; the fleet fraction then covers the busier leg.
        ship = Shipping(sp, self.econ, 1, 0, annual_co2_shipped_tpa=captured/cargo).run()
        meoh = cargo*(1+ship.ships_required_fractional*ship.annual_meoh_bunker_per_ship_t)
        aux = meoh*(self.synth.sec_electricity_kwh_per_t+self.synth.sec_heat_kwh_per_t/self.synth.heat_efficiency)/1000
        h2 = meoh*h2_ratio*1000
        trunk_sec = route["renewables_to_port_km"]/100*self.transport.compression_kwh_per_kg_per_100km
        loss = _line_loss(route, p)
        energy = h2*(life_average_sec(p)+trunk_sec)/1000 + aux/(1-loss)
        caps = d["capacities"]
        # The continuous ratings (trunk/pipeline, terminal storage)
        # are re-ratable in technical resize mode -- they cost themselves to
        # whatever this draw needs, via _installed_capacity, so they are not a
        # ceiling on this port's share. Equipment counts (PEM modules, synthesis
        # trains, carriers) and the frozen renewable footprint always
        # stay hard limits.
        limits = {
            "synthesis availability": d["site_n_synth"][port]*self.synth.capacity_tpy*self.synth.available_hours_per_year/HOURS_PER_YEAR/meoh,
            "carriers": d["site_ships"][port]/(ship.ships_required_fractional*cargo),
        }
        limits["PEM nameplate"] = d["site_n_pem"][port]*p.small_module_mw*HOURS_PER_YEAR/(h2*life_average_sec(p)/1000)
        available = w*load_wind_profile(port).sum()+s*load_solar_profile(port).sum()
        limits["annual renewable energy"] = available/energy
        if not self.resize_capacities:
            limits["trunk pipeline"] = caps[f"{port}.trunk_kg_per_hour"]*HOURS_PER_YEAR/h2
        if not self.resize_capacities:
            if trunk_sec > 0:
                limits["pipeline compressor"] = caps[f"{port}.pipeline_compressor_mw"]*HOURS_PER_YEAR/(h2*trunk_sec/1000)
            if (ship.payload_per_voyage_t*meoh/cargo > caps[f"{port}.terminal_storage_au_t"]*(1+1e-9)
                or ship.payload_per_voyage_t > caps[f"{port}.terminal_storage_jp_t"]*(1+1e-9)):
                limits["terminal sailing parcel"] = 0.0
        reason = min(limits, key=limits.get)
        return min(1.0, max(0.0, limits[reason])), reason

    def _fixed_allocation(self, fracs, required_meoh, fuel_ratio, h2_ratio, site_w, site_s):
        """Move production among built sites; never resize or pool their equipment."""
        if self._fixed_operation is not None:
            return dict(self._fixed_operation["result"].site_h2_frac)
        args = (required_meoh, fuel_ratio, h2_ratio)
        caps, reasons = {}, {}
        for port in fracs:
            caps[port], reasons[port] = self._fixed_annual_limit(
                port, *args, site_w[port], site_s[port])
        def shortfall():
            basis = ("dispatch search supports" if any("hourly" in s for s in reasons.values())
                     else "annual equipment and energy ceilings permit at most")
            achievable = sum(caps.values())
            error = ValueError(f"Infeasible fixed build: {basis} "
                f"{achievable:.4%} of the H2 target; "
                + "; ".join(f"{p}: {reasons[p]} ({caps[p]:.4%} of target)" for p in fracs))
            error.achievable_fraction = achievable
            return error
        if sum(caps.values()) < 1-1e-8:
            raise shortfall()
        def carries(port, fraction):
            return fraction > 0 and self._site_carries(
                port, fraction, *args, site_w[port], site_s[port])
        feasible = {p: fracs[p] <= caps[p]*(1+1e-9) and carries(p, fracs[p]) for p in fracs}
        if all(feasible.values()):
            return fracs
        for port in fracs:
            if carries(port, caps[port]):
                continue
            lo, hi = (fracs[port] if feasible[port] else 0.0), caps[port]
            for _ in range(18):
                mid = (lo+hi)/2
                if carries(port, mid):
                    lo = mid
                else:
                    hi = mid
            caps[port], reasons[port] = lo, "hourly dispatch and storage limits"
        if sum(caps.values()) < 1-1e-8:
            raise shortfall()
        shares = {p: min(fracs[p], caps[p]) for p in fracs}
        room = sum(caps.values())-sum(shares.values())
        gap = 1-sum(shares.values())
        shares = {p: v+gap*(caps[p]-v)/room for p, v in shares.items()}
        if not all(carries(p, v) for p, v in shares.items()):
            error = ValueError("Infeasible fixed build: operational allocation search "
                                "found no verified hourly schedule")
            error.achievable_fraction = sum(v for p, v in shares.items() if carries(p, v))
            raise error
        return shares

    def _regional_compression_mwh(self, port, dispatch):
        return dispatch.regional_compression_mwh * (1-_line_loss(self.routes[port], self.pem))

    def _site_potential(
        self,
        port,
        share,
        required_meoh,
        fuel_ratio,
        h2_ratio,
        w,
        s,
        upper=4.0,
        tol=0.02,
    ):
        """Largest multiple of its energy share, up to upper, a site can carry."""
        args = (required_meoh, fuel_ratio, h2_ratio, w, s)
        *_, gen_w, gen_s, loss, hours, aux, demand, comp, net = self._site_plan(
            port, share, *args
        )
        # Demand is linear in the multiple, so one plan gives the multiple whose
        # annual demand uses all generation. Sites usually stop a little short of
        # it, so step down from there instead of bisecting from zero.
        need = float(demand.sum()) * (life_average_sec(self.pem) + comp) / 1000
        need += float(aux.sum()) / (1 - loss)
        bound = float((gen_w + gen_s).sum()) / need
        hi = min(upper, 1 / share)
        if hi < bound:
            if self._site_carries(port, hi * share, *args):
                return hi
        else:
            hi = bound
        lo, gap = 0.0, tol
        while gap < hi:
            if self._site_carries(port, (hi - gap) * share, *args):
                lo = hi - gap
                break
            hi -= gap
            gap *= 2
        else:
            gap = hi
        while gap > tol:
            gap /= 2
            if self._site_carries(port, (lo + gap) * share, *args):
                lo += gap
        return lo

    def run(self, allow_infeasible=False):
        if (not self.fixed_design
                and not self._evaluating_typical and self.site_h2_allocation_weight is None
                and self.site_pem_modules is None and self.site_synth_modules is None):
            from .typical import solve_typical
            solved = solve_typical(chain=self, allow_infeasible=allow_infeasible)
            return solved.full_result
        if self._fixed_operation is None:
            # An economic draw pre-seeds this cache with the verified reference's
            # dispatch results so its own run reuses them instead of resimulating;
            # a fresh reset here would silently discard that before dispatch runs.
            self._fixed_dispatch_cache = {}
        if self.fixed_design and self.fixed_design.get("electricity_supply", "dedicated_renewables") != (
            "dedicated_renewables"
        ):
            raise ValueError("Installed design uses a different electricity supply configuration")
        r = SupplyChainResult()
        r.synth_available_hours_per_year = self.synth.available_hours_per_year
        r.cracker_available_hours_per_year = self.cracker.available_hours_per_year
        e, p, c, t = self.econ, self.pem, self.cracker, self.transport
        if not 0 <= p.min_load_percent <= 100:
            raise ValueError("PEM minimum load must be in [0,100] percent")
        if not math.isfinite(p.ramp_rate_percent_per_s) or p.ramp_rate_percent_per_s*3600 < 100:
            raise ValueError("PEM ramp rate requires sub-hourly dispatch")
        target = self.target.target_h2_tpa
        if target <= 0 or not np.isfinite(target):
            raise ValueError("Positive finite H2 target required")
        for hours in (c.available_hours_per_year, self.synth.available_hours_per_year):
            if not 0 < hours <= HOURS_PER_YEAR:
                raise ValueError("Operating hours must be in (0,8760]")
        if (
            not 0 <= c.co2_capture_rate <= 1
            or not 0 <= self.co2_cap.cargo_loss_fraction < 1
        ):
            raise ValueError("Invalid CO2 capture or cargo-loss fraction")
        if (
            p.battery_storage_duration_hours <= 0
            or p.transmission_circuit_capacity_mw <= 0
        ):
            raise ValueError("Positive battery duration and circuit rating required")
        h2_ratio, co2_ratio = synth_feed_ratios(self.synth)
        cracker_yield = net_h2_per_t_feed(c)
        fuel_ratio = supplemental_meoh_per_t_feed(c)
        r.target_h2_tpa = target
        r.required_cracker_meoh_tpa = target / cracker_yield
        gross_cracker_yield = H2_STOICH_KG_PER_KG_MEOH*converted_fraction(c)*c.psa_h2_recovery
        net_cracker_capacity = c.capacity_tpd_h2*cracker_yield/gross_cracker_yield
        r.n_cracker = math.ceil(
            target / (net_cracker_capacity * c.available_hours_per_year / 24)
        )
        if self.fixed_design:
            r.n_cracker = self.fixed_design["n_cracker"]
            achievable = r.n_cracker*net_cracker_capacity*c.available_hours_per_year/24
            if target > achievable*(1+1e-9):
                error = ValueError("Installed cracker fleet cannot meet required "
                                    f"throughput ({achievable/target:.4%} of target)")
                error.achievable_fraction = achievable/target
                raise error
        r.cracker_actual_op_hrs = target / (r.n_cracker * net_cracker_capacity) * 24
        r.cracker_capacity_factor = r.cracker_actual_op_hrs / HOURS_PER_YEAR
        r.cracker = MeohCracker(
            replace(c, operating_hours_per_year=r.cracker_actual_op_hrs),
            e,
            net_cracker_capacity,
            net_cracker_capacity / cracker_yield,
            installed_heat_pump_capacity_mw_th=(
                self.fixed_design.get("heat_pump_capacity_mw_th", 0.0) if self.fixed_design else None),
        ).run()
        r.fleet_cracker_capex_aud = r.cracker.capex_aud * r.n_cracker
        r.fleet_h2_fuel_tpa = r.cracker.h2_fuel_tpa * r.n_cracker
        r.fleet_h2_delivered_tpa = target
        r.cracker_component_aud_per_kg_h2 = r.cracker.levelised_cost_aud_per_kg_h2
        r.cracker_vent_ei = (
            r.cracker.vented_co2_tpd
            * (r.cracker_actual_op_hrs / 24)
            * r.n_cracker
            / target
        )
        r.cracker_heat_ei = r.cracker.supplemental_heat_ei_kg_co2_per_kg_h2
        r.cracker_electricity_ei = r.cracker.electricity_ei_kg_co2_per_kg_h2
        if self.site_wind_mw is None and not self.fixed_design:
            raise ValueError(
                "site_wind_mw is required for an hourly, non-frozen run; there is no "
                "single-site screening fallback (a bare SupplyChain() instead delegates "
                "to solve_typical(), which supplies it)"
            )
        site_w = dict(self.site_wind_mw or {})
        site_s = dict(self.site_solar_mw or {})
        # A hub too small to run one synthesis train on its minimum schedule
        # cannot carry any of the demand; switching it on would strand
        # production, so small capacities leave the site unused.
        floor = max(p.min_site_renewables_mw, 1e-9)
        active = [
            port
            for port in PORT_SITES
            if site_w.get(port, 0) + site_s.get(port, 0) >= floor
        ]
        if not active:
            raise ValueError("No active production sites")
        site_w = {port: site_w.get(port, 0) for port in active}
        site_s = {port: site_s.get(port, 0) for port in active}
        energies = {
            port: site_w.get(port, 0) * load_wind_profile(port).sum()
            + site_s.get(port, 0) * load_solar_profile(port).sum()
            for port in active
        }
        fracs = {
            port: energy / sum(energies.values()) for port, energy in energies.items()
        }
        if (
            not self.fixed_design
            and self.site_h2_allocation_weight is None
            and len(active) > 1
            and self.site_h2_storage_kg is not None
            and self.site_battery_mwh is not None
        ):
            # There are no per-site targets. Each hub carries the total demand in
            # proportion to what its hourly dispatch can deliver, and a hub that
            # can carry none of it is left unused.
            demand_args = (r.required_cracker_meoh_tpa, fuel_ratio, h2_ratio)
            potential = {
                port: self._site_potential(
                    port, fracs[port], *demand_args, site_w[port], site_s[port]
                )
                for port in active
            }
            # Feasibility is not monotonic in load (the train count reshapes the
            # schedule), so a scaled-down share is re-checked rather than assumed.
            for _ in range(4):
                carried = sum(fracs[port] * potential[port] for port in active)
                if carried < 1:
                    # The fleet cannot carry the demand; spread the gap by energy.
                    shares = {
                        port: fracs[port] * (potential[port] + 1 - carried)
                        for port in active
                    }
                    break
                shares = {
                    port: fracs[port] * potential[port] / carried for port in active
                }
                short = [
                    port
                    for port in active
                    if shares[port] > 0
                    and not self._site_carries(
                        port, shares[port], *demand_args, site_w[port], site_s[port]
                    )
                ]
                if not short:
                    break
                for port in short:
                    potential[port] = self._site_potential(
                        port,
                        fracs[port],
                        *demand_args,
                        site_w[port],
                        site_s[port],
                        upper=shares[port] / fracs[port],
                    )
            active = [port for port in active if shares[port] > 0]
            site_w = {port: site_w[port] for port in active}
            site_s = {port: site_s[port] for port in active}
            fracs = {port: shares[port] for port in active}
        if not self.fixed_design and self.site_h2_allocation_weight is not None:
            weights = self.site_h2_allocation_weight
            if set(weights)-set(PORT_SITES) or any(not np.isfinite(v) or v < 0 for v in weights.values()):
                raise ValueError("Invalid site production allocation weights")
            if any(weights.get(port, 0) > 0 and port not in active for port in PORT_SITES):
                raise ValueError("Production allocated to a site without renewables")
            active = [port for port in active if weights.get(port, 0) > 0]
            total_weight = sum(weights[port] for port in active)
            if total_weight <= 0:
                raise ValueError("Positive production allocation required")
            fracs = {port: weights[port]/total_weight for port in active}
            site_w = {port: site_w[port] for port in active}
            site_s = {port: site_s[port] for port in active}
        if self.fixed_design:
            fracs = dict(self.fixed_design["site_h2_frac"])
            active = list(fracs)
            site_w = {port: site_w[port] for port in active}
            site_s = {port: site_s[port] for port in active}
            fracs = self._fixed_allocation(fracs, r.required_cracker_meoh_tpa,
                fuel_ratio, h2_ratio, site_w, site_s)
        r.site_wind_mw = site_w
        r.site_solar_mw = site_s
        r.site_h2_frac = fracs
        r.wind_region = "+".join(active)
        amounts = {
            key: 0.0
            for key in (
                "pem_capital",
                "pem_om",
                "pem_stack",
                "pem_water",
                "renewables",
                "battery",
                "storage",
                "transport",
                "synth",
                "co2_capture",
                "co2_makeup",
                "shipping",
                "terminal",
            )
        }
        pem_results = []
        synth_results = []
        ship_results = []
        capture_results = []
        if self.site_h2_storage_kg is None:
            raise ValueError("site_h2_storage_kg is required for an hourly, non-frozen run; "
                              "there is no fleet-level fallback split by production share")
        if self.site_battery_mwh is None:
            raise ValueError("site_battery_mwh is required for an hourly, non-frozen run; "
                              "there is no fleet-level fallback split by production share")
        total_core_energy = 0
        total_h2 = 0
        active_hours_weighted = 0
        self._trace_inputs = {}
        for port in active:
            info = PORT_SITE_INFO[port]
            route = self.routes[port]
            frac = fracs[port]
            hcap = self.site_h2_storage_kg.get(port, 0)
            ecap = self.site_battery_mwh.get(port, 0)
            lcap = (
                0
                if self.site_local_h2_storage_kg is None
                else self.site_local_h2_storage_kg.get(port, 0)
            )
            w, s = site_w.get(port, 0), site_s.get(port, 0)
            for value, limit, name in [
                (w, info.max_wind_mw, "wind"),
                (s, info.max_solar_mw, "solar"),
                (hcap, info.max_h2_storage_kg, "regional H2 storage"),
                (lcap, info.max_local_h2_storage_kg, "local H2 storage"),
                (ecap, info.max_battery_mwh, "battery"),
            ]:
                if value > limit + 1e-6 and not self.allow_capacity_overrides:
                    raise ValueError(
                        f"{port}: {name} capacity exceeds workbook maximum"
                    )
            (
                crack_meoh,
                heat_meoh,
                bog,
                sr,
                cargo,
                bunker,
                meoh,
                ns,
                gen_w,
                gen_s,
                loss,
                hours,
                aux,
                demand,
                comp,
                net,
            ) = self._site_plan(
                port,
                frac,
                r.required_cracker_meoh_tpa,
                fuel_ratio,
                h2_ratio,
                w,
                s,
            )
            ship_results.append(sr)
            r.site_ship_results[port] = sr
            h2 = meoh * h2_ratio * 1000
            syn = MeohSynthesis(
                replace(self.synth, operating_hours_per_year=hours), e
            ).run()
            synth_results.append((syn, ns))
            try:
                npem, dispatch = self._size_pem_fleet(
                    net,
                    demand,
                    hcap + lcap,
                    ecap,
                    comp,
                    allow_infeasible,
                    lcap,
                    port,
                )
            except ValueError as exc:
                raise ValueError(f"{port}: {exc}") from exc
            for key, value in (("reference_soc_h2", dispatch.converged_soc_h2),
                               ("reference_soc_e", dispatch.converged_soc_e)):
                r.installed_design.setdefault(key, {})[port] = (
                    self.fixed_design[key][port] if self.fixed_design else value)
            regional_mwh = self._regional_compression_mwh(port, dispatch)
            regional_bus_mwh = regional_mwh / (1 - loss)
            surplus = dispatch.curtailed_elec_mwh
            core_sec = life_average_sec(p)
            core_energy = dispatch.pem_electricity_mwh
            # An idle fleet at an infeasible candidate still needs finite unit costs.
            flh = max(core_energy / (npem * p.small_module_mw), 1.0) if npem else 1.0
            pr = PemElectrolyser(
                replace(p, operating_hours_per_year=flh),
                e.power_price_aud_per_kwh,
                e.discount_rate,
            ).run()
            pem_results.append((pr, npem))
            pem_mw = npem * p.small_module_mw
            pem_capex = pem_capex_aud(p, pem_mw)
            crf_pem = Economics.crf(e.discount_rate, p.system_life_years)
            amounts["pem_capital"] += pem_capex * crf_pem
            amounts["pem_om"] += pem_capex * p.opex_percent_capex_per_year / 100
            # Stack replacement stays on capex_aud_per_kw: module size moves balance of plant.
            amounts["pem_stack"] += pr.annual_stack_replacement_aud * npem
            amounts["pem_water"] += (
                h2 * p.water_l_per_kg_h2 * p.water_cost_aud_per_kl / 1000
            )
            r.fleet_pem_capex_aud += pem_capex
            r.site_pem_large_modules[port], r.site_pem_small_modules[port] = module_layout(p, pem_mw)
            r.pem_large_module_mw, r.pem_small_module_mw = p.large_module_mw, p.small_module_mw
            # Each site pays its own cost multiplier and connection circuits.
            offshore = port == "PortHastings"
            wcap = (
                w
                * 1000
                * (
                    p.offshore_wind_capex_aud_per_kw
                    if offshore
                    else p.wind_capex_aud_per_kw
                )
                * info.renewables_capex_mult
            )
            scap = s * 1000 * p.solar_capex_aud_per_kw * info.renewables_capex_mult
            wannual = wcap * Economics.crf(
                e.discount_rate,
                p.offshore_wind_life_years if offshore else p.wind_life_years,
            )
            sannual = scap * Economics.crf(e.discount_rate, p.solar_life_years)
            wom = w * (
                p.offshore_wind_opex_aud_per_mw_per_yr
                if offshore
                else p.wind_opex_aud_per_mw_per_yr
            )
            som = s * p.solar_opex_aud_per_mw_per_yr
            circuits = math.ceil(
                dispatch.transmission_peak_mw
                / p.transmission_circuit_capacity_mw - 1e-9
            )
            circuits = self._installed_capacity(r, port, "transmission_circuits", circuits)
            transmission = (
                circuits * route["renewables_to_port_km"] * p.transmission_capex_aud_per_km
            )
            transannual = transmission * crf_pem
            transom = transmission * p.transmission_opex / 100
            amounts["renewables"] += (
                wannual + sannual + wom + som + transannual + transom
            )
            r.wind_capex_aud += wcap
            r.solar_capex_aud += scap
            r.annual_wind_capex_aud += wannual
            r.annual_solar_capex_aud += sannual
            r.annual_wind_opex_aud += wom
            r.annual_solar_opex_aud += som
            r.transmission_capex_aud += transmission
            r.annual_transmission_capex_aud += transannual
            r.annual_transmission_opex_aud += transom
            r.transmission_loss_mwh += (
                float(aux.sum()) * loss / (1 - loss) + regional_bus_mwh - regional_mwh
            )
            bcap = ecap * 1000 * p.battery_capex_aud_per_kwh * info.battery_capex_mult
            ba = bcap * Economics.crf(e.discount_rate, p.battery_life_years)
            bo = bcap * p.battery_opex_percent_capex_per_year / 100
            amounts["battery"] += ba + bo
            r.battery_capex_aud += bcap
            r.annual_battery_capex_aud += ba
            r.annual_battery_opex_aud += bo
            regional_capex = hcap * self.site_storage_costs.get(port, info.h2_storage_capex_aud_per_kg)
            local_capex, local_annual = _engineered_storage_cost(
                info.local_h2_storage_type, lcap, p, e
            )
            local_flow = self._installed_capacity(r, port, "local_injection_kg_per_hour", dispatch.local_charge_peak_kg_per_hour)
            regional_flow = self._installed_capacity(r, port, "regional_injection_kg_per_hour", dispatch.regional_charge_peak_kg_per_hour)
            self._installed_capacity(r, port, "local_injection_mw",
                dispatch.local_charge_peak_kg_per_hour*p.storage_injection_kwh_per_kg/1000)
            self._installed_capacity(r, port, "regional_injection_mw",
                dispatch.regional_charge_peak_kg_per_hour*p.storage_injection_kwh_per_kg/1000)
            extra_storage_capex = 0.0
            for capacity, flow in ((lcap, local_flow), (hcap, regional_flow)):
                blocks = math.ceil(capacity/(p.engineered_storage_large_t*1000))
                included = blocks*p.storage_compressor_ref_flow_kg_per_hour
                extra_storage_capex += max(0, flow-included) / p.storage_compressor_ref_flow_kg_per_hour * p.storage_compressor_ref_capex_aud
            storage_annual = regional_capex * (
                Economics.crf(e.discount_rate, p.regional_storage_life_years)
                + p.regional_storage_opex_percent_capex / 100
            ) + local_annual + extra_storage_capex * (
                Economics.crf(e.discount_rate, p.engineered_storage_life_years)
                + p.regional_storage_opex_percent_capex/100)
            amounts["storage"] += storage_annual
            r.site_storage_annual_aud[port] = storage_annual
            r.h2_storage_capex_aud += regional_capex + local_capex + extra_storage_capex
            trunk_capacity = self._installed_capacity(r, port, "trunk_kg_per_hour",
                math.ceil(dispatch.trunk_peak_kg_per_hour/t.pipeline_capacity_kg_per_hour-1e-9)*t.pipeline_capacity_kg_per_hour)
            # Floating-point inventory noise cannot require a pipe to an absent store.
            regional_peak = dispatch.regional_peak_kg_per_hour if hcap > 0 else 0.0
            regional_capacity = self._installed_capacity(r, port, "regional_kg_per_hour",
                math.ceil(regional_peak/t.pipeline_capacity_kg_per_hour-1e-9)*t.pipeline_capacity_kg_per_hour)
            pipe_capex = _pipeline_capex(
                t, trunk_capacity, route["renewables_to_port_km"], 0)
            if hcap > 0:
                pipe_capex += _pipeline_capex(t, regional_capacity,
                    route["pipeline_distance_km"], route["offshore_fraction"])
            pipeline_peak_mw = dispatch.pipeline_compression_peak_mw
            pipeline_peak_mw = self._installed_capacity(r, port, "pipeline_compressor_mw", pipeline_peak_mw)
            compressor_capex = pipeline_peak_mw*1000*t.compressor_capex_aud_per_kw
            pipeannual = pipe_capex * (
                Economics.crf(e.discount_rate, t.pipeline_life_years)
                + t.pipeline_opex_percent_capex / 100
            ) + compressor_capex * (
                Economics.crf(e.discount_rate, p.engineered_storage_life_years)
                + t.pipeline_opex_percent_capex / 100
            )
            amounts["transport"] += pipeannual
            r.h2_transport_capex_aud += pipe_capex + compressor_capex
            r.fleet_transport_electricity_kwh_per_year += (
                dispatch.transport_electricity_mwh - regional_bus_mwh + regional_mwh
            ) * 1000
            amounts["synth"] += syn.synth_plant_aud_per_kg_meoh * meoh * 1000
            r.fleet_meoh_synth_capex_aud += syn.capex_aud * ns
            captured = crack_meoh * captured_co2_per_t_feed(c)
            cr = Co2Capture(
                self.co2_cap,
                captured,
                target * frac,
                power_ei_kg_co2_per_kwh=c.power_ei_kg_co2_per_kwh,
            ).run()
            returned = cr.annual_co2_returned_tpa
            co2feed = meoh * co2_ratio
            makeup = co2feed - returned
            capture_results.append(cr)
            r.site_co2_capture_results[port] = cr
            amounts["co2_capture"] += cr.annual_cost_aud
            amounts["co2_makeup"] += makeup * self.dac.dac_cost_aud_per_t
            amounts["shipping"] += (
                sr.total_ship_cost_usd_per_kg_meoh
                * cargo
                * 1000
                / e.aud_usd_exchange_rate
            )
            schedule = demand/(h2_ratio*1000)
            seasonal_inventory = float(np.ptp(np.r_[0.0, np.cumsum(schedule-schedule.mean())]))
            # Uniform sailings plus one full parcel at each terminal. Australia
            # also buffers seasonal synthesis variation and internal bunker fuel.
            inventory_au = seasonal_inventory+sr.payload_per_voyage_t*(1+bunker/cargo)
            inventory_jp = sr.payload_per_voyage_t
            if not self.fixed_design:
                # Days of cover sizes the original tanks. Fixed operation is
                # constrained by actual seasonal inventory and sailing parcels.
                inventory_au = max(meoh*self.ship.terminal_storage_days/DAYS_PER_YEAR, inventory_au)
                inventory_jp = max(cargo*self.ship.terminal_storage_days/DAYS_PER_YEAR, inventory_jp)
            inventory_au = self._installed_capacity(r, port, "terminal_storage_au_t", inventory_au)
            inventory_jp = self._installed_capacity(r, port, "terminal_storage_jp_t", inventory_jp)
            inventory = inventory_au+inventory_jp
            inventory = self._installed_capacity(r, port, "terminal_storage_t", inventory)
            terminal_capex = inventory * self.ship.terminal_capex_aud_per_t
            amounts["terminal"] += terminal_capex * (
                Economics.crf(e.discount_rate, self.ship.ship_lifetime_years)
                + self.ship.terminal_opex_percent_capex / 100
            )
            r.ship_fuel_ei += bunker * CO2_STOICH_KG_PER_KG_MEOH / target
            # Fuel-oil carbon enters from outside the loop, so it is kept apart.
            r.ship_fossil_fuel_ei += sr.annual_fossil_fuel_co2_tpa / target
            r.meoh_loss_ei += cargo * bog * CO2_STOICH_KG_PER_KG_MEOH / target
            r.synth_process_ei += (
                meoh * (co2_ratio - CO2_STOICH_KG_PER_KG_MEOH) / target
            )
            # Return-leg ship fuel is already in ship_fuel_ei.
            r.co2_capture_ei += (
                captured - returned + cr.liquefaction_electricity_co2_tpa
            ) / target
            r.fleet_annual_meoh_tpa += meoh
            r.fleet_annual_synth_co2_feed_tpa += co2feed
            r.fleet_annual_co2_returned_tpa += returned
            r.fleet_annual_co2_makeup_tpa += makeup
            r.fleet_co2_captured_tpa += captured
            r.fleet_heat_meoh_tpa += heat_meoh
            r.fleet_ship_fuel_meoh_tpa += bunker
            r.fleet_meoh_shipped_tpa += cargo
            r.fleet_meoh_loss_tpa += cargo * bog
            r.fleet_ship_total += sr.ships_required_integer  # replaced below if pooled
            r.fleet_synth_electricity_kwh_per_year += float(aux.sum()) * 1000
            r.wind_annual_energy_mwh += float(gen_w.sum())
            r.solar_annual_energy_mwh += float(gen_s.sum())
            r.wind_curtailment_mwh += max(0.0, surplus)
            r.battery_throughput_mwh += dispatch.battery_throughput_mwh
            r.battery_loss_mwh += dispatch.battery_loss_mwh
            r.unmet_demand_mwh += dispatch.unmet_electricity_mwh + max(0.0, -surplus)
            r.unmet_h2_kg += dispatch.unmet_h2_kg
            r.site_shortfalls[port] = dict(unmet_h2_kg=dispatch.unmet_h2_kg,
                unmet_electricity_mwh=dispatch.unmet_electricity_mwh+max(0.0,-surplus))
            if dispatch.unmet_h2_kg > h2*1e-8:
                # Generation spilled while H2 went short means timing, not the
                # year's energy, binds.
                r.site_shortfalls[port]["limit"] = ("hourly dispatch and storage limits"
                    if surplus > 1e-3 else "annual renewable energy")
            r.n_pem += npem
            r.n_synth += ns
            total_core_energy += core_energy
            total_h2 += h2
            active_hours_weighted += dispatch.pem_operating_hours * npem
            r.site_n_pem[port] = npem
            r.site_n_synth[port] = ns
            r.site_h2_storage_kg[port] = hcap + lcap
            r.site_local_h2_storage_kg[port] = lcap
            r.site_regional_h2_storage_kg[port] = hcap
            r.site_local_h2_storage_type[port] = info.local_h2_storage_type
            r.site_regional_compression_mwh[port] = regional_mwh
            r.site_transmission_loss_fraction[port] = loss
            r.site_compression_kwh_per_kg[port] = comp
            r.site_battery_mwh[port] = ecap
            r.site_pem_operating_hours[port] = flh
            r.site_synth_operating_hours[port] = hours
            r.site_annual_meoh_tpa[port] = meoh
            r.site_routes[port] = dict(route)
            r.site_h2_transport_annual_aud[port] = pipeannual
            self._trace_inputs[port] = ((
                net,
                npem * p.small_module_mw,
                core_sec,
                demand,
                hcap + lcap,
                ecap,
                ecap / p.battery_storage_duration_hours,
                p.battery_round_trip_efficiency_percent / 100,
                None, None, lcap,
                *self._compression_inputs(port, comp).values(),
            ), self._dispatch_ratings(port))
        if self.ship.pooled_fleet:
            # Ports paid for fractional ship-years; the shared fleet rounds up once.
            charged = sum(s.ships_charged for s in ship_results)
            r.fleet_ship_total = math.ceil(charged - 1e-9)
            amounts["shipping"] += (r.fleet_ship_total - charged) * ship_results[0].annual_cost_per_ship_aud
        preserve_assets = self.preserve_assets or bool(self.fixed_design and self.fixed_design.get("preserve_assets"))
        if preserve_assets:
            self._cost_idle_assets(r, amounts)
            site_w, site_s = r.site_wind_mw, r.site_solar_mw
        r.pem = replace(
            pem_results[0][0],
            small_module_mw=p.small_module_mw,
            annual_energy_mwh=total_core_energy / r.n_pem,
            annual_h2_kg=total_h2 / r.n_pem,
            operating_hours_per_year=total_core_energy / (r.n_pem * p.small_module_mw),
            capacity_factor=total_core_energy
            / (r.n_pem * p.small_module_mw * HOURS_PER_YEAR),
            h2_output_tpd=total_h2 / r.n_pem / 1000 / DAYS_PER_YEAR,
            water_kl_per_year=total_h2 * p.water_l_per_kg_h2 / 1000 / r.n_pem,
            annual_water_cost_aud=amounts["pem_water"] / r.n_pem,
            capex_aud=r.fleet_pem_capex_aud / r.n_pem,
            annual_opex_aud=amounts["pem_om"] / r.n_pem,
            annual_stack_replacement_aud=amounts["pem_stack"] / r.n_pem,
            annual_electricity_cost_aud=total_core_energy
            * 1000
            * e.power_price_aud_per_kwh
            / r.n_pem,
        )
        r.synth = replace(
            synth_results[0][0],
            annual_meoh_tpa=r.fleet_annual_meoh_tpa / r.n_synth,
            capacity_factor=r.fleet_annual_meoh_tpa
            / (r.n_synth * self.synth.capacity_tpy),
            electricity_kwh_per_year=r.fleet_annual_meoh_tpa
            * self.synth.sec_electricity_kwh_per_t
            / r.n_synth,
            heat_kwh_per_year=r.fleet_annual_meoh_tpa
            * self.synth.sec_heat_kwh_per_t
            / r.n_synth,
            synth_plant_aud_per_kg_meoh=amounts["synth"]
            / (r.fleet_annual_meoh_tpa * 1000),
        )
        r.ship_fleet = ship_results[0]
        r.co2_capture = capture_results[0]
        r.pem_actual_op_hrs = total_core_energy / (r.n_pem * p.small_module_mw)
        r.pem_capacity_factor = r.pem_actual_op_hrs / HOURS_PER_YEAR
        r.pem_flex_op_hrs = active_hours_weighted / r.n_pem
        r.synth_actual_op_hrs = (
            r.fleet_annual_meoh_tpa
            / (r.n_synth * self.synth.capacity_tpy)
            * HOURS_PER_YEAR
        )
        r.synth_capacity_factor = r.synth_actual_op_hrs / HOURS_PER_YEAR
        r.fleet_pem_h2_tpa = total_h2 / 1000
        r.fleet_pem_electricity_kwh_per_year = total_core_energy * 1000
        r.fleet_total_electricity_kwh_per_year = (
            total_core_energy * 1000
            + r.fleet_synth_electricity_kwh_per_year
            + r.fleet_transport_electricity_kwh_per_year
        )
        r.solar_capacity_mw = sum(site_s.values())
        r.wind_capacity_mw = sum(site_w.values())
        total_nameplate = r.solar_capacity_mw + r.wind_capacity_mw
        r.region_renewables_capex_multiplier = (
            sum(
                (site_w.get(port, 0) + site_s.get(port, 0))
                * PORT_SITE_INFO[port].renewables_capex_mult
                for port in site_w.keys() | site_s.keys()
            )
            / total_nameplate if total_nameplate else 1.0
        )
        total_battery = sum(r.site_battery_mwh.values())
        r.region_battery_capex_multiplier = (
            sum(
                value * PORT_SITE_INFO[port].battery_capex_mult
                for port, value in r.site_battery_mwh.items()
            )
            / total_battery
            if total_battery
            else 1.0
        )
        r.solar_profile_mean_cf = (
            r.solar_annual_energy_mwh / (r.solar_capacity_mw * HOURS_PER_YEAR)
            if r.solar_capacity_mw
            else 0
        )
        r.wind_profile_mean_cf = (
            r.wind_annual_energy_mwh / (r.wind_capacity_mw * HOURS_PER_YEAR)
            if r.wind_capacity_mw
            else 0
        )
        r.solar_capex_aud_per_kw_effective = (
            r.solar_capex_aud / (r.solar_capacity_mw * 1000)
            if r.solar_capacity_mw
            else 0
        )
        r.wind_capex_aud_per_kw_effective = (
            r.wind_capex_aud / (r.wind_capacity_mw * 1000) if r.wind_capacity_mw else 0
        )
        r.h2_storage_capacity_kg = sum(r.site_h2_storage_kg.values())
        r.battery_energy_capacity_mwh = sum(r.site_battery_mwh.values())
        r.battery_power_capacity_mw = (
            r.battery_energy_capacity_mwh / p.battery_storage_duration_hours
        )
        r.battery_cycles_per_year = (
            r.battery_throughput_mwh / r.battery_energy_capacity_mwh
            if r.battery_energy_capacity_mwh
            else 0
        )
        generated = r.solar_annual_energy_mwh + r.wind_annual_energy_mwh
        r.gross_renewable_energy_mwh = generated
        r.transmission_loss_fraction = r.transmission_loss_mwh / generated if generated else 0.0
        r.wind_curtailment_fraction = r.wind_curtailment_mwh / generated if generated else 0.0
        r.annual_renewables_total_aud = amounts["renewables"]
        r.annual_h2_storage_capex_aud = amounts["storage"]
        r.annual_h2_transport_aud = amounts["transport"]
        r.integrated_renewables_aud_per_kwh = (
            amounts["renewables"] / r.fleet_total_electricity_kwh_per_year
        )
        r.ppa_aud_per_kwh = e.power_price_aud_per_kwh
        r.annual_ppa_equivalent_aud = (
            r.ppa_aud_per_kwh * r.fleet_total_electricity_kwh_per_year
        )
        renew_synth = (
            amounts["renewables"]
            * r.fleet_synth_electricity_kwh_per_year
            / r.fleet_total_electricity_kwh_per_year
        )
        r.annual_synth_electricity_renewable_aud = renew_synth
        pem_total = (
            sum(
                amounts[k]
                for k in (
                    "pem_capital",
                    "pem_om",
                    "pem_stack",
                    "pem_water",
                    "storage",
                    "battery",
                    "transport",
                )
            )
            + amounts["renewables"]
            - renew_synth
        )
        r.pem_lcoh_aud_per_kg_h2 = pem_total / total_h2
        meoh_kg = r.fleet_annual_meoh_tpa * 1000
        r.h2_feed_aud_per_kg_meoh = pem_total / meoh_kg
        r.co2_feed_aud_per_kg_meoh = (
            amounts["co2_capture"] + amounts["co2_makeup"]
        ) / meoh_kg
        r.synth_plant_aud_per_kg_meoh = (amounts["synth"] + renew_synth) / meoh_kg
        r.lcomeoh_aud_per_kg_meoh = (
            r.h2_feed_aud_per_kg_meoh
            + r.co2_feed_aud_per_kg_meoh
            + r.synth_plant_aud_per_kg_meoh
        )
        r.cracker_heat_production_aud_per_kg_h2 = (
            r.lcomeoh_aud_per_kg_meoh * r.fleet_heat_meoh_tpa / target
        )
        r.ship_fuel_production_aud_per_kg_h2 = (
            r.lcomeoh_aud_per_kg_meoh * r.fleet_ship_fuel_meoh_tpa / target
        )
        r.lcomeoh_component_aud_per_kg_h2 = (
            r.lcomeoh_aud_per_kg_meoh * meoh_kg / (target * 1000)
        )
        r.ship_component_aud_per_kg_h2 = (
            amounts["shipping"] + amounts["terminal"]
        ) / (target * 1000)
        r.co2_capture_component_aud_per_kg_h2 = amounts["co2_capture"] / (target * 1000)
        r.h2_transport_component_aud_per_kg_h2 = amounts["transport"] / (target * 1000)
        r.delivered_cost_aud_per_kg_h2 = (
            r.lcomeoh_component_aud_per_kg_h2
            + r.ship_component_aud_per_kg_h2
            + r.cracker_component_aud_per_kg_h2
        )
        for field_name in (
            "lcomeoh_component",
            "ship_component",
            "cracker_component",
            "co2_capture_component",
            "delivered_cost",
        ):
            setattr(
                r,
                field_name + "_usd_per_kg_h2",
                getattr(r, field_name + "_aud_per_kg_h2") * e.aud_usd_exchange_rate,
            )
        r.delivered_cost_aud_per_t_h2 = r.delivered_cost_aud_per_kg_h2 * 1000
        r.delivered_cost_usd_per_t_h2 = r.delivered_cost_usd_per_kg_h2 * 1000
        r.cost_breakdown_aud_per_kg_h2 = {
            key: value / (target * 1000) for key, value in amounts.items()
        }
        r.cost_breakdown_aud_per_kg_h2["cracker"] = r.cracker_component_aud_per_kg_h2
        r.cost_breakdown_aud_per_kg_h2["Total"] = r.delivered_cost_aud_per_kg_h2
        r.cost_breakdown_usd_per_kg_h2 = {
            key: value * e.aud_usd_exchange_rate
            for key, value in r.cost_breakdown_aud_per_kg_h2.items()
        }
        r.fleet_co2_required_tpa = r.fleet_annual_synth_co2_feed_tpa
        r.circularity_rate = r.fleet_annual_co2_returned_tpa / r.fleet_co2_required_tpa
        r.co2_balance_closure_ratio = (
            r.fleet_co2_captured_tpa / r.fleet_co2_required_tpa
        )
        r.delivered_ei_kg_co2_per_kg_h2 = sum(
            (
                r.ship_fuel_ei,
                r.ship_fossil_fuel_ei,
                r.meoh_loss_ei,
                r.synth_process_ei,
                r.cracker_vent_ei,
                r.cracker_heat_ei,
                r.cracker_electricity_ei,
                r.co2_capture_ei,
                r.pem_electricity_ei,
                r.synth_electricity_ei,
            )
        )
        # External electricity only: internal fuel conversion is not added a second time.
        r.energy_intensity_gj_per_t_h2 = (
            (
                r.fleet_total_electricity_kwh_per_year
                + r.cracker.annual_electricity_mwh * r.n_cracker * 1000
                + sum(s.liquefaction_electricity_mwh for s in r.site_co2_capture_results.values())*1000
            )
            * 3.6
            / (target * 1000)
        )
        # Each site must also pass its own tolerance, as _meets_demand requires on
        # a strict run; the fleet sum alone let a small site sit 20x over its limit.
        r.hourly_feasible = bool(
            r.unmet_h2_kg <= total_h2 * 1e-8
            and r.unmet_demand_mwh <= 1e-3
            and not r.equipment_shortfalls
            and not any("limit" in s for s in r.site_shortfalls.values())
        )
        r.annual_balanced = bool(r.unmet_h2_kg <= total_h2*1e-8
                                 and r.unmet_demand_mwh <= 1e-3 and not r.equipment_shortfalls)
        r.installed_design.update(site_n_pem=dict(r.site_n_pem), site_n_synth=dict(r.site_n_synth),
            preserve_assets=preserve_assets,
            electricity_supply=r.electricity_supply,
            heat_pump_capacity_mw_th=r.cracker.heat_pump_capacity_mw_th,
            n_cracker=r.n_cracker, site_h2_frac=dict(self.fixed_design["site_h2_frac"] if self.fixed_design else r.site_h2_frac),
            site_ships={p:s.ships_charged for p,s in r.site_ship_results.items()})
        if not r.hourly_feasible:
            r.fleet_h2_delivered_tpa = target * max(0, 1 - r.unmet_h2_kg / total_h2)
        r.model_basis = "hourly supply with per-site constraints and periodic inventories"
        if not r.hourly_feasible:
            r.model_basis = (
                "INFEASIBLE candidate; costs are requested-throughput diagnostics only"
            )
            if not allow_infeasible:
                raise ValueError(f"Infeasible hourly supply: {r.unmet_h2_kg:,.0f} kg H2 and "
                                 f"{r.unmet_demand_mwh:,.3f} MWh process electricity unmet; "
                                 + "; ".join(r.equipment_shortfalls))
        if r.hourly_feasible:
            from .validation import validate_result
            validate_result(r)
        return r

    def dispatch_traces(self):
        """Trace the exact independent site dispatches used in the latest run."""
        from .renewables_dispatch import run_year_flex_trace

        if not self._trace_inputs:
            self.run()
        traces = {}
        for port, args in self._trace_inputs.items():
            trace = run_year_flex_trace(*args[0], **args[1])
            traces[port] = trace
        return traces
