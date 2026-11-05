"""MeOH pathway parameters.

All numeric values are loaded from inputs/input_data.xlsx at runtime.
The dataclass definitions name the fields and their types; the Excel
is the single source of truth for every default value.
Each module sheet contains its Inputs table and references. Sensitivity bounds
are read by header name from those tables in meoh_pathway/sensitivity.py.
"""

from dataclasses import dataclass
from pathlib import Path
import math

from base_case.run_base_case import load_grey
from inputs.loader import load_params, load_site_inputs, from_dict, parameter_input, XLSX_PATH

LOCAL_H2_STORAGE_TYPES = ("lined_rock_cavern", "underground_pipe")


@dataclass(frozen=True)
class PortSite:
    """Static per-port site record, joined across the three site sheets."""

    key: str
    max_wind_mw: float
    max_solar_mw: float
    max_battery_mwh: float
    max_h2_storage_kg: float
    renewables_capex_mult: float
    battery_capex_mult: float
    h2_storage_capex_aud_per_kg: float
    max_local_h2_storage_kg: float
    local_h2_storage_type: str


def _local_storage_type(site, value):
    kind = str(value).strip()
    if kind not in LOCAL_H2_STORAGE_TYPES:
        raise ValueError(
            f"{site}.local_h2_storage_type must be one of {LOCAL_H2_STORAGE_TYPES}"
        )
    return kind


def _build_port_site_info(path: Path = XLSX_PATH) -> dict:
    rows = load_site_inputs(path)
    return {
        p: PortSite(
            key=p,
            max_wind_mw=float(r["wind_mw"]["Upper bound"]),
            max_solar_mw=float(r["solar_mw"]["Upper bound"]),
            max_battery_mwh=float(r["battery_mwh"]["Upper bound"]),
            max_h2_storage_kg=float(r["h2_storage_kg"]["Upper bound"]),
            renewables_capex_mult=float(r["renewables_capex_mult"]["Value"]),
            battery_capex_mult=float(r["battery_capex_mult"]["Value"]),
            h2_storage_capex_aud_per_kg=float(
                r["h2_storage_capex_aud_per_kg"]["Value"]
            ),
            max_local_h2_storage_kg=float(r["local_h2_storage_kg"]["Upper bound"]),
            local_h2_storage_type=_local_storage_type(
                p, r["local_h2_storage_type"]["Value"]
            ),
        )
        for p, r in rows.items()
    }


# Site records live beside each module's shared inputs. Field keys join them
# across sheets, so sorting a storage table cannot change the site assignment.
PORT_SITE_INFO = _build_port_site_info()
PORT_SITES = tuple(PORT_SITE_INFO)


@dataclass
class Economics:
    aud_usd_exchange_rate: float
    discount_rate: float
    cracker_life_years: int
    # Legacy memo (workbook power_price_usd_per_kwh): PPA-equivalent reporting
    # and the standalone PEM calculator only, not delivered cost.
    power_price_aud_per_kwh: float

    @staticmethod
    def crf(r: float, n: int) -> float:
        from .costing import capital_recovery

        return capital_recovery(r, n)


@dataclass
class PemParameters:
    # Electrolyser stack
    small_module_mw: float
    min_load_percent: float
    ramp_rate_percent_per_s: float
    sec_kwh_per_kg_h2: float
    operating_hours_per_year: float
    water_l_per_kg_h2: float
    water_cost_aud_per_kl: float
    stack_life_hours: float
    system_life_years: int
    degradation_percent_per_1000_hours: float
    capex_aud_per_kw: float
    opex_percent_capex_per_year: float
    stack_replacement_percent: float
    # Site PEM is built from large_module_mw plants plus small_module_mw plants.
    # capex_aud_per_kw applies at capex_reference_mw; balance of plant scales the sizes.
    large_module_mw: float
    capex_reference_mw: float
    bop_capex_aud_per_kw_small: float
    bop_capex_aud_per_kw_large: float

    # Solar share is for annual screening; hourly deployment uses site capacities.
    hybrid_solar_share_percent: float
    solar_capex_aud_per_kw: float
    wind_capex_aud_per_kw: float
    solar_opex_aud_per_mw_per_yr: float
    wind_opex_aud_per_mw_per_yr: float

    # Line from the renewables site to the port; length is per-site route data.
    transmission_losses_percent: float  # %/km of throughput lost
    transmission_capex_aud_per_km: float
    transmission_opex: float  # % CAPEX/yr
    # Wind + solar below this leaves a site unused (see run()).
    min_site_renewables_mw: float

    # Engineered underground H2 storage, AUD per kg working gas.
    engineered_storage_small_t: float
    engineered_storage_large_t: float
    lrc_capex_aud_per_kg_small: float
    lrc_capex_aud_per_kg_large: float
    lrc_opex_percent_capex: float
    underground_pipe_capex_aud_per_kg: float
    underground_pipe_opex_percent_capex: float
    engineered_storage_life_years: int
    regional_storage_life_years: int
    regional_storage_opex_percent_capex: float
    storage_injection_kwh_per_kg: float
    storage_compressor_ref_flow_kg_per_hour: float
    storage_compressor_ref_capex_aud: float

    # Battery power = energy / workbook duration.
    battery_round_trip_efficiency_percent: float
    battery_life_years: int
    battery_capex_aud_per_kwh: float

    solar_life_years: int
    wind_life_years: int
    offshore_wind_life_years: int
    offshore_wind_capex_aud_per_kw: float
    offshore_wind_opex_aud_per_mw_per_yr: float
    transmission_circuit_capacity_mw: float
    battery_storage_duration_hours: float
    battery_opex_percent_capex_per_year: float

@dataclass
class MeohSynthParameters:
    available_hours_per_year: float
    # Design capacity (nameplate, 8760 hr/yr basis  -  design sustained rate x 8760)
    capacity_tpy: float  # Nameplate capacity [t/year] @ 8760 hr/yr
    ref_capacity_tpy: float  # Reference capacity [t/year] @ 8760 hr/yr
    plant_life_years: int  # Plant economic life [years]

    overall_co2_yield_percent: float
    heat_efficiency: float
    min_load_percent: float  # Minimum stable load when turned down [% of scheduled rate]

    # Energy consumption
    sec_electricity_kwh_per_t: float  # Specific electricity [kWh/t MeOH]
    sec_heat_kwh_per_t: float  # Specific heat [kWh/t MeOH]

    # Economic parameters
    operating_hours_per_year: float  # Operating hours per year [hours]
    ref_capex_aud: float  # Reference capex at ref_capacity [AUD]
    capex_scaling_exponent: float  # Capex scaling exponent [dimensionless]
    fom_percent_capex_per_year: float  # Fixed O&M as % of capex [%/year]
    vom_aud_per_t: float  # Variable O&M [AUD/t]
    catalyst_cost_aud_per_tonne: (
        float  # Catalyst cost per tonne of catalyst [AUD/t catalyst]
    )
    catalyst_inventory_g_per_tpy: (
        float  # Catalyst inventory requirement [g per t MeOH/year capacity]
    )
    catalyst_life_years: float  # Catalyst replacement life [years]


@dataclass
class ShipParameters:
    # Ship (one reference design, no scaling  -  CAPEX/OPEX are absolute per vessel)
    carrier_cargo_volume_m3: float  # Cargo tank volume [m3]; payload on both legs
    co2_cargo_density_kg_per_m3: float  # LCO2 on the return leg [kg/m3]
    ref_ship_capex_aud: float  # Newbuild CAPEX per vessel [AUD]
    ship_annual_opex_aud: float  # Fixed annual OPEX per vessel [AUD/yr]
    sea_fuel_consumption_tpd: float  # Main-engine fuel burn at sea [t/d]
    port_fuel_consumption_tpd: float  # Auxiliary fuel burn in port [t/d]
    ship_lifetime_years: int  # Vessel economic life [years]

    # Voyage (Australia <-> Japan leg)
    shipping_distance_nm: float  # One-way distance [nm]
    voyage_speed_kn: float  # Service speed [knots]
    port_turnaround_days: float  # Combined load + discharge time [d]
    boiloff_rate_pct_per_day: float  # BOG on loaded leg [%/d]

    reference_fuel_lhv_mj_per_kg: float
    available_days_per_year: float
    terminal_storage_days: float
    terminal_capex_aud_per_t: float
    terminal_opex_percent_capex: float

    # "methanol": bunkers drawn from the e-MeOH stream (base case).
    # "fuel_oil": fossil HFO-equivalent bought at fuel_oil_price_usd_per_t.
    fuel_type: str = "methanol"
    fuel_oil_price_usd_per_t: float = 0.0
    # 0: each port has its own fleet, rounded up per port. 1: ports share one
    # fleet; each port pays for its fractional ship-years and the fleet is
    # rounded up once.
    pooled_fleet: float = 0


@dataclass
class CrackerParameters:
    available_hours_per_year: float
    ref_capex_aud_per_tpd_h2: float
    ref_capacity_tpd_h2: float
    capex_scaling_factor: float
    annual_opex_percent_capex: float
    operating_hours_per_year: float
    co2_capture_rate: float
    thermal_gj_per_t_meoh: float
    cracker_conversion_rate: float
    parasitic_kwh_per_kg_h2: float
    cracker_power_price_aud_per_kwh: float
    cracker_grid_ei_kg_co2_per_kwh: float
    # Per-unit nameplate H2 production rate of one cracker module [t H2/d].
    # Fleet count n_cracker = ceil(target / annual H2 per unit at nominal hours).
    # Defaults to ref_capacity_tpd_h2 in load_cracker(); add a separate Excel row to override.
    capacity_tpd_h2: float
    min_capacity_tpd_h2: float
    max_capacity_tpd_h2: float
    methanol_recycle_enabled: float
    psa_h2_recovery: float
    tail_gas_heat_efficiency: float
    burner_efficiency: float
    # Shares of net reformer heat supplied electrically or by burning product H2;
    # the remainder burns internal methanol. They sum to at most 1.
    electric_heat_share: float
    electric_heater_efficiency: float
    h2_heat_share: float
    steam_water_cost_aud_per_kl: float
    zero_carbon_power_price_aud_per_kwh: float
    heat_pump_enabled: float
    heat_pump_cop: float
    heat_pump_eligible_heat_share: float
    heat_pump_source_gj_per_t_meoh: float
    heat_pump_capex_aud_per_kw_th: float
    heat_pump_installation_factor: float
    heat_pump_opex_percent_capex: float
    heat_pump_life_years: float
    cracker_power_source: float
    co2_capture_config: float
    capture_cost_concentrated_aud_per_t_co2: float
    capture_cost_dilute_aud_per_t_co2: float
    # Concentrated (config 0) is PSA on pressurised reformate: electric, no regeneration heat.
    # Dilute (config 1) is amine capture from flue gas.
    capture_electricity_concentrated_kwh_per_t_co2: float
    capture_electricity_dilute_kwh_per_t_co2: float
    capture_regen_heat_concentrated_gj_per_t_co2: float
    capture_regen_heat_dilute_gj_per_t_co2: float

    @property
    def capture_electricity_kwh_per_t_co2(self):
        return (self.capture_electricity_dilute_kwh_per_t_co2 if self.co2_capture_config
                else self.capture_electricity_concentrated_kwh_per_t_co2)

    @property
    def capture_regen_heat_gj_per_t_co2(self):
        return (self.capture_regen_heat_dilute_gj_per_t_co2 if self.co2_capture_config
                else self.capture_regen_heat_concentrated_gj_per_t_co2)

    @property
    def capture_cost_aud_per_t_co2(self):
        return (self.capture_cost_dilute_aud_per_t_co2 if self.co2_capture_config
                else self.capture_cost_concentrated_aud_per_t_co2)

    @property
    def power_price_aud_per_kwh(self):
        if self.cracker_power_source not in (0, 1):
            raise ValueError("Cracker power source must be 0 or 1")
        return (self.zero_carbon_power_price_aud_per_kwh if self.cracker_power_source
                else self.cracker_power_price_aud_per_kwh)

    @property
    def power_ei_kg_co2_per_kwh(self):
        if self.cracker_power_source not in (0, 1):
            raise ValueError("Cracker power source must be 0 or 1")
        return 0.0 if self.cracker_power_source else self.cracker_grid_ei_kg_co2_per_kwh


@dataclass
class Co2CaptureParameters:
    cargo_loss_fraction: float
    liquefaction_storage_aud_per_t: float
    liquefaction_electricity_kwh_per_t_co2: float


@dataclass
class DacParameters:
    # Delivered cost of the make-up CO2 the return voyage does not cover.
    dac_cost_aud_per_t: float


@dataclass
class TargetParameters:
    target_h2_tpa: float


def load_economics(path: Path = XLSX_PATH) -> Economics:
    v = load_params(path, "economics")
    # Price indices are consumed by the loader's cost-basis conversion.
    indices = tuple(
        k for k in v if k == "cost_base_year" or k.startswith(("aud_cpi_", "usd_cpi_"))
    )
    return from_dict(Economics, v, ignored=indices)


def load_pem(path: Path = XLSX_PATH) -> PemParameters:
    # PEM + dedicated renewables + battery are operated as one integrated
    # H2 production block, so their rows are pooled across their four
    # module sheets before constructing the dataclass.
    return from_dict(
        PemParameters,
        load_params(
            path,
            ("green_h2_pem", "h2_storage", "renewable_supply", "renewable_storage"),
        ),
        ignored=(
            "h2_storage_default_capacity_kg",
            "transmission_distance_km",
            "region_index",
            "oversizing_factor",
            "solar_oversizing_factor",
        ),
    )


def load_synth(path: Path = XLSX_PATH) -> MeohSynthParameters:
    return from_dict(MeohSynthParameters, load_params(path, "meoh_synthesis"))


def load_ship(path: Path = XLSX_PATH) -> ShipParameters:
    return from_dict(ShipParameters, load_params(path, "shipping"))


def load_cracker(path: Path = XLSX_PATH) -> CrackerParameters:
    d = dict(load_params(path, "meoh_cracking"))
    # The revised sheet carries physical availability as the operating-hours ceiling.
    d.setdefault("available_hours_per_year", float(parameter_input(
        path, "meoh_cracking", "operating_hours_per_year")["Upper bound"]))
    capacity = parameter_input(path, "meoh_cracking", "capacity_tpd_h2")
    d["min_capacity_tpd_h2"] = float(capacity["Lower bound"])
    d["max_capacity_tpd_h2"] = float(capacity["Upper bound"])
    return from_dict(CrackerParameters, d)


def load_co2_capture(path: Path = XLSX_PATH) -> Co2CaptureParameters:
    return from_dict(Co2CaptureParameters, load_params(path, "co2_capture"))


def load_dac(path: Path = XLSX_PATH) -> DacParameters:
    return from_dict(DacParameters, load_params(path, "co2_balancing"))


def default_target() -> TargetParameters:
    """Total H2 demand: what the saved grey base case delivers."""
    return TargetParameters(target_h2_tpa=load_grey().target_h2_tpa)


@dataclass
class H2TransportParameters:
    pipeline_capex_aud_per_km: float
    offshore_pipeline_capex_aud_per_km: float
    pipeline_capacity_kg_per_hour: float
    pipeline_life_years: int
    pipeline_opex_percent_capex: float
    compression_kwh_per_kg_per_100km: float
    transport_legs: float
    compressor_capex_aud_per_kw: float


def load_h2_transport(path: Path = XLSX_PATH) -> H2TransportParameters:
    return from_dict(H2TransportParameters, load_params(path, "h2_transport"))


def load_routes(path: Path = XLSX_PATH):
    from inputs.loader import load_input_rows, _split_field

    result = {p: {} for p in PORT_SITES}
    for sheet in ("shipping", "h2_transport"):
        for _, row in load_input_rows(path, sheet):
            port, field = _split_field(row.get("Field"))
            if port is None:
                continue
            if port not in result or field in result[port]:
                raise ValueError(f"Unknown or duplicate route: {port}.{field}")
            result[port][field] = float(row["Value"])
            if not math.isfinite(result[port][field]):
                raise ValueError(f"Non-finite route input: {port}.{field}")
            result[port][field + "_note"] = row.get("Use / assumption", "")
    for port, values in result.items():
        if (
            not {
                "route_distance_nm",
                "renewables_to_port_km",
                "pipeline_distance_km",
                "offshore_fraction",
            }
            <= values.keys()
        ):
            raise ValueError(f"Missing route inputs for {port}")
        if (
            values["route_distance_nm"] <= 0
            or values["renewables_to_port_km"] < 0
            or values["pipeline_distance_km"] < 0
            or not 0 <= values["offshore_fraction"] <= 1
        ):
            raise ValueError(f"Invalid route inputs for {port}")
    return result
