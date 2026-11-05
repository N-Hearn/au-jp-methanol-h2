"""Base case parameters.

All numeric values are loaded from inputs/input_data.xlsx at runtime.
Inputs and references are kept together on the corresponding module sheet.
Derived quantities (annual LNG energy, delivered H2 target) live in code.
"""

from dataclasses import dataclass, fields
from pathlib import Path

from inputs.loader import load_params, from_dict, XLSX_PATH
from constants import MTPA_TO_TONNE


def derive_target_h2_tpa(
    lng_capacity_mtpa: float, h2_yield_kg_ng_per_kg_h2: float
) -> float:
    """Annual H2 tonnage the LNG stream would yield if reformed [t H2/yr]."""
    return lng_capacity_mtpa * MTPA_TO_TONNE / h2_yield_kg_ng_per_kg_h2


@dataclass
class LngSupplyParameters:
    lng_capacity_mtpa: float
    lng_lhv_mj_per_kg: float
    lng_feed_gas_cost_aud_per_gj: float
    lng_liquefaction_cost_aud_per_gj: float
    lng_shipping_cost_aud_per_gj: float
    supply_chain_ghg_kg_co2_per_gj: float

    @property
    def lng_price_aud_per_gj(self):
        return (self.lng_feed_gas_cost_aud_per_gj + self.lng_liquefaction_cost_aud_per_gj
                + self.lng_shipping_cost_aud_per_gj)


@dataclass
class AtrPlantParameters:
    plant_capacity_tpd_h2: float
    operating_hours_per_year: float
    h2_yield_kg_ng_per_kg_h2: float
    grey_prod_cost_ex_feed_aud_per_kg_h2: float
    grey_plant_gate_ei_kg_co2_per_kg_h2: float
    electricity_kwh_per_kg_h2: float  # ASU, on the Japanese power basis


@dataclass
class AtrCcsParameters:
    """The blue-only rows (atr_ccs sheet): storage in Australia and the LCO2 fleet route and fuel."""

    co2_storage_cost_aud_per_t_co2: float
    lco2_route_distance_nm: float
    lco2_fuel_oil_price_aud_per_t: float


# The LNG cases share these rows with the methanol loop so both pathways sit on
# one basis. Each mirrors a subset of the meoh dataclass of the same group
# (cracker, co2_cap, ship, econ), so Monte Carlo routes the rows by sheet.

@dataclass
class JapanPlantParameters:
    """Japanese power basis and pre-combustion (PSA) capture (meoh_cracking sheet)."""

    cracker_power_source: float
    cracker_power_price_aud_per_kwh: float
    zero_carbon_power_price_aud_per_kwh: float
    cracker_grid_ei_kg_co2_per_kwh: float
    co2_capture_rate: float
    capture_cost_concentrated_aud_per_t_co2: float
    capture_electricity_concentrated_kwh_per_t_co2: float

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
class Co2LiquefactionParameters:
    """CO2 liquefaction and terminal storage in Japan (co2_capture sheet)."""

    liquefaction_storage_aud_per_t: float
    liquefaction_electricity_kwh_per_t_co2: float


@dataclass
class CarrierParameters:
    """The methanol loop's carrier design, here as a dedicated LCO2 fleet (shipping sheet)."""

    carrier_cargo_volume_m3: float
    co2_cargo_density_kg_per_m3: float
    ref_ship_capex_aud: float
    ship_annual_opex_aud: float
    sea_fuel_consumption_tpd: float
    port_fuel_consumption_tpd: float
    ship_lifetime_years: float
    voyage_speed_kn: float
    port_turnaround_days: float
    available_days_per_year: float


@dataclass
class DiscountParameters:
    discount_rate: float


@dataclass
class DeliveredH2Target:
    target_h2_tpa: float
    aud_usd_exchange_rate: float


def load_lng(path: Path = XLSX_PATH) -> LngSupplyParameters:
    return from_dict(LngSupplyParameters, load_params(path, "lng_supply"))


def load_atr(path: Path = XLSX_PATH) -> AtrPlantParameters:
    return from_dict(AtrPlantParameters, load_params(path, "atr_plant"))


def load_atr_ccs(path: Path = XLSX_PATH) -> AtrCcsParameters:
    return from_dict(AtrCcsParameters, load_params(path, "atr_ccs"))


def _shared(cls, path, sheet):
    values = load_params(path, sheet)
    return cls(**{f.name: values[f.name] for f in fields(cls)})


def load_japan_plant(path: Path = XLSX_PATH) -> JapanPlantParameters:
    return _shared(JapanPlantParameters, path, "meoh_cracking")


def load_co2_liquefaction(path: Path = XLSX_PATH) -> Co2LiquefactionParameters:
    return _shared(Co2LiquefactionParameters, path, "co2_capture")


def load_carrier(path: Path = XLSX_PATH) -> CarrierParameters:
    return _shared(CarrierParameters, path, "shipping")


def load_discount(path: Path = XLSX_PATH) -> DiscountParameters:
    return _shared(DiscountParameters, path, "economics")


def default_target(path: Path = XLSX_PATH) -> DeliveredH2Target:
    lng = load_lng(path)
    atr = load_atr(path)
    aud_usd = load_params(path, "economics")["aud_usd_exchange_rate"]
    return DeliveredH2Target(
        target_h2_tpa=derive_target_h2_tpa(
            lng.lng_capacity_mtpa, atr.h2_yield_kg_ng_per_kg_h2
        ),
        aud_usd_exchange_rate=aud_usd,
    )
