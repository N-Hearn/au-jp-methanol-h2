"""Delivered H2 cost and emissions intensity for the Japan base case."""

from dataclasses import dataclass

from .parameters import (
    LngSupplyParameters,
    AtrPlantParameters,
    DeliveredH2Target,
    JapanPlantParameters,
)
from .lng_supply import LngSupply
from .atr_plant import AtrPlant
from constants import H2_LHV_MJ_PER_KG, KG_PER_TONNE, MJ_PER_GJ


@dataclass
class DeliveredH2Result:
    feed_gj_per_kg_h2: float
    feed_cost_aud_per_kg_h2: float
    prod_cost_ex_feed_aud_per_kg_h2: float
    atr_electricity_cost_aud_per_kg_h2: float

    delivered_cost_aud_per_kg_h2: float
    delivered_cost_usd_per_kg_h2: float
    delivered_cost_aud_per_t_h2: float
    delivered_cost_usd_per_gj_h2_lhv: float

    plant_gate_ei_kg_co2_per_kg_h2: float
    atr_electricity_ei_kg_co2_per_kg_h2: float
    upstream_ei_kg_co2_per_kg_h2: float
    delivered_ei_kg_co2_per_kg_h2: float

    target_h2_tpa: float
    plant_capacity_tpd_h2: float
    annual_h2_per_plant_tpa: float
    fleet_plants: int
    fleet_annual_h2_tpa: float


class DeliveredH2:
    def __init__(
        self,
        lng_params: LngSupplyParameters,
        atr_params: AtrPlantParameters,
        target: DeliveredH2Target,
        japan: JapanPlantParameters,
    ):
        self.lng_params = lng_params
        self.atr_params = atr_params
        self.target = target
        self.japan = japan

    def run(self) -> DeliveredH2Result:
        lng = LngSupply(self.lng_params).run()
        atr = AtrPlant(
            self.atr_params, ng_lhv_mj_per_kg=self.lng_params.lng_lhv_mj_per_kg
        ).run(target_h2_tpa=self.target.target_h2_tpa)

        # ASU power on the same Japanese power basis as the methanol cracker.
        kwh = self.atr_params.electricity_kwh_per_kg_h2
        if kwh < 0:
            raise ValueError("ATR electricity must be non-negative")
        electricity_cost = kwh * self.japan.power_price_aud_per_kwh
        electricity_ei = kwh * self.japan.power_ei_kg_co2_per_kwh

        feed_cost_aud = atr.feed_gas_gj_per_kg_h2 * lng.lng_price_aud_per_gj
        delivered_aud_per_kg = feed_cost_aud + atr.prod_cost_ex_feed_aud_per_kg_h2 + electricity_cost
        delivered_aud_per_t = delivered_aud_per_kg * KG_PER_TONNE
        delivered_usd_per_kg = delivered_aud_per_kg * self.target.aud_usd_exchange_rate
        delivered_usd_per_gj = delivered_usd_per_kg / (H2_LHV_MJ_PER_KG / MJ_PER_GJ)

        upstream_ei = lng.supply_chain_ghg_kg_co2_per_gj * atr.feed_gas_gj_per_kg_h2
        delivered_ei = atr.plant_gate_ei_kg_co2_per_kg_h2 + electricity_ei + upstream_ei

        return DeliveredH2Result(
            feed_gj_per_kg_h2=atr.feed_gas_gj_per_kg_h2,
            feed_cost_aud_per_kg_h2=feed_cost_aud,
            prod_cost_ex_feed_aud_per_kg_h2=atr.prod_cost_ex_feed_aud_per_kg_h2,
            atr_electricity_cost_aud_per_kg_h2=electricity_cost,
            delivered_cost_aud_per_kg_h2=delivered_aud_per_kg,
            delivered_cost_usd_per_kg_h2=delivered_usd_per_kg,
            delivered_cost_aud_per_t_h2=delivered_aud_per_t,
            delivered_cost_usd_per_gj_h2_lhv=delivered_usd_per_gj,
            plant_gate_ei_kg_co2_per_kg_h2=atr.plant_gate_ei_kg_co2_per_kg_h2,
            atr_electricity_ei_kg_co2_per_kg_h2=electricity_ei,
            upstream_ei_kg_co2_per_kg_h2=upstream_ei,
            delivered_ei_kg_co2_per_kg_h2=delivered_ei,
            target_h2_tpa=atr.target_h2_tpa,
            plant_capacity_tpd_h2=atr.plant_capacity_tpd_h2,
            annual_h2_per_plant_tpa=atr.annual_h2_per_plant_tpa,
            fleet_plants=atr.fleet_plants,
            fleet_annual_h2_tpa=atr.fleet_annual_h2_tpa,
        )
