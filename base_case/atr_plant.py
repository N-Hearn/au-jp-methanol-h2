"""ATR hydrogen plant at the Japanese import terminal (grey H2)."""

import math
from dataclasses import dataclass

from .parameters import AtrPlantParameters
from constants import HOURS_PER_DAY, MJ_PER_GJ


@dataclass
class AtrPlantResult:
    plant_capacity_tpd_h2: float
    annual_h2_per_plant_tpa: float

    feed_gas_gj_per_kg_h2: float
    prod_cost_ex_feed_aud_per_kg_h2: float
    plant_gate_ei_kg_co2_per_kg_h2: float

    target_h2_tpa: float
    fleet_plants: int
    fleet_annual_h2_tpa: float


class AtrPlant:
    def __init__(self, params: AtrPlantParameters, ng_lhv_mj_per_kg: float):
        self.p = params
        self.ng_lhv_mj_per_kg = ng_lhv_mj_per_kg

    def run(self, target_h2_tpa: float) -> AtrPlantResult:
        p = self.p

        annual_h2_per_plant = p.plant_capacity_tpd_h2 * (
            p.operating_hours_per_year / HOURS_PER_DAY
        )
        feed_gj_per_kg_h2 = (
            p.h2_yield_kg_ng_per_kg_h2 * self.ng_lhv_mj_per_kg / MJ_PER_GJ
        )

        prod_cost_ex_feed = p.grey_prod_cost_ex_feed_aud_per_kg_h2
        ei_plant_gate = p.grey_plant_gate_ei_kg_co2_per_kg_h2

        fleet_plants = int(math.ceil(target_h2_tpa / annual_h2_per_plant))
        annual_h2_per_plant = target_h2_tpa / fleet_plants
        fleet_h2 = target_h2_tpa

        return AtrPlantResult(
            plant_capacity_tpd_h2=p.plant_capacity_tpd_h2,
            annual_h2_per_plant_tpa=annual_h2_per_plant,
            feed_gas_gj_per_kg_h2=feed_gj_per_kg_h2,
            prod_cost_ex_feed_aud_per_kg_h2=prod_cost_ex_feed,
            plant_gate_ei_kg_co2_per_kg_h2=ei_plant_gate,
            target_h2_tpa=target_h2_tpa,
            fleet_plants=fleet_plants,
            fleet_annual_h2_tpa=fleet_h2,
        )
