"""LNG supply from Australia to Japan."""

from dataclasses import dataclass

from .parameters import LngSupplyParameters
from constants import MTPA_TO_TONNE


@dataclass
class LngSupplyResult:
    lng_capacity_tpa: float
    annual_lng_energy_gj: float
    lng_price_aud_per_gj: float
    supply_chain_ghg_kg_co2_per_gj: float


class LngSupply:
    def __init__(self, params: LngSupplyParameters):
        self.p = params

    def run(self) -> LngSupplyResult:
        p = self.p
        lng_tpa = p.lng_capacity_mtpa * MTPA_TO_TONNE
        annual_energy_gj = p.lng_capacity_mtpa * p.lng_lhv_mj_per_kg * MTPA_TO_TONNE

        return LngSupplyResult(
            lng_capacity_tpa=lng_tpa,
            annual_lng_energy_gj=annual_energy_gj,
            lng_price_aud_per_gj=p.lng_price_aud_per_gj,
            supply_chain_ghg_kg_co2_per_gj=p.supply_chain_ghg_kg_co2_per_gj,
        )
