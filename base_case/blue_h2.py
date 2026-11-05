"""Blue H2 base case: the grey LNG/ATR reference chain plus syngas capture and
CO2 return to Australia.

Follows the LNG energy-vector chain of Al Ghafri et al. (2023): ATR in Japan with
pre-combustion capture, CO2 liquefaction, LCO2 shipping back to Australia and
geological storage. LNG feed, ATR yield and the common H2 target are the grey
case's, so the two LNG cases differ only by the capture chain.

Every Japan-end and shipping assumption is the methanol loop's: PSA capture on
the pressurised syngas at the cracker's rate, cost and electricity; liquefaction
and terminal storage at the loop's allowance; one Japanese power price and grid
factor; and the loop's carrier design, here as a dedicated LCO2 fleet that
returns in ballast on fuel oil. Only storage in Australia is blue's own.
"""

import math
from dataclasses import dataclass

from .delivered_h2 import DeliveredH2, DeliveredH2Result
from .parameters import (
    AtrCcsParameters, AtrPlantParameters, CarrierParameters, Co2LiquefactionParameters,
    DeliveredH2Target, DiscountParameters, JapanPlantParameters, LngSupplyParameters,
)
from constants import (
    FUEL_OIL_CO2_EF_KG_PER_KG, H2_LHV_MJ_PER_KG, HOURS_PER_DAY, KG_PER_TONNE, MJ_PER_GJ,
)


@dataclass
class BlueH2Result:
    grey: DeliveredH2Result

    captured_co2_kg_per_kg_h2: float
    vented_co2_kg_per_kg_h2: float
    fleet_co2_stored_tpa: float
    capture_electricity_kwh_per_kg_h2: float
    liquefaction_electricity_kwh_per_kg_h2: float
    lco2_ships: int

    capture_cost_aud_per_kg_h2: float
    co2_liquefaction_cost_aud_per_kg_h2: float
    co2_shipping_cost_aud_per_kg_h2: float
    co2_storage_cost_aud_per_kg_h2: float

    capture_power_ei_kg_co2_per_kg_h2: float
    co2_liquefaction_ei_kg_co2_per_kg_h2: float
    co2_shipping_ei_kg_co2_per_kg_h2: float

    delivered_cost_aud_per_kg_h2: float
    delivered_cost_usd_per_kg_h2: float
    delivered_cost_usd_per_gj_h2_lhv: float
    delivered_ei_kg_co2_per_kg_h2: float

    @property
    def target_h2_tpa(self):
        return self.grey.target_h2_tpa


class BlueH2:
    def __init__(
        self,
        lng_params: LngSupplyParameters,
        atr_params: AtrPlantParameters,
        ccs_params: AtrCcsParameters,
        target: DeliveredH2Target,
        japan: JapanPlantParameters,
        liquefaction: Co2LiquefactionParameters,
        carrier: CarrierParameters,
        discount: DiscountParameters,
    ):
        self.grey = DeliveredH2(lng_params, atr_params, target, japan)
        self.ccs = ccs_params
        self.target = target
        self.japan = japan
        self.liquefaction = liquefaction
        self.carrier = carrier
        self.discount = discount

    def _lco2_fleet(self, co2_tpa):
        """(ships, AUD per t CO2, kg CO2 per kg CO2 shipped) for a dedicated LCO2 fleet.

        The methanol loop's carrier on the Japan-storage route: laden out, back in
        ballast at the same fuel rate (as the loop charges both legs at the laden rate).
        """
        from meoh_pathway.costing import capital_recovery

        s, c = self.carrier, self.ccs
        if min(s.carrier_cargo_volume_m3, s.co2_cargo_density_kg_per_m3, s.voyage_speed_kn,
               s.available_days_per_year, c.lco2_route_distance_nm) <= 0 or co2_tpa <= 0:
            raise ValueError("Invalid LCO2 fleet inputs")
        payload_t = s.carrier_cargo_volume_m3 * s.co2_cargo_density_kg_per_m3 / KG_PER_TONNE
        sea_days = 2.0 * c.lco2_route_distance_nm / (s.voyage_speed_kn * HOURS_PER_DAY)
        voyages = s.available_days_per_year / (sea_days + s.port_turnaround_days)
        ships_frac = co2_tpa / (voyages * payload_t)
        ships = math.ceil(ships_frac)
        fuel_t = ships_frac * voyages * (sea_days * s.sea_fuel_consumption_tpd
                                          + s.port_turnaround_days * s.port_fuel_consumption_tpd)
        annual_ship_aud = (s.ref_ship_capex_aud * capital_recovery(self.discount.discount_rate,
                                                                   s.ship_lifetime_years)
                           + s.ship_annual_opex_aud)
        cost_aud_per_t = (ships * annual_ship_aud + fuel_t * c.lco2_fuel_oil_price_aud_per_t) / co2_tpa
        return ships, cost_aud_per_t, fuel_t * FUEL_OIL_CO2_EF_KG_PER_KG / co2_tpa

    def run(self) -> BlueH2Result:
        j, liq = self.japan, self.liquefaction
        if not 0 <= j.co2_capture_rate <= 1:
            raise ValueError("Invalid CO2 capture rate")
        if min(j.capture_electricity_concentrated_kwh_per_t_co2,
               liq.liquefaction_electricity_kwh_per_t_co2) < 0:
            raise ValueError("Capture and liquefaction electricity must be non-negative")
        grey = self.grey.run()

        # ATR forms all its CO2 in the reactor, so capture acts on the plant-gate CO2.
        captured = j.co2_capture_rate * grey.plant_gate_ei_kg_co2_per_kg_h2
        vented = grey.plant_gate_ei_kg_co2_per_kg_h2 - captured
        captured_t = captured / KG_PER_TONNE
        capture_kwh = captured_t * j.capture_electricity_concentrated_kwh_per_t_co2
        liquefaction_kwh = captured_t * liq.liquefaction_electricity_kwh_per_t_co2
        ships, shipping_aud_per_t, shipping_ei_per_kg = self._lco2_fleet(captured * grey.target_h2_tpa)

        # As in the loop, liquefaction electricity is priced inside its allowance.
        capture_cost = (captured_t * j.capture_cost_concentrated_aud_per_t_co2
                        + capture_kwh * j.power_price_aud_per_kwh)
        liquefaction_cost = captured_t * liq.liquefaction_storage_aud_per_t
        shipping_cost = captured_t * shipping_aud_per_t
        storage_cost = captured_t * self.ccs.co2_storage_cost_aud_per_t_co2
        delivered_aud = (grey.delivered_cost_aud_per_kg_h2 + capture_cost + liquefaction_cost
                         + shipping_cost + storage_cost)
        delivered_usd = delivered_aud * self.target.aud_usd_exchange_rate

        capture_power_ei = capture_kwh * j.power_ei_kg_co2_per_kwh
        liquefaction_ei = liquefaction_kwh * j.power_ei_kg_co2_per_kwh
        shipping_ei = captured * shipping_ei_per_kg
        delivered_ei = (vented + grey.atr_electricity_ei_kg_co2_per_kg_h2 + capture_power_ei
                        + liquefaction_ei + shipping_ei + grey.upstream_ei_kg_co2_per_kg_h2)

        return BlueH2Result(
            grey=grey,
            captured_co2_kg_per_kg_h2=captured,
            vented_co2_kg_per_kg_h2=vented,
            fleet_co2_stored_tpa=captured * grey.target_h2_tpa,
            capture_electricity_kwh_per_kg_h2=capture_kwh,
            liquefaction_electricity_kwh_per_kg_h2=liquefaction_kwh,
            lco2_ships=ships,
            capture_cost_aud_per_kg_h2=capture_cost,
            co2_liquefaction_cost_aud_per_kg_h2=liquefaction_cost,
            co2_shipping_cost_aud_per_kg_h2=shipping_cost,
            co2_storage_cost_aud_per_kg_h2=storage_cost,
            capture_power_ei_kg_co2_per_kg_h2=capture_power_ei,
            co2_liquefaction_ei_kg_co2_per_kg_h2=liquefaction_ei,
            co2_shipping_ei_kg_co2_per_kg_h2=shipping_ei,
            delivered_cost_aud_per_kg_h2=delivered_aud,
            delivered_cost_usd_per_kg_h2=delivered_usd,
            delivered_cost_usd_per_gj_h2_lhv=delivered_usd / (H2_LHV_MJ_PER_KG / MJ_PER_GJ),
            delivered_ei_kg_co2_per_kg_h2=delivered_ei,
        )
