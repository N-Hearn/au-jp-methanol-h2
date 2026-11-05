"""Liquefaction and storage of cracker-captured CO2 in Japan.

The liquid CO2 rides back to Australia on the methanol carriers
(shipping), so this stage carries no ships, freight or voyage
emissions. Sized on the CO2 actually captured at the cracker; any synth
CO2 demand above what is returned is sourced as external make-up CO2 in
the supply chain.
"""

import math
from dataclasses import dataclass

from .parameters import Co2CaptureParameters
from constants import KG_PER_TONNE


@dataclass
class Co2CaptureResult:
    annual_co2_captured_tpa: float
    annual_co2_returned_tpa: float

    liquefaction_storage_aud_per_t: float
    annual_cost_aud: float
    cost_aud_per_kg_h2: float

    liquefaction_electricity_mwh: float
    liquefaction_electricity_co2_tpa: float


class Co2Capture:
    def __init__(
        self,
        params: Co2CaptureParameters,
        annual_co2_captured_tpa: float,
        annual_h2_delivered_tpa: float,
        power_ei_kg_co2_per_kwh=0.0,
    ):
        self.p = params
        self.annual_co2_captured_tpa = annual_co2_captured_tpa
        self.annual_h2_delivered_tpa = annual_h2_delivered_tpa
        self.power_ei = power_ei_kg_co2_per_kwh

    def run(self) -> Co2CaptureResult:
        p = self.p
        if any(not math.isfinite(v) for v in vars(p).values()):
            raise ValueError("CO2 capture inputs must be finite")
        if (not math.isfinite(self.power_ei) or self.power_ei < 0
                or p.liquefaction_electricity_kwh_per_t_co2 < 0):
            raise ValueError("Invalid liquefaction electricity or emission factor")
        if (not 0 <= p.cargo_loss_fraction < 1
                or not math.isfinite(self.annual_co2_captured_tpa)
                or not math.isfinite(self.annual_h2_delivered_tpa)
                or self.annual_co2_captured_tpa < 0 or self.annual_h2_delivered_tpa <= 0):
            raise ValueError("Invalid CO2 cargo loss or throughput")

        captured = self.annual_co2_captured_tpa
        annual_cost_aud = captured * p.liquefaction_storage_aud_per_t
        electricity_mwh = captured * p.liquefaction_electricity_kwh_per_t_co2 / 1000
        return Co2CaptureResult(
            annual_co2_captured_tpa=captured,
            annual_co2_returned_tpa=captured * (1 - p.cargo_loss_fraction),
            liquefaction_storage_aud_per_t=p.liquefaction_storage_aud_per_t,
            annual_cost_aud=annual_cost_aud,
            cost_aud_per_kg_h2=annual_cost_aud / (self.annual_h2_delivered_tpa * KG_PER_TONNE),
            liquefaction_electricity_mwh=electricity_mwh,
            liquefaction_electricity_co2_tpa=electricity_mwh * self.power_ei,
        )
