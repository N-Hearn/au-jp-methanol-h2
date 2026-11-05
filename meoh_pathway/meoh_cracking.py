"""Methanol decomposition (cracker) plant in Japan (single module).

CH3OH + H2O -> CO2 + 3 H2.

`meoh_feed_tpd` is the MeOH actually delivered to the cracker (i.e. after
shipping BOG losses), not the synth-plant nameplate output.

Fresh methanol is fully converted when unconverted methanol is recycled.
Without recycle, unconverted feed burns in the tail gas. Capture targets either
reaction CO2 before combustion or all flue CO2, including supplemental fuel.
"""

from dataclasses import dataclass
import math

from .parameters import CrackerParameters, Economics
from constants import (
    CO2_STOICH_KG_PER_KG_MEOH,
    HOURS_PER_DAY,
    KG_PER_TONNE,
    MEOH_LHV_MJ_PER_KG,
    H2_STOICH_KG_PER_KG_MEOH,
    H2_LHV_MJ_PER_KG,
    M_H2O,
    M_MEOH,
)


@dataclass
class CrackerResult:
    plant_capacity_tpd_h2: float
    gross_capacity_tpd_h2: float
    annual_h2_tpa: float

    capex_aud: float
    crf: float
    annual_opex_aud: float

    annualised_capex_aud: float
    levelised_cost_aud_per_kg_h2: float

    captured_co2_tpd: float
    vented_co2_tpd: float
    effective_capture_rate: float
    recycle_meoh_tpd: float
    capture_regen_heat_gj_per_year: float
    capture_electricity_mwh: float
    annual_capture_cost_aud: float
    regeneration_electricity_mwh: float
    reformer_electricity_mwh: float
    heat_pump_regeneration_heat_mwh: float
    heat_pump_source_limited: bool
    tail_gas_heat_capped: bool
    co2_capture_config: int
    cracker_power_source: int
    power_price_aud_per_kwh: float
    power_ei_kg_co2_per_kwh: float
    tail_gas_heat_dumped_gj_per_year: float

    gross_thermal_gj_per_year: float
    tail_gas_heat_credit_gj_per_year: float
    thermal_gj_per_year: float  # net external heat duty
    heat_electricity_mwh: float  # electricity supplying the electric share
    resistance_heat_mwh: float
    heat_pump_heat_mwh: float
    heat_pump_source_mwh: float
    heat_pump_electricity_mwh: float
    heat_pump_capacity_mw_th: float
    heat_pump_capex_aud: float
    heat_pump_annualised_capex_aud: float
    h2_fuel_tpa: float  # product H2 burned for the H2-fired share

    annual_electricity_mwh: float
    annual_electricity_cost_aud: float
    electricity_co2_tpa: float
    electricity_cost_aud_per_kg_h2: float
    electricity_ei_kg_co2_per_kg_h2: float

    annual_supplemental_heat_cost_aud: float
    supplemental_heat_co2_tpa: float
    supplemental_heat_cost_aud_per_kg_h2: float
    supplemental_heat_ei_kg_co2_per_kg_h2: float


def converted_fraction(p):
    if not 0 < p.cracker_conversion_rate <= 1 or not 0 < p.psa_h2_recovery <= 1:
        raise ValueError("Cracker conversion and PSA recovery must be in (0, 1]")
    if p.methanol_recycle_enabled not in (0, 1):
        raise ValueError("Methanol recycle switch must be 0 or 1")
    return 1.0 if p.methanol_recycle_enabled else p.cracker_conversion_rate


def tail_gas_heat_gj_per_t_feed(p):
    converted = converted_fraction(p)
    if not 0 < p.tail_gas_heat_efficiency <= 1 or not 0 < p.burner_efficiency <= 1:
        raise ValueError("Cracker heat efficiencies must be in (0, 1]")
    tail = (
        1 - converted
    ) * MEOH_LHV_MJ_PER_KG + H2_STOICH_KG_PER_KG_MEOH * converted * (
        1 - p.psa_h2_recovery
    ) * H2_LHV_MJ_PER_KG
    return tail * p.tail_gas_heat_efficiency


def external_heat_gj_per_t_feed(p):
    """Reformer duty after tail-gas recovery [GJ/t fresh feed], excluding capture."""
    _check_heat_shares(p)
    if p.co2_capture_config not in (0, 1) or not 0 <= p.co2_capture_rate <= 1:
        raise ValueError("Invalid CO2 capture configuration or rate")
    if not all(math.isfinite(v) and v >= 0 for v in
               (p.thermal_gj_per_t_meoh, p.capture_regen_heat_gj_per_t_co2,
                p.capture_electricity_kwh_per_t_co2)):
        raise ValueError("Invalid reformer or capture heat demand")
    return max(0.0, p.thermal_gj_per_t_meoh-tail_gas_heat_gj_per_t_feed(p))


def captured_co2_per_t_feed(p):
    carbon = (1 + supplemental_meoh_per_t_feed(p) if p.co2_capture_config
              else converted_fraction(p))
    return carbon * CO2_STOICH_KG_PER_KG_MEOH * p.co2_capture_rate


def _check_heat_shares(p):
    e, h = p.electric_heat_share, p.h2_heat_share
    if not (0 <= e <= 1 and 0 <= h <= 1 and e + h <= 1 + 1e-12):
        raise ValueError("Electric and H2 heat shares must be in [0, 1] and sum to at most 1")


def supplemental_meoh_per_t_feed(p):
    _check_heat_shares(p)
    return (
        external_heat_gj_per_t_feed(p)
        * max(0.0, 1 - p.electric_heat_share - p.h2_heat_share)
        / (MEOH_LHV_MJ_PER_KG * p.burner_efficiency)
    )


def h2_fuel_per_t_feed(p):
    """Product H2 burned for the H2-fired share of reformer heat [t H2/t MeOH feed]."""
    _check_heat_shares(p)
    return (
        external_heat_gj_per_t_feed(p)
        * p.h2_heat_share
        / (H2_LHV_MJ_PER_KG * p.burner_efficiency)
    )


def net_h2_per_t_feed(p):
    """Delivered H2 per tonne of feed after any H2 burned for heat [t/t]."""
    net = (
        H2_STOICH_KG_PER_KG_MEOH * converted_fraction(p) * p.psa_h2_recovery
        - h2_fuel_per_t_feed(p)
    )
    if net <= 0:
        raise ValueError("H2-fired reformer heat consumes all recovered hydrogen")
    return net


class MeohCracker:
    def __init__(
        self,
        cracker: CrackerParameters,
        economics: Economics,
        plant_capacity_tpd_h2: float,
        meoh_feed_tpd: float,
        installed_heat_pump_capacity_mw_th=None,
    ):
        self.p = cracker
        self.econ = economics
        self.plant_capacity_tpd_h2 = plant_capacity_tpd_h2
        self.meoh_feed_tpd = meoh_feed_tpd
        self.installed_heat_pump_capacity_mw_th = installed_heat_pump_capacity_mw_th

    def run(self) -> CrackerResult:
        p = self.p

        # Capex (six-tenths scaled from reference). All costs in AUD.
        # Reformer and PSA also process the H2 later burned for heat.
        h2_fuel_tpd = self.meoh_feed_tpd * h2_fuel_per_t_feed(p)
        gross_tpd_h2 = self.plant_capacity_tpd_h2 + h2_fuel_tpd
        if not p.min_capacity_tpd_h2-1e-8 <= gross_tpd_h2 <= p.max_capacity_tpd_h2+1e-8:
            raise ValueError("Gross cracker capacity outside workbook Min/Max")
        ref_capex_aud = p.ref_capex_aud_per_tpd_h2 * p.ref_capacity_tpd_h2
        scale_ratio = gross_tpd_h2 / p.ref_capacity_tpd_h2
        capex_aud = ref_capex_aud * (scale_ratio**p.capex_scaling_factor)
        fixed_opex_aud = capex_aud * (p.annual_opex_percent_capex / 100.0)

        # Annual H2 throughput
        if not 0 < p.operating_hours_per_year <= 8760:
            raise ValueError("Invalid cracker operating hours")
        expected_h2 = self.meoh_feed_tpd * net_h2_per_t_feed(p)
        if (
            abs(expected_h2 - self.plant_capacity_tpd_h2)
            > max(1, self.plant_capacity_tpd_h2) * 1e-8
        ):
            raise ValueError("Cracker feed and H2 nameplate do not reconcile")
        op_days_per_year = p.operating_hours_per_year / HOURS_PER_DAY
        annual_h2_tpa = self.plant_capacity_tpd_h2 * op_days_per_year
        annual_h2_kg = annual_h2_tpa * KG_PER_TONNE
        h2_fuel_tpa = h2_fuel_tpd * op_days_per_year

        converted = converted_fraction(p)
        total_co2_tpd = self.meoh_feed_tpd * CO2_STOICH_KG_PER_KG_MEOH
        captured_co2_tpd = self.meoh_feed_tpd * captured_co2_per_t_feed(p)
        effective_capture_rate = (1 if p.co2_capture_config else converted) * p.co2_capture_rate
        vented_co2_tpd = total_co2_tpd * (1-effective_capture_rate)
        annual_captured_co2 = captured_co2_tpd * op_days_per_year
        regen_heat = annual_captured_co2 * p.capture_regen_heat_gj_per_t_co2

        # Reformer heat: gross duty minus tail-gas combustion credit.
        # Surplus tail-gas heat (if conversion is low) is assumed dumped, not exported.
        gross_thermal_gj_per_year = (
            p.thermal_gj_per_t_meoh * self.meoh_feed_tpd * op_days_per_year
        )
        thermal_gj_per_year = (
            external_heat_gj_per_t_feed(p) * self.meoh_feed_tpd * op_days_per_year
        )
        tail_gas_heat_gj_per_year = gross_thermal_gj_per_year - thermal_gj_per_year
        tail_gas_dumped = max(0.0, tail_gas_heat_gj_per_t_feed(p)
            * self.meoh_feed_tpd * op_days_per_year - tail_gas_heat_gj_per_year)
        supplemental_meoh_tpa = (
            supplemental_meoh_per_t_feed(p) * self.meoh_feed_tpd * op_days_per_year
        )
        if not 0 < p.electric_heater_efficiency <= 1:
            raise ValueError("Electric heater efficiency must be in (0, 1]")
        # Electrification displaces only the fired heat. Reaction CO2 from
        # CH3OH + H2O -> CO2 + 3 H2 is unchanged and still meets the capture split.
        heat_electricity_mwh = (
            (thermal_gj_per_year * p.electric_heat_share + regen_heat)
            / 3.6
            / p.electric_heater_efficiency
        )
        reformer_electric_heat = thermal_gj_per_year * p.electric_heat_share / 3.6
        regeneration_heat = regen_heat / 3.6
        resistance_heat_mwh = reformer_electric_heat + regeneration_heat
        hp_heat = hp_source = hp_power = hp_capacity = hp_capex = hp_annual_capex = 0.0
        hp_source_limited = False
        if p.heat_pump_enabled not in (0, 1):
            raise ValueError("Heat pump switch must be 0 or 1")
        if p.heat_pump_enabled:
            hp_inputs = (p.heat_pump_cop, p.heat_pump_eligible_heat_share,
                         p.heat_pump_source_gj_per_t_meoh, p.heat_pump_capex_aud_per_kw_th,
                         p.heat_pump_installation_factor, p.heat_pump_opex_percent_capex,
                         p.heat_pump_life_years)
            if (not all(math.isfinite(v) for v in hp_inputs)
                    or p.heat_pump_cop <= 1
                    or not 0 <= p.heat_pump_eligible_heat_share <= 1
                    or p.heat_pump_source_gj_per_t_meoh < 0
                    or p.heat_pump_capex_aud_per_kw_th < 0
                    or p.heat_pump_installation_factor < 1
                    or p.heat_pump_opex_percent_capex < 0
                    or p.heat_pump_life_years <= 0):
                raise ValueError("Invalid heat pump assumptions")
            hp_crf = Economics.crf(self.econ.discount_rate, p.heat_pump_life_years)
            installed_per_kw = p.heat_pump_capex_aud_per_kw_th * p.heat_pump_installation_factor
            capital_and_om_per_kwh = (installed_per_kw
                * (hp_crf + p.heat_pump_opex_percent_capex / 100) / p.operating_hours_per_year)
            saved_power_per_kwh = 1 / p.electric_heater_efficiency - 1 / p.heat_pump_cop
            # Linear installed cost gives an endpoint optimum: zero or the duty/source limit.
            # Source heat is residual cooling heat, explicitly additional to the tail-gas credit.
            installed = self.installed_heat_pump_capacity_mw_th
            if installed is not None and (not math.isfinite(installed) or installed < 0):
                raise ValueError("Invalid installed heat pump capacity")
            if installed is not None or saved_power_per_kwh * p.power_price_aud_per_kwh > capital_and_om_per_kwh:
                source_available = (p.heat_pump_source_gj_per_t_meoh
                    * self.meoh_feed_tpd * op_days_per_year / 3.6)
                eligible_heat = regeneration_heat + reformer_electric_heat*p.heat_pump_eligible_heat_share
                source_limit = source_available / (1 - 1 / p.heat_pump_cop)
                hp_heat = min(eligible_heat, source_limit)
                if installed is not None:
                    hp_heat = min(hp_heat, installed * p.operating_hours_per_year)
                hp_power = hp_heat / p.heat_pump_cop
                hp_source_limited = eligible_heat > 0 and source_limit <= hp_heat + 1e-8
                hp_source = hp_heat - hp_power
                hp_capacity = hp_heat / p.operating_hours_per_year if installed is None else installed
                hp_capex = hp_capacity * 1000 * installed_per_kw
                hp_annual_capex = hp_capex * hp_crf
                resistance_heat_mwh -= hp_heat
                heat_electricity_mwh = resistance_heat_mwh / p.electric_heater_efficiency + hp_power
                fixed_opex_aud += hp_capex * p.heat_pump_opex_percent_capex / 100

        hp_regen = min(hp_heat, regeneration_heat)
        hp_reformer = hp_heat-hp_regen
        regeneration_power = ((regeneration_heat-hp_regen)/p.electric_heater_efficiency
                              + (hp_regen/p.heat_pump_cop if hp_regen else 0))
        reformer_power = ((reformer_electric_heat-hp_reformer)/p.electric_heater_efficiency
                         + (hp_reformer/p.heat_pump_cop if hp_reformer else 0))

        capture_power = annual_captured_co2 * p.capture_electricity_kwh_per_t_co2 / 1000
        capture_cost = annual_captured_co2 * p.capture_cost_aud_per_t_co2
        annual_electricity_kwh = (
            (annual_h2_kg + h2_fuel_tpa * KG_PER_TONNE) * p.parasitic_kwh_per_kg_h2
            + (heat_electricity_mwh + capture_power) * 1000.0
        )
        annual_electricity_mwh = annual_electricity_kwh / 1000.0
        annual_electricity_cost_aud = (
            annual_electricity_kwh * p.power_price_aud_per_kwh
        )
        electricity_co2_kg = annual_electricity_kwh * p.power_ei_kg_co2_per_kwh
        electricity_co2_tpa = electricity_co2_kg / KG_PER_TONNE
        electricity_cost_per_kg_h2 = annual_electricity_cost_aud / annual_h2_kg
        electricity_ei_per_kg_h2 = electricity_co2_kg / annual_h2_kg

        # Internal methanol fuel production is charged once in integrated LCOMeOH.
        # Flue-gas capture also removes carbon from supplemental methanol fuel.
        annual_supplemental_heat_cost_aud = 0.0
        supplemental_heat_co2_tpa = supplemental_meoh_tpa * CO2_STOICH_KG_PER_KG_MEOH * (
            1-p.co2_capture_rate if p.co2_capture_config else 1)
        supplemental_heat_cost_per_kg_h2 = 0.0
        supplemental_heat_ei_per_kg_h2 = (
            supplemental_heat_co2_tpa * KG_PER_TONNE / annual_h2_kg
        )
        water_cost = (
            self.meoh_feed_tpd
            * op_days_per_year
            * converted
            * M_H2O
            / M_MEOH
            * p.steam_water_cost_aud_per_kl
        )

        # Total opex and levelised cost (all AUD)
        annual_opex_aud = fixed_opex_aud + annual_electricity_cost_aud + water_cost + capture_cost
        crf = Economics.crf(self.econ.discount_rate, self.econ.cracker_life_years)
        annualised_capex_aud = capex_aud * crf + hp_annual_capex
        capex_aud += hp_capex
        levelised_aud_per_kg = (annualised_capex_aud + annual_opex_aud) / annual_h2_kg

        return CrackerResult(
            plant_capacity_tpd_h2=self.plant_capacity_tpd_h2,
            gross_capacity_tpd_h2=gross_tpd_h2,
            annual_h2_tpa=annual_h2_tpa,
            capex_aud=capex_aud,
            crf=crf,
            annual_opex_aud=annual_opex_aud,
            annualised_capex_aud=annualised_capex_aud,
            levelised_cost_aud_per_kg_h2=levelised_aud_per_kg,
            captured_co2_tpd=captured_co2_tpd,
            vented_co2_tpd=vented_co2_tpd,
            effective_capture_rate=effective_capture_rate,
            recycle_meoh_tpd=(self.meoh_feed_tpd*(1/p.cracker_conversion_rate-1)
                              if p.methanol_recycle_enabled else 0.0),
            capture_regen_heat_gj_per_year=regen_heat,
            capture_electricity_mwh=capture_power,
            annual_capture_cost_aud=capture_cost,
            regeneration_electricity_mwh=regeneration_power,
            reformer_electricity_mwh=reformer_power,
            heat_pump_regeneration_heat_mwh=hp_regen,
            heat_pump_source_limited=hp_source_limited,
            tail_gas_heat_capped=tail_gas_heat_gj_per_t_feed(p) >= p.thermal_gj_per_t_meoh,
            co2_capture_config=int(p.co2_capture_config),
            cracker_power_source=int(p.cracker_power_source),
            power_price_aud_per_kwh=p.power_price_aud_per_kwh,
            power_ei_kg_co2_per_kwh=p.power_ei_kg_co2_per_kwh,
            tail_gas_heat_dumped_gj_per_year=tail_gas_dumped,
            gross_thermal_gj_per_year=gross_thermal_gj_per_year,
            tail_gas_heat_credit_gj_per_year=tail_gas_heat_gj_per_year,
            thermal_gj_per_year=thermal_gj_per_year,
            heat_electricity_mwh=heat_electricity_mwh,
            resistance_heat_mwh=resistance_heat_mwh,
            heat_pump_heat_mwh=hp_heat,
            heat_pump_source_mwh=hp_source,
            heat_pump_electricity_mwh=hp_power,
            heat_pump_capacity_mw_th=hp_capacity,
            heat_pump_capex_aud=hp_capex,
            heat_pump_annualised_capex_aud=hp_annual_capex,
            h2_fuel_tpa=h2_fuel_tpa,
            annual_electricity_mwh=annual_electricity_mwh,
            annual_electricity_cost_aud=annual_electricity_cost_aud,
            electricity_co2_tpa=electricity_co2_tpa,
            electricity_cost_aud_per_kg_h2=electricity_cost_per_kg_h2,
            electricity_ei_kg_co2_per_kg_h2=electricity_ei_per_kg_h2,
            annual_supplemental_heat_cost_aud=annual_supplemental_heat_cost_aud,
            supplemental_heat_co2_tpa=supplemental_heat_co2_tpa,
            supplemental_heat_cost_aud_per_kg_h2=supplemental_heat_cost_per_kg_h2,
            supplemental_heat_ei_kg_co2_per_kg_h2=supplemental_heat_ei_per_kg_h2,
        )
