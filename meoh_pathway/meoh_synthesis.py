"""Synthesis costs and fresh stoichiometric feeds adjusted by overall yield."""

import math
from dataclasses import dataclass

from .parameters import MeohSynthParameters, Economics, load_synth, load_economics

from constants import (
    KG_PER_TONNE,
    HOURS_PER_DAY,
    HOURS_PER_YEAR,
    CO2_STOICH_KG_PER_KG_MEOH,
)


def synth_feed_ratios(p: MeohSynthParameters):
    """Fresh feed per unit product; overall yield includes recycle/purge loss."""
    from constants import CO2_STOICH_KG_PER_KG_MEOH, H2_STOICH_KG_PER_KG_MEOH

    if not 0 < p.overall_co2_yield_percent <= 100:
        raise ValueError("Overall synthesis yield must be in (0, 100]")
    yield_fraction = p.overall_co2_yield_percent / 100.0
    return (
        H2_STOICH_KG_PER_KG_MEOH / yield_fraction,
        CO2_STOICH_KG_PER_KG_MEOH / yield_fraction,
    )


@dataclass
class MeohSynthResult:
    """Methanol synthesis plant performance and economics."""

    # Production capacity and throughput
    plant_capacity_tpy_meoh: float  # Nameplate capacity [t/year] @ 8760 hr/yr
    annual_meoh_tpa: float  # Annual methanol production [t/year]
    capacity_factor: float  # Capacity factor [fraction] = op_hrs / 8760

    # Feed requirements [t/day, rated when running]
    h2_feed_tpd: float  # H2 feed requirement
    co2_feed_tpd: float  # CO2 feed requirement

    # Process CO2e: CO2 fed but not incorporated into MeOH (vented via purge)
    process_co2e_t_per_t_meoh: float

    # Energy consumption [annual]
    electricity_kwh_per_year: float  # Annual electricity [kWh]
    heat_kwh_per_year: float  # Annual heat [kWh]

    # Economic metrics
    capex_aud: float  # Total capital cost [AUD]
    capex_annualised_aud_per_year: float  # Annualised capex [AUD/year]
    fom_aud_per_year: float  # Fixed O&M [AUD/year]
    vom_aud_per_year: float  # Variable O&M [AUD/year]
    catalyst_replacement_aud_per_year: float  # Catalyst replacement [AUD/year]

    synth_plant_aud_per_kg_meoh: float  # Total cost per kg MeOH [AUD/kg]


class MeohSynthesis:
    """Calculates methanol synthesis plant performance based on stoichiometry."""

    def __init__(
        self,
        synth: MeohSynthParameters,
        economics: Economics,
    ):
        self.p = synth
        self.econ = economics

    def run(self) -> MeohSynthResult:
        p = self.p
        if (
            not 0 < p.operating_hours_per_year <= HOURS_PER_YEAR
            or not 0 < p.heat_efficiency <= 1
            or not 0 <= p.min_load_percent <= 100
        ):
            raise ValueError(
                "Invalid synthesis operating hours, heater efficiency or minimum load"
            )

        # capacity_tpy is the 8760 hr/yr nameplate. Capacity factor = op_hrs / 8760;
        # annual production = nameplate * capacity_factor. Rated daily flow when running
        # is nameplate / 365 (i.e. design tonnes/hr * 24), independent of capacity factor.
        capacity_factor = p.operating_hours_per_year / HOURS_PER_YEAR
        annual_meoh_tpa = p.capacity_tpy * capacity_factor
        daily_meoh_tpd = p.capacity_tpy * (HOURS_PER_DAY / HOURS_PER_YEAR)

        # Feed mass balance (see synth_feed_ratios). Reported as rated daily
        # flows when running.
        h2_feed_per_kg_meoh, co2_feed_per_kg_meoh = synth_feed_ratios(p)
        co2_feed_tpd = daily_meoh_tpd * co2_feed_per_kg_meoh
        h2_feed_tpd = daily_meoh_tpd * h2_feed_per_kg_meoh

        # CO2 fed but not incorporated into MeOH (vented or oxidised from purge).
        process_co2e_t_per_t_meoh = co2_feed_per_kg_meoh - CO2_STOICH_KG_PER_KG_MEOH

        # --- ENERGY CONSUMPTION [annual] --- scales with actual annual production.
        electricity_kwh_per_year = annual_meoh_tpa * p.sec_electricity_kwh_per_t
        heat_kwh_per_year = annual_meoh_tpa * p.sec_heat_kwh_per_t

        # --- ECONOMICS ---
        # Capital cost scaled from reference plant
        capex_aud = (
            p.ref_capex_aud
            * (p.capacity_tpy / p.ref_capacity_tpy) ** p.capex_scaling_exponent
        )

        crf = Economics.crf(self.econ.discount_rate, p.plant_life_years)
        capex_annualised_aud = capex_aud * crf

        # Fixed O&M tied to nameplate capex.
        fom_aud_per_year = capex_aud * (p.fom_percent_capex_per_year / 100.0)

        # Variable O&M scales with actual annual production.
        vom_aud_per_year = annual_meoh_tpa * p.vom_aud_per_t

        # Catalyst inventory sized off nameplate. Catalyst ages on calendar time
        # as well as throughput, so reload intervals are fixed at catalyst_life_years
        # regardless of utilisation. Initial inventory is bundled into plant CAPEX;
        # each subsequent reload at k * catalyst_life_years is discounted to PV then
        # annualised via CRF over plant life (mirrors PEM stack replacement).
        catalyst_inventory_tonnes = (
            p.capacity_tpy * p.catalyst_inventory_g_per_tpy
        ) / 1e6
        catalyst_replacement_cost_aud = (
            catalyst_inventory_tonnes * p.catalyst_cost_aud_per_tonne
        )
        r = self.econ.discount_rate
        n_catalyst_replacements = max(
            0, math.ceil(p.plant_life_years / p.catalyst_life_years) - 1
        )
        pv_catalyst = sum(
            catalyst_replacement_cost_aud / (1.0 + r) ** (k * p.catalyst_life_years)
            for k in range(1, n_catalyst_replacements + 1)
        )
        catalyst_replacement_aud_per_year = pv_catalyst * crf

        total_annual_cost_aud = (
            capex_annualised_aud
            + fom_aud_per_year
            + vom_aud_per_year
            + catalyst_replacement_aud_per_year
        )
        synth_plant_aud_per_kg_meoh = total_annual_cost_aud / (
            annual_meoh_tpa * KG_PER_TONNE
        )

        return MeohSynthResult(
            plant_capacity_tpy_meoh=p.capacity_tpy,
            annual_meoh_tpa=annual_meoh_tpa,
            capacity_factor=capacity_factor,
            h2_feed_tpd=h2_feed_tpd,
            co2_feed_tpd=co2_feed_tpd,
            process_co2e_t_per_t_meoh=process_co2e_t_per_t_meoh,
            electricity_kwh_per_year=electricity_kwh_per_year,
            heat_kwh_per_year=heat_kwh_per_year,
            capex_aud=capex_aud,
            capex_annualised_aud_per_year=capex_annualised_aud,
            fom_aud_per_year=fom_aud_per_year,
            vom_aud_per_year=vom_aud_per_year,
            catalyst_replacement_aud_per_year=catalyst_replacement_aud_per_year,
            synth_plant_aud_per_kg_meoh=synth_plant_aud_per_kg_meoh,
        )


def main():
    """Standalone run of one nameplate synth module (no fleet scaling, no utilisation trim).

    Usage:
        python -m meoh_pathway.meoh_synthesis
    """
    synth = load_synth()
    econ = load_economics()
    r = MeohSynthesis(synth, econ).run()

    print("\n--- MeOH synthesis (single nameplate module) ---")
    print("\nCapacity")
    print(f"  Nameplate (8760 hr)   {r.plant_capacity_tpy_meoh / 1e3:>10,.1f} kt/yr")
    print(
        f"  Annual production     {r.annual_meoh_tpa / 1e3:>10,.1f} kt/yr   "
        f"(op hrs {synth.operating_hours_per_year:,.0f}, CF {r.capacity_factor * 100:.1f}%)"
    )

    print("\nDaily flows (when running)")
    print(f"  H2 feed               {r.h2_feed_tpd:>10,.1f} t/d")
    print(f"  CO2 feed              {r.co2_feed_tpd:>10,.1f} t/d")
    print(f"  Process CO2e (vent)   {r.process_co2e_t_per_t_meoh:>10,.3f} t/t MeOH")

    print("\nEnergy (annual)")
    print(f"  Electricity           {r.electricity_kwh_per_year / 1e6:>10,.1f} GWh/yr")
    print(f"  Heat                  {r.heat_kwh_per_year / 1e6:>10,.1f} GWh/yr")

    print("\nEconomics")
    print(
        f"  CAPEX                 {r.capex_aud / r.plant_capacity_tpy_meoh:>10,.0f} AUD/tpa"
    )
    print(
        f"  FOM                   {r.fom_aud_per_year / r.plant_capacity_tpy_meoh:>10,.0f} AUD/tpa"
    )
    print(
        f"  VOM                   {r.vom_aud_per_year / r.annual_meoh_tpa:>10,.0f} AUD/t"
    )
    print(
        f"  Catalyst              {r.catalyst_replacement_aud_per_year / r.annual_meoh_tpa:>10,.0f} AUD/t"
    )
    print(
        f"  Total                 {r.synth_plant_aud_per_kg_meoh:>10,.3f} AUD/kg MeOH"
    )


if __name__ == "__main__":
    main()
