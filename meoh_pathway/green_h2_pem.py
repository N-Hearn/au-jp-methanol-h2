"""Green H2 from PEM electrolysis (single module, single plant).

PEM stack-level conversion: given nameplate capacity and equivalent
full-load operating hours per year (a PemParameters input), this module
returns annual H2 production, electricity demand, water demand, and the
PEM CAPEX/OPEX/electricity/water costs.

Renewable capacity sizing, battery balancing, and H2 storage sizing all
happen at the supply-chain level (see supply_chain.py) - they depend on
how PEM couples to downstream synth and don't belong here.
"""

import math
from dataclasses import dataclass

from .parameters import PemParameters
from constants import HOURS_PER_YEAR, DAYS_PER_YEAR, KG_PER_TONNE, KW_PER_MW, L_PER_KL
from .costing import scaled_unit_cost


def _crf(r: float, n: int) -> float:
    from .costing import capital_recovery

    return capital_recovery(r, n)


def life_average_sec(p):
    """Linear SEC ageing over one stack life, with throughput-based replacement."""
    if (
        p.stack_life_hours <= 0
        or p.sec_kwh_per_kg_h2 <= 0
        or p.degradation_percent_per_1000_hours < 0
    ):
        raise ValueError("Invalid PEM efficiency or lifetime")
    return p.sec_kwh_per_kg_h2 * (
        1 + 0.5 * p.stack_life_hours / 1000 * p.degradation_percent_per_1000_hours / 100
    )


def bop_capex_aud_per_kw(p, module_mw):
    """Balance-of-plant capital per kW, log-log between the two module sizes."""
    return scaled_unit_cost(module_mw, p.small_module_mw, p.bop_capex_aud_per_kw_small,
                            p.large_module_mw, p.bop_capex_aud_per_kw_large)


def module_capex_aud_per_kw(p, module_mw):
    """Installed capital per kW of one module.

    capex_aud_per_kw applies at capex_reference_mw. Stacks repeat with size,
    so only the balance-of-plant difference moves the cost.
    """
    cost = (p.capex_aud_per_kw + bop_capex_aud_per_kw(p, module_mw)
            - bop_capex_aud_per_kw(p, p.capex_reference_mw))
    if not math.isfinite(cost) or cost <= 0:
        raise ValueError("PEM module capital must be positive")
    return cost


def module_layout(p, nameplate_mw):
    """Split a nameplate into large plants first, then small ones."""
    per_large = p.large_module_mw / p.small_module_mw
    if not 0 < p.small_module_mw <= p.large_module_mw or not nameplate_mw >= 0:
        raise ValueError("PEM modules need 0 < small <= large size and a non-negative nameplate")
    if abs(per_large - round(per_large)) > 1e-9:
        raise ValueError("Large PEM module must be a whole number of small modules")
    small = round(nameplate_mw / p.small_module_mw)
    if abs(small * p.small_module_mw - nameplate_mw) > 1e-6:
        raise ValueError("PEM nameplate must be a whole number of small modules")
    large, small = divmod(small, round(per_large))
    return large, small


def pem_capex_aud(p, nameplate_mw):
    """Installed capital of a site's PEM, each plant costed at its own size."""
    large, small = module_layout(p, nameplate_mw)
    return KW_PER_MW * (large * p.large_module_mw * module_capex_aud_per_kw(p, p.large_module_mw)
                        + small * p.small_module_mw * module_capex_aud_per_kw(p, p.small_module_mw))


def module_label(large, large_module_mw, small, small_module_mw):
    parts = [f"{large} x {large_module_mw:,.0f} MW"] if large else []
    if small or not parts:
        parts.append(f"{small} x {small_module_mw:,.0f} MW")
    return " + ".join(parts)


@dataclass
class PemResult:
    small_module_mw: float
    sec_kwh_per_kg_h2: float  # life-average (degradation-adjusted)
    operating_hours_per_year: float
    capacity_factor: float

    annual_energy_mwh: float
    annual_h2_kg: float
    h2_output_tpd: float

    water_kl_per_year: float
    annual_water_cost_aud: float

    capex_aud: float
    annual_opex_aud: float
    annual_stack_replacement_aud: float
    annual_electricity_cost_aud: float


class PemElectrolyser:
    def __init__(
        self,
        params: PemParameters,
        power_price_aud_per_kwh: float,
        discount_rate: float,
    ):
        self.p = params
        self.power_price_aud_per_kwh = power_price_aud_per_kwh
        self.discount_rate = discount_rate

    def run(self) -> PemResult:
        p = self.p
        r = self.discount_rate
        op_hr = p.operating_hours_per_year

        if not 0 < op_hr <= HOURS_PER_YEAR:
            raise ValueError("PEM full-load hours must be in (0, 8760]")
        stack_life_years = p.stack_life_hours / op_hr
        avg_sec = life_average_sec(p)

        # --- Annual electricity, H2, and water -----------------------------
        # PEM runs at nameplate for op_hr full-load-equivalent hours per year.
        annual_energy_mwh = p.small_module_mw * op_hr
        annual_h2_kg = annual_energy_mwh * KW_PER_MW / avg_sec
        h2_tpd = annual_h2_kg / DAYS_PER_YEAR / KG_PER_TONNE

        water_kl = annual_h2_kg * p.water_l_per_kg_h2 / L_PER_KL
        annual_water_cost_aud = water_kl * p.water_cost_aud_per_kl

        # --- CAPEX / fixed OPEX / electricity OPEX -------------------------
        capex_aud = p.small_module_mw * KW_PER_MW * p.capex_aud_per_kw
        opex_aud = capex_aud * p.opex_percent_capex_per_year / 100.0
        electricity_cost_aud = (
            annual_energy_mwh * KW_PER_MW * self.power_price_aud_per_kwh
        )

        # --- Stack replacements --------------------------------------------
        # Each replacement costs stack_replacement_percent of CAPEX, paid
        # at year k * stack_life_years for k = 1..n. Discount each PV back
        # to t=0 then annuitise over system life.
        replacement_cost = capex_aud * p.stack_replacement_percent / 100
        # ceil(life/stack)-1 so a replacement at exactly t=system_life isn't counted.
        n_replacements = max(0, math.ceil(p.system_life_years / stack_life_years) - 1)
        pv_stack = sum(
            replacement_cost / (1 + r) ** (k * stack_life_years)
            for k in range(1, n_replacements + 1)
        )
        annual_stack_aud = (
            pv_stack * _crf(r, p.system_life_years) if n_replacements > 0 else 0.0
        )

        return PemResult(
            small_module_mw=p.small_module_mw,
            sec_kwh_per_kg_h2=avg_sec,
            operating_hours_per_year=op_hr,
            capacity_factor=op_hr / HOURS_PER_YEAR,
            annual_energy_mwh=annual_energy_mwh,
            annual_h2_kg=annual_h2_kg,
            h2_output_tpd=h2_tpd,
            water_kl_per_year=water_kl,
            annual_water_cost_aud=annual_water_cost_aud,
            capex_aud=capex_aud,
            annual_opex_aud=opex_aud,
            annual_stack_replacement_aud=annual_stack_aud,
            annual_electricity_cost_aud=electricity_cost_aud,
        )


def main():
    """Standalone run of one nameplate PEM module (no fleet scaling, no renewables sizing).

    Usage:
        python -m meoh_pathway.green_h2_pem
    """
    from .parameters import load_pem, load_economics

    pem = load_pem()
    econ = load_economics()
    power_price_aud = econ.power_price_aud_per_kwh
    r = PemElectrolyser(pem, power_price_aud, econ.discount_rate).run()

    crf = _crf(econ.discount_rate, pem.system_life_years)
    annualised_capex_aud = r.capex_aud * crf
    annual_total_aud = (
        annualised_capex_aud
        + r.annual_opex_aud
        + r.annual_stack_replacement_aud
        + r.annual_electricity_cost_aud
        + r.annual_water_cost_aud
    )
    lcoh_aud_per_kg = annual_total_aud / r.annual_h2_kg

    print("\nPEM electrolyser (single nameplate module)")
    print("\nCapacity")
    print(f"  Nameplate             {r.small_module_mw:>10,.1f} MW")
    print(
        f"  Operating hours       {r.operating_hours_per_year:>10,.0f} h/yr  "
        f"(CF {r.capacity_factor * 100:.1f}%)"
    )
    print(
        f"  Life-avg SEC          {r.sec_kwh_per_kg_h2:>10,.2f} kWh/kg H2  "
        f"(nameplate {pem.sec_kwh_per_kg_h2:.2f})"
    )

    print("\nProduction (annual)")
    print(f"  Energy in             {r.annual_energy_mwh / 1e3:>10,.1f} GWh/yr")
    print(
        f"  H2 out                {r.annual_h2_kg / 1e6:>10,.2f} kt/yr  "
        f"({r.h2_output_tpd:,.1f} t/d)"
    )
    print(f"  Water                 {r.water_kl_per_year / 1e3:>10,.1f} ML/yr")

    print("\nEconomics (PEM-only, excludes renewables / battery / H2 storage)")
    print(f"  CAPEX                 {r.capex_aud / 1e6:>10,.1f} M AUD")
    print(f"  Annualised CAPEX      {annualised_capex_aud / 1e6:>10,.1f} M AUD/yr")
    print(f"  Fixed OPEX            {r.annual_opex_aud / 1e6:>10,.1f} M AUD/yr")
    print(
        f"  Stack replacements    {r.annual_stack_replacement_aud / 1e6:>10,.1f} M AUD/yr"
    )
    print(
        f"  Electricity           {r.annual_electricity_cost_aud / 1e6:>10,.1f} M AUD/yr  "
        f"(@ {power_price_aud * 1000:.1f} AUD/MWh, legacy memo price)"
    )
    print(f"  Water                 {r.annual_water_cost_aud / 1e6:>10,.2f} M AUD/yr")
    print(f"  LCOH (PEM-only)       {lcoh_aud_per_kg:>10,.3f} AUD/kg H2")


if __name__ == "__main__":
    main()
