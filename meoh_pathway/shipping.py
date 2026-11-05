"""Dual-purpose carriers: methanol Australia -> Japan, liquid CO2 back.

One low-pressure LCO2 carrier design (~7 bara, -50 C tanks) carries both
cargoes, so neither leg sails in ballast. Methanol is an ambient liquid the
pressurised tanks hold as-is. Per-vessel CAPEX and OPEX are absolute (one
reference design, no 6/10ths scaling).

Each leg's payload is tank volume * cargo density. The hull is rated for a
full LCO2 load plus bunkers, as LCO2 carriers are, so weight never binds and
DWT is not an input. The fleet is sized on whichever leg needs more voyages.

Refuelling: green methanol is bunkered in Australia for the round trip and
drawn from the same green-MeOH stream, priced at LCOMeOH (USD/t) passed in by
the supply-chain wrapper. Topping up from cargo in Japan instead moves the same
tonnes, so the balance does not change. The laden fuel rate applies on both
legs. BOG on the methanol leg is treated as pure cargo loss. With fuel_type
"fuel_oil" the HFO-equivalent burn is bought at fuel_oil_price_usd_per_t
instead, and no methanol is bunkered.

voyage_co2_kg_per_kg_meoh is derived here (fuel burn * combustion EF / cargo)
so cost and emissions stay consistent with the same fuel-burn assumption.

Fleet: a dedicated per-port fleet pays for ceil(ships needed). With
pooled_fleet the port pays for its fractional ship-years; the supply chain
rounds the shared fleet up once.
"""

import math
from dataclasses import dataclass

from .parameters import Economics, ShipParameters
from constants import (
    MEOH_LHV_MJ_PER_KG,
    HOURS_PER_DAY,
    KG_PER_TONNE,
    MEOH_DENSITY_KG_PER_M3,
    MEOH_COMBUSTION_CO2_EF_KG_PER_KG,
    FUEL_OIL_CO2_EF_KG_PER_KG,
)


@dataclass
class ShipResult:
    """Carrier fleet performance and economics."""

    # Vessel design
    carrier_cargo_volume_m3: float  # Cargo tank volume [m3]
    payload_per_voyage_t: float  # Methanol payload per voyage [t]
    co2_payload_per_voyage_t: float  # LCO2 payload per voyage [t]

    # Voyage timing
    voyage_distance_nm_round: float  # Round-trip distance [nm]
    sea_days_per_voyage: float  # Time at sea [d]
    loaded_days: float  # Loaded leg only (BOG applies here) [d]
    total_days_per_voyage: float  # Sea + port turnaround [d]
    voyages_per_ship_per_year: float  # Round trips per ship [voyages/yr]
    annual_cargo_per_ship_tpa: float  # Methanol capacity per ship [t/yr]
    annual_co2_per_ship_tpa: float  # LCO2 capacity per ship [t/yr]

    # Fleet sizing
    annual_meoh_shipped_tpa: float  # Total MeOH to be shipped [t/yr]
    annual_co2_shipped_tpa: float  # Total LCO2 to be returned [t/yr]
    ships_required_fractional: float  # Fractional fleet count
    ships_required_integer: int  # Integer fleet count
    ships_charged: float  # Ships this port pays for (fractional share if pooled)
    annual_cost_per_ship_aud: float  # Annualised CAPEX + fixed OPEX per ship [AUD/yr]

    # BOG and fuel
    bog_fraction_per_voyage: float  # Cargo lost as BOG per voyage [fraction]
    annual_bog_per_ship_t: float  # BOG cargo loss per ship [t/yr]
    annual_sea_fuel_per_ship_t: float  # Main-engine fuel per ship [t/yr]
    annual_port_fuel_per_ship_t: float  # Auxiliary fuel per ship [t/yr]
    annual_fuel_per_ship_t: float  # Sea + port fuel per ship [t/yr]
    annual_meoh_bunker_per_ship_t: float  # Of which methanol drawn from cargo [t/yr]
    annual_fossil_fuel_co2_tpa: float  # Fleet fuel-oil CO2 [t/yr]; zero for methanol

    # Emissions
    voyage_co2_kg_per_kg_meoh: float  # Voyage CO2 intensity [kg CO2/kg MeOH]

    # Economic metrics (per kg of MeOH shipped)
    ship_capex_aud_per_vessel: float  # Newbuild CAPEX per ship [AUD]
    ship_cost_usd_per_kg_meoh: float  # CAPEX + fixed OPEX share [USD/kg MeOH]
    ship_fuel_cost_usd_per_kg_meoh: float  # Bought bunker fuel share; zero for own methanol [USD/kg MeOH]
    total_ship_cost_usd_per_kg_meoh: float  # Total shipping cost [USD/kg MeOH]


class Shipping:
    """Carrier fleet sized to the busier leg, with cost and voyage CO2."""

    def __init__(
        self,
        ship: ShipParameters,
        economics: Economics,
        annual_meoh_shipped_tpa: float,
        ship_fuel_price_usd_per_t: float,
        installed_ships=None,
        annual_co2_shipped_tpa=0.0,
    ):
        self.p = ship
        self.econ = economics
        self.annual_meoh_tpa = annual_meoh_shipped_tpa
        self.ship_fuel_price_usd_per_t = ship_fuel_price_usd_per_t
        self.installed_ships = installed_ships
        self.annual_co2_tpa = annual_co2_shipped_tpa

    def run(self) -> ShipResult:
        p = self.p
        if p.fuel_type not in ("methanol", "fuel_oil"):
            raise ValueError(f"Unknown ship fuel {p.fuel_type!r}")
        if p.fuel_type == "fuel_oil" and p.fuel_oil_price_usd_per_t <= 0:
            raise ValueError("Fuel-oil bunkers need a positive fuel price")
        if p.pooled_fleet not in (0, 1):
            raise ValueError("Pooled fleet switch must be 0 or 1")
        if any(not math.isfinite(v) for k, v in vars(p).items() if k != "fuel_type"):
            raise ValueError("Shipping inputs must be finite")
        if (not 0 < p.available_days_per_year <= 365 or p.port_turnaround_days < 0
                or min(p.sea_fuel_consumption_tpd, p.port_fuel_consumption_tpd,
                       p.boiloff_rate_pct_per_day) < 0):
            raise ValueError("Invalid vessel availability, turnaround or fuel/loss rate")
        if (
            min(
                p.voyage_speed_kn,
                p.available_days_per_year,
                p.shipping_distance_nm,
                self.annual_meoh_tpa,
                p.carrier_cargo_volume_m3,
                p.co2_cargo_density_kg_per_m3,
            )
            <= 0
            or not math.isfinite(self.annual_co2_tpa)
            or self.annual_co2_tpa < 0
        ):
            raise ValueError("Invalid shipping throughput or voyage inputs")

        # --- Payload --------------------------------------------------------
        payload_t = p.carrier_cargo_volume_m3 * MEOH_DENSITY_KG_PER_M3 / KG_PER_TONNE
        co2_payload_t = p.carrier_cargo_volume_m3 * p.co2_cargo_density_kg_per_m3 / KG_PER_TONNE

        # --- Voyage timing --------------------------------------------------
        sea_days = 2.0 * p.shipping_distance_nm / (p.voyage_speed_kn * HOURS_PER_DAY)
        total_days = sea_days + p.port_turnaround_days
        voyages_per_ship = p.available_days_per_year / total_days
        annual_cargo_per_ship = voyages_per_ship * payload_t
        annual_co2_per_ship = voyages_per_ship * co2_payload_t

        # --- Fleet sizing (busier leg) ---------------------------------------
        ships_required_frac = max(self.annual_meoh_tpa / annual_cargo_per_ship,
                                  self.annual_co2_tpa / annual_co2_per_ship)
        ships_required_int = math.ceil(ships_required_frac)
        ships_charged = ships_required_frac if p.pooled_fleet else ships_required_int
        if self.installed_ships is not None:
            if ships_required_frac > self.installed_ships * (1+1e-9):
                raise ValueError(
                    f"Installed carriers: {self.installed_ships} ships, "
                    f"{ships_required_frac:.2f} required for {self.annual_meoh_tpa:.0f} t/y methanol "
                    f"and {self.annual_co2_tpa:.0f} t/y CO2")
            ships_charged = self.installed_ships
            ships_required_int = math.ceil(self.installed_ships - 1e-9)

        # --- CAPEX + fixed OPEX (absolute per vessel, no scaling) ----------
        # Annualise vessel CAPEX over ship life; OPEX is already AUD/yr.
        crf = Economics.crf(self.econ.discount_rate, p.ship_lifetime_years)
        annual_capex_per_ship_aud = p.ref_ship_capex_aud * crf
        annual_cost_per_ship_aud = annual_capex_per_ship_aud + p.ship_annual_opex_aud

        fleet_annual_cost_aud = ships_charged * annual_cost_per_ship_aud
        fleet_annual_cost_usd = fleet_annual_cost_aud * self.econ.aud_usd_exchange_rate
        ship_cost_per_kg_meoh = fleet_annual_cost_usd / (
            self.annual_meoh_tpa * KG_PER_TONNE
        )

        # --- BOG (loaded leg only) ------------------------------------------
        # One-way transit + half port time; pure cargo loss for ambient MeOH.
        loaded_days = sea_days / 2.0 + p.port_turnaround_days / 2.0
        bog_frac = p.boiloff_rate_pct_per_day / 100.0 * loaded_days
        if not 0 <= bog_frac < 1:
            raise ValueError("Methanol cargo loss per voyage must be in [0,1)")
        annual_bog_per_ship_t = voyages_per_ship * payload_t * bog_frac

        # --- Fuel (sea + port) ----------------------------------------------
        # Consumption rates are HFO-equivalent; methanol burns the same energy.
        methanol = p.fuel_type == "methanol"
        to_fuel = p.reference_fuel_lhv_mj_per_kg / MEOH_LHV_MJ_PER_KG if methanol else 1.0
        annual_sea_fuel_per_ship_t = (
            voyages_per_ship * sea_days * p.sea_fuel_consumption_tpd * to_fuel
        )
        annual_port_fuel_per_ship_t = (
            voyages_per_ship * p.port_turnaround_days * p.port_fuel_consumption_tpd * to_fuel
        )
        annual_fuel_per_ship_t = (
            annual_sea_fuel_per_ship_t + annual_port_fuel_per_ship_t
        )

        # Methanol is priced at LCOMeOH by the caller; fuel oil at market.
        fleet_annual_fuel_t = ships_required_frac * annual_fuel_per_ship_t
        fuel_price = self.ship_fuel_price_usd_per_t if methanol else p.fuel_oil_price_usd_per_t
        fleet_annual_fuel_cost_usd = fleet_annual_fuel_t * fuel_price
        fuel_cost_per_kg_meoh = fleet_annual_fuel_cost_usd / (
            self.annual_meoh_tpa * KG_PER_TONNE
        )

        # --- Voyage CO2 (derived from fuel burn for cost/EF consistency) ---
        # Units: (t fuel) * (kg CO2 / kg fuel) / (t cargo) = kg CO2 / kg cargo.
        fuel_ef = MEOH_COMBUSTION_CO2_EF_KG_PER_KG if methanol else FUEL_OIL_CO2_EF_KG_PER_KG
        voyage_co2_kg_per_kg_meoh = (
            fleet_annual_fuel_t * fuel_ef / self.annual_meoh_tpa
        )

        total_cost_per_kg_meoh = ship_cost_per_kg_meoh + fuel_cost_per_kg_meoh

        return ShipResult(
            carrier_cargo_volume_m3=p.carrier_cargo_volume_m3,
            payload_per_voyage_t=payload_t,
            co2_payload_per_voyage_t=co2_payload_t,
            voyage_distance_nm_round=2.0 * p.shipping_distance_nm,
            sea_days_per_voyage=sea_days,
            loaded_days=loaded_days,
            total_days_per_voyage=total_days,
            voyages_per_ship_per_year=voyages_per_ship,
            annual_cargo_per_ship_tpa=annual_cargo_per_ship,
            annual_co2_per_ship_tpa=annual_co2_per_ship,
            annual_meoh_shipped_tpa=self.annual_meoh_tpa,
            annual_co2_shipped_tpa=self.annual_co2_tpa,
            ships_required_fractional=ships_required_frac,
            ships_required_integer=ships_required_int,
            ships_charged=ships_charged,
            annual_cost_per_ship_aud=annual_cost_per_ship_aud,
            bog_fraction_per_voyage=bog_frac,
            annual_bog_per_ship_t=annual_bog_per_ship_t,
            annual_sea_fuel_per_ship_t=annual_sea_fuel_per_ship_t,
            annual_port_fuel_per_ship_t=annual_port_fuel_per_ship_t,
            annual_fuel_per_ship_t=annual_fuel_per_ship_t,
            annual_meoh_bunker_per_ship_t=annual_fuel_per_ship_t if methanol else 0.0,
            annual_fossil_fuel_co2_tpa=0.0 if methanol else fleet_annual_fuel_t * fuel_ef,
            voyage_co2_kg_per_kg_meoh=voyage_co2_kg_per_kg_meoh,
            ship_capex_aud_per_vessel=p.ref_ship_capex_aud,
            ship_cost_usd_per_kg_meoh=ship_cost_per_kg_meoh,
            ship_fuel_cost_usd_per_kg_meoh=fuel_cost_per_kg_meoh,
            total_ship_cost_usd_per_kg_meoh=total_cost_per_kg_meoh,
        )


def main():
    from .supply_chain import SupplyChain

    r = SupplyChain().run()
    for port, ship in r.site_ship_results.items():
        print(
            f"{port}: {ship.ships_required_integer} carriers; {ship.annual_meoh_shipped_tpa:,.0f} t/year "
            f"methanol out, {ship.annual_co2_shipped_tpa:,.0f} t/year CO2 back"
        )


if __name__ == "__main__":
    main()
