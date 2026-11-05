"""Text summaries for the MeOH pathway: delivered result and sensitivity tornadoes.

Reporting stays out of scripts/ so the scripts can import it as a
library instead of importing a runner script for its side functions.
"""

from typing import List


def print_meoh_summary(r):
    print(r.model_basis)
    if not r.hourly_feasible:
        print(f"SHORTFALL: {r.unmet_h2_kg:,.0f} kg H2 and {r.unmet_demand_mwh:,.3f} MWh electricity unmet")
        for port, short in r.site_shortfalls.items():
            print(f"  {port}: {short['unmet_h2_kg']:,.0f} kg H2; {short['unmet_electricity_mwh']:,.3f} MWh electricity unmet")
        for reason in r.equipment_shortfalls:
            print(f"  {reason}")
        print("No feasible delivered-cost result is reported.")
        return
    print(f"Delivered H2: {r.fleet_h2_delivered_tpa:,.0f} t/year")
    print(
        f"Cost: {r.delivered_cost_aud_per_kg_h2:.3f} AUD/kg H2 ({r.delivered_cost_usd_per_kg_h2:.3f} USD/kg H2)"
    )
    print(
        f"Emissions intensity: {r.delivered_ei_kg_co2_per_kg_h2:.3f} kg CO2e/kg H2; not a full lifecycle assessment"
    )
    print(
        f"Fleets: {r.n_pem * r.pem.small_module_mw:,.0f} MW PEM, {r.n_synth} synthesis trains, {r.n_cracker} crackers"
    )
    print(f"Available hours: synthesis {r.synth_available_hours_per_year:,.0f}; "
          f"cracker {r.cracker_available_hours_per_year:,.0f} h/year")
    print(f"Actual fleet-average full-load hours: synthesis {r.synth_actual_op_hrs:,.0f}; "
          f"cracker {r.cracker_actual_op_hrs:,.0f} h/year")
    print(
        f"Cracker per unit: CAPEX {r.cracker.capex_aud / 1e6:.2f} million AUD; annualised {r.cracker.annualised_capex_aud / 1e6:.2f} million AUD/year"
    )
    print(f"Cracker own cost: {r.cracker_component_aud_per_kg_h2:.3f} AUD/kg H2")
    print(f"Cracker gross/net nameplate: {r.cracker.gross_capacity_tpd_h2:.1f}/"
          f"{r.cracker.plant_capacity_tpd_h2:.1f} t H2/day; capture configuration {r.cracker.co2_capture_config}")
    kg = r.cracker.annual_h2_tpa*1000
    print(f"Japanese electricity: reformer {r.cracker.reformer_electricity_mwh*1000/kg:.3f}, "
          f"regeneration {r.cracker.regeneration_electricity_mwh*1000/kg:.3f}, "
          f"capture {r.cracker.capture_electricity_mwh*1000/kg:.3f} kWh/kg H2; "
          f"price {r.cracker.power_price_aud_per_kwh:.4f} AUD/kWh")
    print(f"Heat limits: tail gas capped={r.cracker.tail_gas_heat_capped}; "
          f"heat pump source limited={r.cracker.heat_pump_source_limited}")
    if r.cracker.co2_capture_config:
        print("Capture cost caveat: flue-gas IEA allowance includes energy, overlapping explicit electricity.")
    print(
        f"Cracker internal heat production: {r.cracker_heat_production_aud_per_kg_h2:.3f} AUD/kg H2 (already in LCOMeOH)"
    )
    print(
        f"LCOMeOH: {r.lcomeoh_aud_per_kg_meoh:.3f} AUD/kg MeOH; {r.lcomeoh_component_aud_per_kg_h2:.3f} AUD/kg delivered H2 including fuel production"
    )
    print(
        f"Methanol: {r.fleet_annual_meoh_tpa / 1e6:.3f} Mt/year produced; {r.required_cracker_meoh_tpa / 1e6:.3f} cracker feed; {r.fleet_heat_meoh_tpa / 1e6:.3f} heat fuel; {r.fleet_ship_fuel_meoh_tpa / 1e6:.3f} bunkers; {r.fleet_meoh_loss_tpa / 1e6:.3f} losses"
    )
    if r.fleet_h2_fuel_tpa > 0:
        print(f"Cracker H2 fired for reformer heat: {r.fleet_h2_fuel_tpa / 1e3:,.1f} kt/year")
    print(
        f"CO2 circularity (returned / fresh synthesis feed): {r.circularity_rate:.1%}"
    )
    print(
        f"PEM SEC: {r.pem.sec_kwh_per_kg_h2:.3f} kWh/kg; compression electricity separate"
    )
    print(
        f"H2 transport: {r.h2_transport_component_aud_per_kg_h2:.3f} AUD/kg H2 capital/O&M, plus {r.fleet_transport_electricity_kwh_per_year / 1e9:.3f} TWh/year in renewables"
    )
    print("\nNon-overlapping cost contributions, AUD/kg delivered H2:")
    for key, value in r.cost_breakdown_aud_per_kg_h2.items():
        print(f"  {key:20} {value:9.4f}")
    print("\nRoutes, one way:")
    for port, values in r.site_routes.items():
        line = f"  {port:18} {values['route_distance_nm']:5.0f} nm; {values['renewables_to_port_km']:4.0f} km H2 trunk"
        local = r.site_local_h2_storage_kg[port] / 1e6
        regional = r.site_regional_h2_storage_kg[port] / 1e6
        line += (
            f"; {local:,.1f} kt {r.site_local_h2_storage_type[port]} at PEM site, "
            f"{regional:,.1f} kt geological {values['pipeline_distance_km']:.0f} km from port"
        )
        print(f"{line}; {values['route_distance_nm_note']}")


def print_tornado_table(points: List, metric: str = "cost", threshold: float = 0) -> None:
    """Print a tornado table sorted by swing, largest first.

    threshold drops points whose swing does not exceed it (default 0: only a
    non-positive swing is dropped, since a genuine zero-effect parameter adds
    nothing to a tornado listing).
    """
    sort_key = (lambda p: p.cost_swing) if metric == "cost" else (lambda p: p.ei_swing)
    pts = sorted(
        [p for p in points if sort_key(p) > threshold], key=sort_key, reverse=True
    )
    if not pts:
        return
    base = pts[0].base_cost if metric == "cost" else pts[0].base_ei
    unit = "AUD/kg H2" if metric == "cost" else "kg CO2/kg H2"
    getter = (
        (lambda p: (p.low_cost, p.high_cost, p.cost_swing))
        if metric == "cost"
        else (lambda p: (p.low_ei, p.high_ei, p.ei_swing))
    )

    print(f"\nTornado ({metric}, base = {base:.4f} {unit})  -- zero-swing omitted")
    print("-" * 78)
    print(f"{'Parameter':<42} {'Low':>10} {'High':>10} {'Swing':>10}")
    print("-" * 78)
    for p in pts:
        lo, hi, sw = getter(p)
        print(f"{p.bound.name:<42} {lo:>10.4f} {hi:>10.4f} {sw:>10.4f}")
