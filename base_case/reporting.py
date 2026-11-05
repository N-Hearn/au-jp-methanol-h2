"""Text summary for the grey LNG/ATR reference case."""


def print_base_summary(p, lng_params):
    fleet_h2_kt = p.target_h2_tpa / 1000.0
    fleet_h2_kg = p.target_h2_tpa * 1000.0
    ng_mass_kt = (
        fleet_h2_kg
        * (p.feed_gj_per_kg_h2 * 1000.0 / lng_params.lng_lhv_mj_per_kg)
        / 1e6
    )
    plant_scale = p.target_h2_tpa / (p.fleet_plants * p.annual_h2_per_plant_tpa)
    capacity_tpd = p.plant_capacity_tpd_h2 * plant_scale
    co2_vent_kt = p.plant_gate_ei_kg_co2_per_kg_h2 * fleet_h2_kg / 1e6
    co2_up_kt = p.upstream_ei_kg_co2_per_kg_h2 * fleet_h2_kg / 1e6

    print("\n--- Reference case: Grey H2 (LNG) ---")
    print("Scale")
    print(f"  H2 target            {fleet_h2_kt:>10,.0f} kt/yr")
    print(f"  ATR plants           {p.fleet_plants:>10d}  x {capacity_tpd:,.0f} t H2/d")
    print("Flows (fleet)")
    print(f"  LNG / NG feed        {ng_mass_kt:>10,.0f} kt/yr")
    print(f"  H2 delivered         {fleet_h2_kt:>10,.0f} kt/yr")
    print(f"  CO2 upstream vent    {co2_up_kt:>10,.0f} kt/yr")
    print(f"  CO2 plant-gate vent  {co2_vent_kt:>10,.0f} kt/yr")
    print("Delivered")
    print(
        f"  Cost                 {p.delivered_cost_aud_per_kg_h2:>10.3f} AUD/kg H2"
        f"   ({p.delivered_cost_usd_per_kg_h2:.3f} USD/kg H2)"
    )
    print(
        f"  EI                   {p.delivered_ei_kg_co2_per_kg_h2:>10.3f} kg CO2/kg H2"
    )
