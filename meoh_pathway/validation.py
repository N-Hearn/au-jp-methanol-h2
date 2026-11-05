"""Numerical acceptance checks, independent of literature calibration targets."""

from dataclasses import asdict
import math

import numpy as np


def close(actual, expected, label, rtol=1e-7, atol=1e-6):
    if not np.isfinite([actual, expected]).all():
        raise ValueError(f"{label}: non-finite value")
    if abs(actual-expected) > atol+rtol*max(abs(actual), abs(expected)):
        raise ValueError(f"{label}: {actual} != {expected}")


def _finite(value, path="result"):
    if isinstance(value, dict):
        for key, item in value.items():
            _finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            _finite(item, f"{path}[{i}]")
    elif isinstance(value, (float, np.floating)) and not math.isfinite(value):
        raise ValueError(f"{path}: non-finite result")


def validate_result(r):
    _finite(asdict(r))
    if not r.hourly_feasible:
        raise ValueError("A feasible hourly run is required")
    close(r.fleet_h2_delivered_tpa, r.target_h2_tpa, "Delivered target")
    close(r.cracker.heat_pump_heat_mwh,
          r.cracker.heat_pump_source_mwh + r.cracker.heat_pump_electricity_mwh,
          "Heat pump energy balance")
    close(r.cracker.gross_thermal_gj_per_year,
          r.cracker.thermal_gj_per_year + r.cracker.tail_gas_heat_credit_gj_per_year,
          "Cracker heat balance")
    close(r.cracker.heat_electricity_mwh,
          r.cracker.reformer_electricity_mwh+r.cracker.regeneration_electricity_mwh,
          "Reformer and regeneration electricity")
    close(r.fleet_co2_captured_tpa,
          r.cracker.captured_co2_tpd*r.cracker_actual_op_hrs/24*r.n_cracker,
          "Cracker captured CO2 aggregation")
    close(sum(r.site_h2_frac.values()), 1, "Site allocation")
    close(r.fleet_annual_meoh_tpa, r.required_cracker_meoh_tpa+r.fleet_heat_meoh_tpa
          +r.fleet_ship_fuel_meoh_tpa+r.fleet_meoh_loss_tpa, "Methanol balance")
    close(r.fleet_annual_synth_co2_feed_tpa,
          r.fleet_annual_co2_returned_tpa+r.fleet_annual_co2_makeup_tpa, "CO2 sourcing")
    liquefaction_co2 = sum(s.liquefaction_electricity_co2_tpa
                           for s in r.site_co2_capture_results.values())
    carbon_out = r.target_h2_tpa*(r.ship_fuel_ei+r.meoh_loss_ei+r.synth_process_ei
                  +r.cracker_vent_ei+r.cracker_heat_ei+r.co2_capture_ei)-liquefaction_co2
    close(r.fleet_annual_co2_makeup_tpa, carbon_out, "Carbon leaving the methanol/CO2 loop")
    close(sum(v for k,v in r.cost_breakdown_aud_per_kg_h2.items() if k != "Total"),
          r.delivered_cost_aud_per_kg_h2, "Cost attribution")
    close(r.delivered_ei_kg_co2_per_kg_h2,
          r.ship_fuel_ei+r.ship_fossil_fuel_ei+r.meoh_loss_ei+r.synth_process_ei+r.cracker_vent_ei
          +r.cracker_heat_ei+r.cracker_electricity_ei+r.co2_capture_ei
          +r.pem_electricity_ei+r.synth_electricity_ei, "Emissions attribution")
    close(r.gross_renewable_energy_mwh,
          r.fleet_total_electricity_kwh_per_year/1000+r.transmission_loss_mwh
          +r.wind_curtailment_mwh+r.battery_loss_mwh, "Annual electricity balance")
    for port, ship in r.site_ship_results.items():
        if ship.ships_required_fractional > ship.ships_charged*(1+1e-9):
            raise ValueError(f"{port}: insufficient carriers")
        if ship.annual_co2_shipped_tpa > ship.ships_charged*ship.annual_co2_per_ship_tpa*(1+1e-9):
            raise ValueError(f"{port}: carriers cannot return the captured CO2")
    for port, capture in r.site_co2_capture_results.items():
        close(r.site_ship_results[port].annual_co2_shipped_tpa,
              capture.annual_co2_captured_tpa, f"{port} CO2 shipped vs captured")


def validate_traces(r, traces):
    totals = dict(pem=0.0, transport=0.0, curtailment=0.0, battery_loss=0.0)
    if set(traces) != set(r.site_h2_frac):
        raise ValueError("Trace sites do not match the integrated deployment")
    for port, t in traces.items():
        for name, values in t.items():
            if not np.isfinite(values).all():
                raise ValueError(f"{port}.{name}: non-finite trace")
        caps = r.installed_design["capacities"]
        if np.any((t["pem_mw"] > 1e-8) & (t["pem_mw"] < t["min_pem_power_mw"]-1e-8)):
            raise ValueError(f"{port}: PEM runs below one module's minimum load")
        for name, values in (
            ("trunk_kg_per_hour", t["trunk_h2_kg"]),
            ("regional_kg_per_hour", np.abs(t["regional_h2_change_kg"])),
            ("local_injection_kg_per_hour", np.maximum(t["local_h2_change_kg"], 0)),
            ("regional_injection_kg_per_hour", np.maximum(t["regional_h2_change_kg"], 0)),
            ("local_injection_mw", t["local_compression_mw"]),
            ("regional_injection_mw", t["regional_injection_mw"]),
            ("pipeline_compressor_mw", t["pipeline_mw"]),
        ):
            cap = caps[f"{port}.{name}"]
            if np.max(values) > cap+max(1, cap)*1e-8:
                raise ValueError(f"{port}: {name} exceeds installed rating")
        close(t["soc_h2_kg"][-1], t["initial_soc_h2"], f"{port} periodic H2", atol=0.1)
        close(t["soc_e_mwh"][-1], t["initial_soc_e"], f"{port} periodic battery", atol=1e-5)
        balance = (t["net_generation_mw"]+t["battery_discharge_mw"]
                   -t["battery_charge_mw"]+t["unmet_electricity_mwh"]
                   -t["pem_mw"]-t["transport_mw"]-t["curtailed_mw"])
        if np.max(np.abs(balance)) > 1e-6:
            raise ValueError(f"{port}: hourly electricity balance does not close")
        h2 = t["initial_soc_h2"]+np.cumsum(t["h2_produced_kg"]-t["h2_delivered_kg"])
        if not np.allclose(h2,t["soc_h2_kg"],rtol=1e-9,atol=0.01):
            raise ValueError(f"{port}: hourly H2 inventory does not close")
        eta = math.sqrt(t["round_trip_efficiency"])
        battery = t["initial_soc_e"]+np.cumsum(t["battery_charge_mw"]*eta-t["battery_discharge_mw"]/eta)
        if not np.allclose(battery,t["soc_e_mwh"],rtol=1e-9,atol=1e-6):
            raise ValueError(f"{port}: hourly battery inventory does not close")
        for values, cap, label in ((t["soc_h2_kg"],r.site_h2_storage_kg[port],"H2"),
                                   (t["soc_e_mwh"],r.site_battery_mwh[port],"battery")):
            if min(values) < -1e-6 or max(values)>cap+max(1,cap)*1e-8:
                raise ValueError(f"{port}: {label} inventory outside capacity")
        close(t["h2_produced_kg"].sum(),t["demand_h2_kg"].sum(),f"{port} annual H2",atol=0.1)
        totals["pem"] += t["pem_mw"].sum()
        totals["transport"] += (t["transport_mw"].sum()
            -t["regional_compression_mw"].sum()*r.site_transmission_loss_fraction[port])
        totals["curtailment"] += t["curtailed_mw"].sum()
        totals["battery_loss"] += t["battery_loss_mwh"].sum()
    close(totals["pem"],r.fleet_pem_electricity_kwh_per_year/1000,"PEM trace aggregation")
    close(totals["transport"],r.fleet_transport_electricity_kwh_per_year/1000,"Compression trace aggregation")
    close(totals["curtailment"] + sum(r.idle_site_renewable_mwh.values()),
          r.wind_curtailment_mwh,"Curtailment trace aggregation")
    close(totals["battery_loss"],r.battery_loss_mwh,"Battery loss trace aggregation")
