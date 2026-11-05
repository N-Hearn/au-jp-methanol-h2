"""Physical search ceilings for process fleets, independent of Excel fleet rows."""

import math
from constants import HOURS_PER_YEAR

from inputs.loader import load_site_inputs
from .parameters import XLSX_PATH, load_pem, load_synth, load_routes
from .renewables_dispatch import load_wind_profile, load_solar_profile


def fleet_bounds(path=XLSX_PATH, deployment=None, pem=None, synth=None):
    p, s = pem or load_pem(path), synth or load_synth(path)
    rows, routes = load_site_inputs(path), load_routes(path)
    limits = {}
    for port, values in rows.items():
        def capacity(field):
            if deployment is not None:
                return deployment.get(f"site_{field}", {}).get(port, 0.0)
            return float(values[field]["Upper bound"])
        peak = float((capacity("wind_mw") * load_wind_profile(port)
                      + capacity("solar_mw") * load_solar_profile(port)).max())
        peak += capacity("battery_mwh") / p.battery_storage_duration_hours
        loss = routes[port]["renewables_to_port_km"] * p.transmission_losses_percent / 100
        if not 0 <= loss < 1:
            raise ValueError(f"{port}: invalid transmission losses")
        pem_max = math.ceil(peak / p.small_module_mw)
        # Local H2 can supply synthesis independently of this hour's PEM.
        # Minimum stable train load and staggered outages bound useful trains.
        power_per_train = (s.capacity_tpy / HOURS_PER_YEAR * (s.sec_electricity_kwh_per_t
                           + s.sec_heat_kwh_per_t / s.heat_efficiency)
                           / 1000 * s.min_load_percent / 100)
        if power_per_train <= 0 or s.available_hours_per_year <= 0:
            raise ValueError("Synthesis fleet ceiling requires positive minimum load and availability")
        synth_max = math.ceil(peak * (1-loss) / power_per_train
                              * HOURS_PER_YEAR / s.available_hours_per_year) if peak else 0
        limits[port] = {"pem_modules": (0, pem_max),
                        "synth_modules": (0, synth_max),
                        "h2_allocation_weight": (0.0, 1.0 if peak else 0.0)}
    return limits


def renewable_shares(deployment):
    energies = {port: deployment["site_wind_mw"].get(port, 0) * load_wind_profile(port).sum()
                + deployment["site_solar_mw"].get(port, 0) * load_solar_profile(port).sum()
                for port in load_site_inputs()}
    total = sum(energies.values())
    if total <= 0:
        raise ValueError("Typical deployment has no renewable energy")
    return {port: float(value / total) for port, value in energies.items()}
