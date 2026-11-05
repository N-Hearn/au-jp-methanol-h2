"""Explicit workbook probability models, separate from sensitivity ranges.

Each module Inputs table carries Role, Distribution, Dependency and Uncertainty
basis columns beside its Typical/Min/Max. A row is sampled only when that
metadata says so; sensitivity bounds alone never opt it in.
"""

from dataclasses import dataclass
from functools import lru_cache

from inputs.loader import (
    MEOH_SHEETS, SHEET_TO_PARAMSET_ATTR, XLSX_PATH, _split_field, load_input_rows,
    numeric_bounds,
)
from .sensitivity import Bound, build_bounds, load_bounds_from_excel

MODES = ("economic", "technical")
ROLES = ("economic", "technical", "design", "scenario")
GREY_SHEETS = ("lng_supply", "atr_plant")
CCS_SHEETS = ("atr_ccs",)
LNG_CASES = ("grey", "grey_ccs")
# Methanol-loop sheets whose rows the LNG cases share, so both pathways draw the
# same Japanese power, capture, liquefaction, carrier and discount inputs. Grey
# reads only the power basis (ATR electricity); blue reads every shared row.
SHARED_SHEETS = ("meoh_cracking", "co2_capture", "shipping", "economics")
GREY_SHARED_FIELDS = {
    ("cracker", "cracker_power_source"), ("cracker", "cracker_power_price_aud_per_kwh"),
    ("cracker", "zero_carbon_power_price_aud_per_kwh"), ("cracker", "cracker_grid_ei_kg_co2_per_kwh"),
}
# Per-site route lengths Monte Carlo may draw. Each is a straight-line distance
# times one route allowance, so the workbook ties them with a shared dependency;
# the other route rows (sailing distance, offshore share) stay put.
ROUTE_FIELDS = ("renewables_to_port_km", "pipeline_distance_km")
INSTALLED_FIELDS = {
    ("pem", "small_module_mw"), ("pem", "large_module_mw"),
    ("pem", "transmission_circuit_capacity_mw"), ("pem", "battery_storage_duration_hours"),
    ("pem", "capex_reference_mw"), ("pem", "min_site_renewables_mw"),
    ("synth", "capacity_tpy"), ("synth", "ref_capacity_tpy"),
    ("cracker", "capacity_tpd_h2"), ("cracker", "ref_capacity_tpd_h2"),
    ("ship", "carrier_cargo_volume_m3"),
    ("ship", "terminal_storage_days"),
    ("transport", "pipeline_capacity_kg_per_hour"),
}
# Usage rates that only feed a bill: no mass or energy balance reads them, so
# the frozen hourly schedule in economic mode stays valid when they move.
COST_ONLY_FIELDS = {
    ("pem", "water_l_per_kg_h2"),
    ("synth", "catalyst_inventory_g_per_tpy"),
    ("synth", "catalyst_life_years"),
    ("cracker", "heat_pump_installation_factor"),
}


def economic_field(group, field):
    if group == "site_storage_costs" or (group, field) in COST_ONLY_FIELDS:
        return True
    if group == "site_routes":
        return False
    return (field == "discount_rate" or "capex" in field or "opex" in field
            or "cost_aud" in field or "price_aud" in field or "_aud_per_" in field
            or field in {"stack_replacement_percent", "system_life_years",
                         "plant_life_years", "ship_lifetime_years", "cracker_life_years"}
            or field.endswith("_life_years"))


def _input_name(sheet, workbook_field):
    """The group.field a workbook row is drawn under, or None if no model input."""
    site, field = _split_field(workbook_field)
    if site is None:
        return f"{SHEET_TO_PARAMSET_ATTR[sheet]}.{field}"
    if field == "h2_storage_capex_aud_per_kg":
        return f"site_storage_costs.{site}"
    if field in ROUTE_FIELDS:
        return f"site_routes.{workbook_field}"
    return None


@lru_cache(maxsize=None)
def workbook_uncertainty(path=XLSX_PATH):
    """{group.field: (sheet, row)} for every input row with uncertainty metadata.

    Rows with neither Role nor Distribution (CPI indices, sailing routes) are
    reference values and never sampled.
    """
    out = {}
    for sheet, row in load_input_rows(path, MEOH_SHEETS + GREY_SHEETS + CCS_SHEETS):
        role, distribution = row.get("Role"), row.get("Distribution")
        if role is None and distribution is None:
            continue
        name = _input_name(sheet, row["Field"])
        label = name or f"{sheet}.{row['Field']}"
        if role not in ROLES:
            raise ValueError(f"{label}: invalid uncertainty role {role!r}")
        if distribution not in ("fixed", "triangular", "uniform"):
            raise ValueError(f"{label}: unsupported distribution {distribution!r}")
        if role in ("design", "scenario") and distribution != "fixed":
            raise ValueError(f"{label}: design and scenario inputs must remain fixed")
        if name is None:
            if distribution != "fixed":
                raise ValueError(f"{label}: sampled row has no model input to draw")
            continue
        if name in out:
            raise ValueError(f"Duplicate uncertainty input {name}")
        out[name] = (sheet, row)
    return out


def role_fields(role, path=XLSX_PATH):
    """(group, field) pairs the workbook gives this Role, site rows excluded."""
    return {tuple(name.split(".", 1)) for name, (_, row) in workbook_uncertainty(path).items()
            if row["Role"] == role and not name.startswith(("site_storage_costs.", "site_routes."))}


def route_bounds(path=XLSX_PATH):
    """Bounded per-site route lengths, as draws on ParamSet.site_routes."""
    bounds = []
    for _, row in load_input_rows(path, "h2_transport"):
        site, field = _split_field(row["Field"])
        if site is None or field not in ROUTE_FIELDS or row.get("Lower bound") is None:
            continue
        base, low, high = numeric_bounds(row, row["Field"])
        if high > low:
            bounds.append(Bound(f"{row.get('Variable') or row['Field']} [km]",
                                "site_routes", row["Field"], base, low, high))
    return bounds


@dataclass(frozen=True)
class UncertainInput(Bound):
    distribution: str
    evidence: str
    dependency: str = ""


def uncertainty_bounds(case, mode, path=XLSX_PATH):
    if mode not in MODES:
        raise ValueError(f"Unknown uncertainty mode {mode!r}; choose from {MODES}")
    roles = (mode,)
    if case in LNG_CASES:
        from .monte_carlo import GreyParams
        candidates = modelled = load_bounds_from_excel(
            path, GREY_SHEETS + CCS_SHEETS + SHARED_SHEETS, GreyParams.base())
    else:
        candidates = modelled = build_bounds() + route_bounds(path)
    by_name = {f"{b.group}.{b.field}": b for b in candidates}
    known = {f"{b.group}.{b.field}" for b in modelled}
    selected = []
    for name, (sheet, row) in workbook_uncertainty(path).items():
        if row["Role"] not in roles or row["Distribution"] == "fixed":
            continue
        if case in LNG_CASES and sheet in SHARED_SHEETS:
            if name not in by_name:
                continue  # a methanol-loop row the LNG cases do not read
            if case == "grey" and tuple(name.split(".", 1)) not in GREY_SHARED_FIELDS:
                continue
        elif (sheet in GREY_SHEETS + CCS_SHEETS) != (case in LNG_CASES):
            continue
        if sheet in CCS_SHEETS and case != "grey_ccs":
            continue
        if name not in known:
            raise ValueError(f"{name}: sampled row is not a bounded model input")
        if name not in by_name:
            continue  # inactive in this pathway
        bound = by_name[name]
        group, field = name.split(".", 1)
        if (group, field) in INSTALLED_FIELDS:
            raise ValueError(f"{name}: installed equipment assumptions must remain fixed")
        if row["Role"] == "economic" and not economic_field(group, field):
            raise ValueError(f"{name}: physical input cannot be sampled as economic")
        if row["Role"] == "technical" and economic_field(group, field):
            raise ValueError(f"{name}: economic input cannot be sampled as technical")
        evidence = str(row.get("Uncertainty basis") or "").strip()
        if not evidence:
            raise ValueError(f"{name}: sampled distribution needs an uncertainty basis")
        dependency = str(row.get("Dependency") or "").strip()
        if dependency and not dependency.startswith("shared:"):
            raise ValueError(f"{name}: unsupported dependency {dependency!r}")
        selected.append(UncertainInput(bound.name, group, field, bound.base,
                                       bound.low, bound.high, row["Distribution"], evidence,
                                       dependency))
    return selected
