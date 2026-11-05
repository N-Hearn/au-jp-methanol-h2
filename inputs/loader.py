"""Load parameter values from input_data.xlsx.

Reads the Inputs table on each module sheet by column header. References
below the table are not model inputs. Returns {field: value} for
constructing parameter dataclasses.

The Excel is the single source of truth for parameter values. The sheet
name identifies the module; the Field column identifies the row. Site
rows use a compound Field key of the form '<Site>.<field>' and are
split by `_split_field`.
"""

import dataclasses
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple, Type, TypeVar

import openpyxl
from openpyxl.utils.cell import range_boundaries


XLSX_PATH = Path(__file__).resolve().parent / "input_data.xlsx"

T = TypeVar("T")

MEOH_SHEETS = (
    "green_h2_pem",
    "h2_storage",
    "renewable_supply",
    "renewable_storage",
    "meoh_synthesis",
    "shipping",
    "meoh_cracking",
    "co2_capture",
    "co2_balancing",
    "economics",
    "h2_transport",
)
SITE_SHEETS = ("renewable_supply", "renewable_storage", "h2_storage")
SITE_INPUT_SHEETS = (
    "renewable_supply",
    "renewable_storage",
    "h2_storage",
)

# Which ParamSet attribute each sheet's non-site rows belong to. Sensitivity
# and optimisation use this to route a row's Field onto the right dataclass
# without carrying a separate Group column in Excel.
SHEET_TO_PARAMSET_ATTR = {
    "green_h2_pem": "pem",
    "h2_storage": "pem",
    "renewable_supply": "pem",
    "renewable_storage": "pem",
    "meoh_synthesis": "synth",
    "shipping": "ship",
    "meoh_cracking": "cracker",
    "co2_capture": "co2_cap",
    "co2_balancing": "dac",
    "economics": "econ",
    "h2_transport": "transport",
    "lng_supply": "lng",
    "atr_plant": "atr",
    "atr_ccs": "ccs",
}


def _sheet_names(sheets) -> Tuple[str, ...]:
    if isinstance(sheets, str):
        return (sheets,)
    return tuple(sheets)


def _split_field(field):
    """Return (site, base_field) for compound Field like 'PortHastings.wind_mw'."""
    if not field or "." not in str(field):
        return None, field
    site, _, base = str(field).partition(".")
    return site, base


def _table_rows(ws):
    tables = [t for t in ws.tables.values() if t.name.startswith("Inputs_")]
    if not tables:
        raise ValueError(f"Sheet '{ws.title}' has no Inputs_* table")
    if len(tables) > 1:
        raise ValueError(f"Multiple Inputs tables on sheet '{ws.title}'")
    lo_col, lo_row, hi_col, hi_row = range_boundaries(tables[0].ref)
    rows = ws.iter_rows(
        min_row=lo_row, max_row=hi_row, min_col=lo_col, max_col=hi_col, values_only=True
    )
    aliases = {"Typical": "Value", "Min": "Lower bound", "Max": "Upper bound"}
    headers = tuple(aliases.get(h, h) for h in next(rows, ()))
    named = [h for h in headers if h is not None]
    if len(set(named)) != len(named) or not {"Field", "Value"} <= set(named):
        raise ValueError(f"{ws.title}: duplicate or missing input headers")
    for row in rows:
        if any(value is not None for value in row):
            yield {
                header: value
                for header, value in zip(headers, row)
                if header is not None
            }


def _cost_basis_converter(wb):
    """Convert rows entered in a source currency and year to real base-year AUD.

    Escalate with the source currency's CPI, then convert USD at the economics
    AUD/USD rate, so historical exchange rates never enter the model.
    """
    econ = {row.get("Field"): row.get("Value") for row in _table_rows(wb["economics"])}
    base = int(float(econ["cost_base_year"]))
    fx = float(econ["aud_usd_exchange_rate"])
    if not math.isfinite(fx) or fx <= 0:
        raise ValueError("Reporting exchange rate must be positive and finite")

    def convert(row, label):
        parts = str(row["Cost basis"]).split()
        currency = parts[0].upper() if parts else ""
        if len(parts) != 2 or currency not in ("AUD", "USD") or not parts[1].isdigit():
            raise ValueError(f"{label}: Cost basis must be 'AUD <year>' or 'USD <year>'")
        index = currency.lower() + "_cpi_"
        try:
            current, original = float(econ[f"{index}{base}"]), float(econ[f"{index}{parts[1]}"])
            if not all(math.isfinite(v) and v > 0 for v in (current, original)):
                raise ValueError(f"{label}: price indices must be positive and finite")
            factor = current / original
        except KeyError as exc:
            raise ValueError(f"{label}: no {exc.args[0]} row on economics") from exc
        if currency == "USD":
            factor /= fx
        for key in ("Value", "Lower bound", "Upper bound"):
            try:
                row[key] = float(row[key]) * factor
            except (TypeError, ValueError):
                pass

    return convert


@lru_cache(maxsize=None)
def load_input_rows(
    path: Path = XLSX_PATH, sheets=MEOH_SHEETS
) -> Tuple[Tuple[str, dict], ...]:
    """Read module input tables. Returns ((sheet, row), ...) so callers
    can route each row by its owning sheet without needing a Group column."""
    wb = openpyxl.load_workbook(path, data_only=True)
    try:
        to_base = _cost_basis_converter(wb)
        result = []
        for name in _sheet_names(sheets):
            seen = set()
            for row in _table_rows(wb[name]):
                if "Value" not in row or "Field" not in row:
                    raise KeyError(
                        f"'{name}' Inputs table needs 'Value' and 'Field' columns"
                    )
                if row.get("Cost basis"):
                    to_base(row, f"{name}.{row['Field']}")
                    # Source-currency keys become AUD keys only after conversion.
                    row["Field"] = row["Field"].replace("_usd", "_aud")
                field = row["Field"]
                if not field or field in seen:
                    raise ValueError(f"{name}: missing or duplicate input Field {field!r}")
                seen.add(field)
                bounds = (row.get("Lower bound"), row.get("Upper bound"))
                if any(v not in (None, "", "-") for v in bounds):
                    site, base_field = _split_field(field)
                    if name in SITE_SHEETS and site is not None and base_field in DEPLOYMENT_CAPACITY_FIELDS:
                        capacity_bounds(row, f"{name}.{field}")
                    else:
                        numeric_bounds(row, f"{name}.{field}")
                result.append((name, row))
        return tuple(result)
    finally:
        wb.close()


def from_dict(cls: Type[T], v: Dict[str, float], *, ignored=(), **extras) -> T:
    """Build dataclass `cls` by matching field names to keys in `v`.

    Fields typed as `int` are cast to int. `extras` supplies values for
    fields not in `v` (runtime selectors, non-Excel defaults). Unknown fields fail unless explicitly listed in ignored. Fields
    missing from both fall back to the dataclass default.
    """
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(v) - names - set(ignored)
    if unknown:
        raise ValueError(f"Unknown inputs for {cls.__name__}: {sorted(unknown)}")
    kwargs = dict(extras)
    for f in dataclasses.fields(cls):
        if f.name in kwargs or f.name not in v:
            continue
        if f.type is int and not float(v[f.name]).is_integer():
            raise ValueError(f"{cls.__name__}.{f.name}: integer input required")
        kwargs[f.name] = int(v[f.name]) if f.type is int else v[f.name]
    return cls(**kwargs)


@lru_cache(maxsize=None)
def inputs_digest(path: Path = XLSX_PATH) -> str:
    """Hash the workbook values the model reads, after cost-basis conversion.

    Sources, notes and formatting are left out, so re-saving the workbook or
    editing a citation does not make saved results stale.
    """
    keys = ("Field", "Value", "Lower bound", "Upper bound", "Cost basis",
            "Role", "Distribution", "Dependency")
    rows = load_input_rows(path, ("lng_supply", "atr_plant") + MEOH_SHEETS)
    table = [[sheet] + [row.get(key) for key in keys] for sheet, row in rows]
    return hashlib.sha256(json.dumps(table, default=str).encode()).hexdigest()


def load_params(
    path: Path = XLSX_PATH,
    sheet=MEOH_SHEETS,
) -> Dict[str, float]:
    """Return {field: value} for all non-site rows across the named sheet(s).

    Accepts one module sheet or a tuple of sheets. Column resolution is by
    header name, so input-table columns can be reordered. Values are cast
    to float; malformed numeric fields and duplicates raise errors. Rows
    without a Field are reference-only and skipped. Site rows (Field of the form '<Site>.<f>') are excluded.
    """
    out: Dict[str, float] = {}
    for _sheet, row in load_input_rows(path, sheet):
        site, field = _split_field(row.get("Field"))
        if site is not None or not field:
            continue
        if field in out:
            raise ValueError(f"Duplicate input: {field}")
        try:
            value = float(row["Value"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{_sheet}.{field}: numeric input required") from exc
        if not math.isfinite(value):
            raise ValueError(f"{_sheet}.{field}: finite input required")
        out[field] = value
    return out


@lru_cache(maxsize=None)
def _param_row_index(path: Path, sheet: str) -> Dict[str, dict]:
    """One-pass lookup of {field: row} for the non-site rows of a sheet."""
    index = {}
    for _sheet, row in load_input_rows(path, sheet):
        site, field = _split_field(row.get("Field"))
        if site is None and field:
            index[field] = row
    return index


def parameter_input(path: Path, sheet: str, field: str) -> dict:
    """Get one non-site input row (including its bounds) by Field name."""
    row = _param_row_index(path, sheet).get(field)
    if row is None:
        raise ValueError(f"{sheet}.{field}: no matching row in Inputs table")
    return row


def numeric_bounds(row: dict, label: str) -> tuple:
    """Require explicit finite bounds; zero is a valid minimum and maximum."""
    try:
        typical, lo, hi = (
            float(row[k]) for k in ("Value", "Lower bound", "Upper bound")
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{label}: Typical, Min and Max must be numeric") from exc
    if not all(math.isfinite(v) for v in (typical, lo, hi)) or not lo <= typical <= hi:
        raise ValueError(f"{label}: require finite Min <= Typical <= Max")
    return typical, lo, hi


# Site rows calculated by the Typical build (meoh_pathway.typical) rather than
# read as a workbook default; their Typical cell is blank.
DEPLOYMENT_CAPACITY_FIELDS = {
    "wind_mw", "solar_mw", "battery_mwh", "h2_storage_kg", "local_h2_storage_kg",
    "pem_modules", "synth_modules",
}


def capacity_bounds(row: dict, label: str) -> tuple:
    """Require finite Min <= Max; Typical is optional (None when blank)."""
    try:
        lo, hi = (float(row[k]) for k in ("Lower bound", "Upper bound"))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{label}: Min and Max must be numeric") from exc
    if not all(math.isfinite(v) for v in (lo, hi)) or not lo <= hi:
        raise ValueError(f"{label}: require finite Min <= Max")
    typical = row.get("Value")
    if typical in (None, "", "-"):
        return None, lo, hi
    try:
        typical = float(typical)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: Typical must be numeric or blank") from exc
    if not math.isfinite(typical) or not lo <= typical <= hi:
        raise ValueError(f"{label}: require finite Min <= Typical <= Max")
    return typical, lo, hi


@lru_cache(maxsize=None)
def load_site_inputs(path: Path = XLSX_PATH) -> dict:
    """Join site rows from their owning modules by the compound Field key."""
    required = {
        "renewable_supply": ("wind_mw", "solar_mw", "renewables_capex_mult",
                             "pem_modules", "synth_modules"),
        "renewable_storage": ("battery_mwh", "battery_capex_mult"),
        "h2_storage": (
            "h2_storage_kg",
            "h2_storage_capex_aud_per_kg",
            "local_h2_storage_kg",
            "local_h2_storage_type",
        ),
    }
    out: Dict[str, Dict[str, dict]] = {}
    site_sets = []
    for sheet in SITE_INPUT_SHEETS:
        sites = set()
        for _sheet, row in load_input_rows(path, sheet):
            site, field = _split_field(row.get("Field"))
            if site is None:
                continue
            site = site.strip()
            if not site or not field:
                raise ValueError(f"Invalid site Field '{row.get('Field')}' in {sheet}")
            if field not in required[sheet]:
                raise ValueError(f"Unknown site field {site}.{field} in {sheet}")
            fields = out.setdefault(site, {})
            if field in fields:
                raise ValueError(f"Duplicate site input {site}.{field}")
            fields[field] = row
            sites.add(site)
        site_sets.append(sites)
    if not site_sets[0] or any(s != site_sets[0] for s in site_sets[1:]):
        raise ValueError(
            "renewable_supply, renewable_storage and h2_storage must contain the same sites"
        )
    for site, fields in out.items():
        for sheet, required_fields in required.items():
            missing = set(required_fields) - {
                field for field in fields if field in required[sheet]
            }
            if missing:
                raise ValueError(
                    f"Missing site inputs for {site} in {sheet}: {sorted(missing)}"
                )
    for site, fields in out.items():
        for field, row in fields.items():
            if field in (
                "wind_mw",
                "solar_mw",
                "battery_mwh",
                "h2_storage_kg",
                "local_h2_storage_kg",
                "pem_modules",
                "h2_allocation_weight",
                "synth_modules",
            ):
                _, lo, _ = capacity_bounds(row, f"{site}.{field}")
                if lo < 0:
                    raise ValueError(f"{site}.{field}: capacity cannot be negative")
                if field in ("synth_modules", "pem_modules") and any(
                    v is not None and v != int(v) for v in capacity_bounds(row, f"{site}.{field}")
                ):
                    raise ValueError(f"{site}.{field}: train counts must be integers")
            elif field != "local_h2_storage_type":
                value = float(row["Value"])
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(
                        f"{site}.{field}: cost factor must be positive and finite"
                    )
    return out
