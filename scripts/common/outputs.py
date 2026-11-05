"""Output locations and reproducibility metadata for the model cases."""

import dataclasses
import json
import typing
from dataclasses import asdict
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs"
TYPICAL_DIR = OUTPUT_DIR / "typical_build"
OPT_DIR = OUTPUT_DIR / "optimised_build"
# Delivery-margin builds (builds.optimisation --delivery-margin); the 10% build
# is the proposed build.
MARGIN_DIR = OPT_DIR / "margin_cases"
PROPOSED_DIR = MARGIN_DIR / "margin_10pct"
MC_DIR = OUTPUT_DIR / "monte_carlo"
CASE_DIRS = {"typical": TYPICAL_DIR, "optimised": OPT_DIR}


def design_dir(build_dir):
    """Design alternatives of one build: co-location, heat and capture configurations."""
    return Path(build_dir) / "design_analysis"


def margin_mc_dir(pct):
    """Technical Monte Carlo of a delivery-margin build."""
    return MC_DIR / f"margin_{pct:g}pct"


def write_result(result, outdir, case):
    from meoh_pathway.optimisation import model_fingerprint

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "results.json").write_text(json.dumps(asdict(result), indent=2, allow_nan=False))
    (outdir / "run_metadata.json").write_text(json.dumps(dict(
        case=case, model_fingerprint=model_fingerprint(), model_basis=result.model_basis,
        hourly_feasible=result.hourly_feasible, annual_balanced=result.annual_balanced,
    ), indent=2))


def _dataclass_from_dict(cls, data):
    """Inverse of dataclasses.asdict(): rebuild nested dataclass fields recursively.

    asdict() flattens nested dataclasses to plain dicts, so cls(**data) alone
    leaves those fields as dicts instead of instances. Used to replay a saved
    results.json without resolving the model again.
    """
    hints = typing.get_type_hints(cls)
    kwargs = {}
    for f in dataclasses.fields(cls):
        value = data.get(f.name)
        ftype = hints.get(f.name, f.type)
        if dataclasses.is_dataclass(ftype) and isinstance(value, dict):
            value = _dataclass_from_dict(ftype, value)
        kwargs[f.name] = value
    return cls(**kwargs)


def read_result(outdir, cls=None):
    """Reconstruct a saved SupplyChainResult from results.json, for --replot."""
    if cls is None:
        from meoh_pathway.supply_chain import SupplyChainResult
        cls = SupplyChainResult
    data = json.loads((Path(outdir) / "results.json").read_text())
    return _dataclass_from_dict(cls, data)
