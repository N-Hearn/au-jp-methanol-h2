"""Calculate and persist the hourly Typical balance at fixed site capacities.

Every port builds to its own workbook maximum wind, solar, battery and H2
storage (both regional and local) capacity -- there is no scale factor and
no per-site targets. The model's own energy/dispatch-based _site_potential
allocation (SupplyChain.run(), triggered by leaving site_h2_allocation_weight
unset) decides how much of the fixed fleet target each site actually
carries: a site whose storage cannot sustain its raw energy share is capped
at what it can carry, and the rest is redistributed to sites with headroom.
This already closes the fleet target at every port's documented maximum, so
there is nothing left to search -- solve_typical is one evaluation, not an
optimisation.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy
from dataclasses import fields

import numpy as np

from inputs.loader import load_site_inputs
from .supply_chain import SupplyChain


TYPICAL_BUILD = Path(__file__).resolve().parent.parent / "outputs/typical_build/typical_build.json"
_SEED_CACHE = {}
_BALANCE_CACHE = {}


def _adopt_chain(chain, evaluated):
    explicit = chain._explicit_allocation
    chain.__dict__.update(evaluated.__dict__)
    chain._explicit_allocation = explicit


def _saved_result(payload, evaluated):
    from .optimisation import OptResult
    values = {field.name: getattr(evaluated, field.name) for field in fields(OptResult)
              if hasattr(evaluated, field.name)}
    values.update(success=bool(payload.get("success", False)),
                  message=payload.get("message", "Loaded verified Typical balance"),
                  x=dict(zip(payload["bounds_names"], payload["x"])),
                  objective=evaluated.delivered_cost_usd_per_kg_h2,
                  full_result=evaluated, search_settings=payload.get("search_settings", {}),
                  nfev=int(payload.get("nfev", 0)))
    return OptResult(**values)


def _max_deployment():
    """Every port at its own workbook maximum wind, solar, battery and H2 storage."""
    rows = load_site_inputs()
    return dict(
        site_wind_mw={p: float(v["wind_mw"]["Upper bound"]) for p, v in rows.items()},
        site_solar_mw={p: float(v["solar_mw"]["Upper bound"]) for p, v in rows.items()},
        site_h2_storage_kg={p: float(v["h2_storage_kg"]["Upper bound"]) for p, v in rows.items()},
        site_battery_mwh={p: float(v["battery_mwh"]["Upper bound"]) for p, v in rows.items()},
        site_local_h2_storage_kg={p: float(v["local_h2_storage_kg"]["Upper bound"]) for p, v in rows.items()},
    )


def _calculated_x(deployment, result):
    """Turn a calculated build back into the DEFAULT_BOUNDS decision vector, so
    typical_seed()/warm_start() keep working off a plain vector like before."""
    from .optimisation import DEFAULT_BOUNDS, FROZEN_DECISIONS
    values = dict(FROZEN_DECISIONS)
    for name, lo, hi, integer in DEFAULT_BOUNDS:
        value = 0.0
        for field, mapping in deployment.items():
            prefix = field[5:] + "_"  # "site_wind_mw" -> "wind_mw_"
            if name.startswith(prefix):
                value = mapping.get(name[len(prefix):], 0.0)
                break
        else:
            # synth_modules is an absolute installed count (like pem_modules),
            # so it reads straight from result.site_n_synth.
            for field, mapping in (("h2_allocation_weight", result.site_h2_frac),
                                   ("pem_modules", result.site_n_pem),
                                   ("synth_modules", result.site_n_synth)):
                prefix = field + "_"
                if name.startswith(prefix):
                    value = mapping.get(name[len(prefix):], 0.0)
                    break
        values[name] = min(hi, max(lo, value))
    return np.array([values[name] for name, *_ in DEFAULT_BOUNDS], dtype=float)


def solve_typical(chain=None, *, allow_infeasible=False, save_path=None):
    """Calculate the Typical build (see module docstring for the rule).

    Saving is explicit; default model calls are read-only and reproduce the solve.
    Infeasible diagnostics are available to the wider optimiser as unverified seeds.
    """
    from .optimisation import DEFAULT_BOUNDS, model_fingerprint, build_supply_chain
    chain = chain or SupplyChain()
    overrides = {"_preserve_assets": True, "_site_storage_costs": dict(chain.site_storage_costs)}
    overrides.update({name: getattr(chain, attr) for name, attr in
                      (("econ", "econ"), ("pem", "pem"), ("synth", "synth"),
                       ("ship", "ship"), ("cracker", "cracker"), ("co2_cap", "co2_cap"),
                       ("dac", "dac"), ("target", "target"), ("transport", "transport"))})
    cache_key = (model_fingerprint(), repr(overrides))
    if save_path is None:
        baseline = SupplyChain()
        baseline_params = {name: getattr(baseline, name) for name in
                           ("econ", "pem", "synth", "ship", "cracker", "co2_cap", "dac", "target", "transport")}
        baseline_storage = {port: float(values["h2_storage_capex_aud_per_kg"]["Value"])
                            for port, values in load_site_inputs().items()}
        if (TYPICAL_BUILD.exists() and all(baseline_storage[port] == value
                for port, value in overrides["_site_storage_costs"].items())
                and all(overrides[name] == value for name, value in baseline_params.items())):
            try:
                payload = json.loads(TYPICAL_BUILD.read_text())
                if payload.get("model_fingerprint") == model_fingerprint() and payload.get("bounds_names") == [b[0] for b in DEFAULT_BOUNDS]:
                    evaluated_chain = build_supply_chain(payload["x"], overrides)
                    evaluated = evaluated_chain.run()
                    # Materialise the usual result interface without treating
                    # the saved search budget as proof of an optimum.
                    result = _saved_result(payload, evaluated)
                    _adopt_chain(chain, evaluated_chain)
                    _BALANCE_CACHE[cache_key] = deepcopy((result, evaluated_chain))
                    return result
            except (ValueError, KeyError, TypeError):
                pass
    if cache_key in _BALANCE_CACHE:
        result, evaluated_chain = deepcopy(_BALANCE_CACHE[cache_key])
        _adopt_chain(chain, evaluated_chain)
        if save_path is not None:
            save_typical(result, save_path)
        return result
    deployment = _max_deployment()
    params = {name: value for name, value in overrides.items() if not name.startswith("_")}
    keyword = {"econ": "economics", "co2_cap": "co2_capture", "transport": "h2_transport"}
    evaluated_chain = SupplyChain(**{keyword.get(name, name): value for name, value in params.items()},
                                  **deployment,
                                  site_storage_costs=dict(overrides.get("_site_storage_costs", {})),
                                  preserve_assets=True)
    # Leaving site_h2_allocation_weight unset (rather than passing energy-share
    # weights ourselves) is what routes this through the model's own
    # _site_potential allocation instead of a naive proportional split; that is
    # the mechanism that actually closes the fleet target at everyone's max.
    # _evaluating_typical only bypasses the
    # run()->solve_typical delegation on this specific chain -- see run().
    evaluated_chain._evaluating_typical = True
    full_result = evaluated_chain.run(allow_infeasible=allow_infeasible)
    # Match the saved-build path, which rebuilds the chain from x with explicit decisions.
    evaluated_chain.site_h2_allocation_weight = dict(full_result.site_h2_frac)
    evaluated_chain.site_pem_modules = dict(full_result.site_n_pem)
    evaluated_chain.site_synth_modules = dict(full_result.site_n_synth)
    _adopt_chain(chain, evaluated_chain)
    names = [b[0] for b in DEFAULT_BOUNDS]
    x = _calculated_x(deployment, full_result)
    search_settings = {
        "basis": "every port at its own workbook maximum wind, solar, battery and H2 "
                 "storage; allocation via the model's own _site_potential mechanism, "
                 "which caps a site short of storage and redistributes the rest",
        "model_fingerprint": model_fingerprint(),
    }
    message = ("Calculated build: every port at its workbook maximum capacity"
              if full_result.hourly_feasible else
              f"Typical build infeasible at workbook maximum: {full_result.unmet_h2_kg:,.0f} "
              f"kg H2 and {full_result.unmet_demand_mwh:,.3f} MWh unmet")
    result = SimpleNamespace(success=bool(full_result.hourly_feasible), message=message,
                             x=dict(zip(names, x)), nfev=1, search_settings=search_settings,
                             full_result=full_result)
    if full_result.hourly_feasible:
        _BALANCE_CACHE[cache_key] = deepcopy((result, evaluated_chain))
    if save_path is not None and full_result.hourly_feasible:
        save_typical(result, save_path)
    return result


def save_typical(result, path=None):
    from .optimisation import DEFAULT_BOUNDS, model_fingerprint, build_supply_chain
    from .validation import close, validate_traces
    names = [b[0] for b in DEFAULT_BOUNDS]
    if result.search_settings.get("model_fingerprint") != model_fingerprint():
        raise ValueError("Model changed after the Typical search; rerun before saving")
    if not result.full_result.hourly_feasible:
        raise ValueError("Cannot save an infeasible Typical balance")
    chain = build_supply_chain([result.x[name] for name in names], {"_preserve_assets": True})
    verified = chain.run()
    frozen_chain = SupplyChain(**SupplyChain.fixed_deployment(verified))
    frozen = frozen_chain.run()
    validate_traces(frozen, frozen_chain.dispatch_traces())
    close(frozen.delivered_cost_aud_per_kg_h2, verified.delivered_cost_aud_per_kg_h2,
          "Frozen Typical cost matches solved balance")
    close(result.full_result.delivered_cost_aud_per_kg_h2, verified.delivered_cost_aud_per_kg_h2,
          "Typical candidate matches current inputs")
    payload = dict(model_fingerprint=model_fingerprint(), bounds_names=names,
                   success=result.success, message=result.message, nfev=result.nfev,
                   x=[float(result.x[name]) for name in names], hourly_feasible=True,
                   installed_design=verified.installed_design, deployment=SupplyChain.fixed_deployment(verified),
                   search_settings=result.search_settings)
    # Resolved at call time so tests can redirect TYPICAL_BUILD.
    path = Path(TYPICAL_BUILD if path is None else path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def typical_seed():
    from .optimisation import DEFAULT_BOUNDS, model_fingerprint, build_supply_chain
    names = [b[0] for b in DEFAULT_BOUNDS]
    fingerprint = model_fingerprint()
    if fingerprint in _SEED_CACHE:
        return _SEED_CACHE[fingerprint].copy()
    if TYPICAL_BUILD.exists():
        try:
            payload = json.loads(TYPICAL_BUILD.read_text())
            if payload.get("model_fingerprint") == fingerprint and payload.get("bounds_names") == names:
                build_supply_chain(payload["x"], {"_preserve_assets": True}).run()
                _SEED_CACHE[fingerprint] = np.asarray(payload["x"], dtype=float)
                return _SEED_CACHE[fingerprint].copy()
        except (ValueError, KeyError, TypeError):
            pass
    try:
        result = solve_typical(allow_infeasible=True)
    except ValueError:
        # Zero Typical generation cannot form production shares, but the wider
        # renewable bounds may still contain a feasible full optimisation.
        from .optimisation import _workbook_x0
        _SEED_CACHE[fingerprint] = _workbook_x0()
        return _SEED_CACHE[fingerprint].copy()
    if result.full_result.hourly_feasible:
        save_typical(result)
    _SEED_CACHE[fingerprint] = np.array([result.x[name] for name in names], dtype=float)
    return _SEED_CACHE[fingerprint].copy()
