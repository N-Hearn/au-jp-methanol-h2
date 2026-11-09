"""Golden-master regression tests.

    python -m unittest tests.test_golden_master

Compares a full SupplyChainResult (nested dataclasses and per-site dicts
included) against fixtures captured by tests/golden/capture_baseline.py. Floats compare to FLOAT_RTOL (relative) or
FLOAT_ATOL (absolute, for zero residuals); ints, bools, strings and counts
compare exactly. A field that needs a wider tolerance goes in RTOL_OVERRIDES
with the reason in the commit message. Physical balances are checked
separately by meoh_pathway/validation.py.

model_fingerprint() hashes inputs, not code, so these tests are what shows a
code change still reproduces the saved results. A failure message ends with
its likely cause, read from tests/golden/captured_from.json: the inputs
changed, optimum.json changed, or neither (the code moved the results). See
tests/golden/capture_baseline.py for when and how to re-capture.
"""

import dataclasses
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from meoh_pathway.optimisation import result_from_saved
from meoh_pathway.supply_chain import SupplyChain
from meoh_pathway import typical
from meoh_pathway.typical import typical_seed
from tests.golden.capture_baseline import current_source, load_captured_from

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
OPTIMUM_PATH = Path(__file__).resolve().parent.parent / "outputs" / "optimised_build" / "optimum.json"

# Dispatch starts from the storage state of charge the last run ended with, so
# results can differ in the last few digits; 1e-6 relative still catches real
# changes. Near-zero solver residuals (e.g. unmet_h2_kg) get an absolute floor.
FLOAT_RTOL = 1e-6
FLOAT_ATOL = 1e-4

# path -> rtol for fields that need more than FLOAT_RTOL. Empty by default.
RTOL_OVERRIDES = {}


def _flatten(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(_flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = obj
    return out


def _diff(baseline, current, rtol_overrides):
    b, c = _flatten(baseline), _flatten(current)
    diffs = []
    for path in sorted(set(b) | set(c)):
        if path not in b or path not in c:
            diffs.append((path, b.get(path), c.get(path), float("inf")))
            continue
        bv, cv = b[path], c[path]
        if isinstance(bv, float) and isinstance(cv, float):
            if bv == cv:
                continue
            rtol = rtol_overrides.get(path, FLOAT_RTOL)
            rel = abs(bv - cv) / max(abs(bv), abs(cv), 1e-30)
            if rel <= rtol or abs(bv - cv) <= FLOAT_ATOL:
                continue
            diffs.append((path, bv, cv, rel))
        elif bv != cv:
            diffs.append((path, bv, cv, float("inf")))
    diffs.sort(key=lambda d: -d[3])
    return diffs


def _format(diffs, limit=30):
    lines = [f"  {p}: baseline={bv!r} current={cv!r} rel={rel:.3g}" for p, bv, cv, rel in diffs[:limit]]
    if len(diffs) > limit:
        lines.append(f"  ... and {len(diffs) - limit} more")
    return "\n".join(lines)


def _cause(name):
    """Why a fixture may have moved, from what it was captured against."""
    recapture = f"python -m tests.golden.capture_baseline {name}"
    captured = load_captured_from().get(name)
    if captured is None:
        return (f"Cause unknown: captured_from.json has no record for {name}. "
                f"If the change was intended, re-capture to add one: {recapture}")
    current = current_source(name)
    if captured.get("model_fingerprint") != current["model_fingerprint"]:
        return ("Likely cause: the inputs (workbook values or hourly profiles) changed since "
                "capture, so this is expected. Re-run the analysis, check the diff is the "
                "change you meant, then re-capture all three: python -m tests.golden.capture_baseline")
    if captured.get("optimum_sha256") != current.get("optimum_sha256"):
        return ("Likely cause: outputs/optimised_build/optimum.json changed since capture "
                "(re-optimised), so this is expected. Check the diff, then re-capture: "
                "python -m tests.golden.capture_baseline optimum frozen")
    return ("Likely cause: a code change. Inputs and build are unchanged since capture. If "
            "the change was meant to move these results (a model correction, new physics), "
            f"check the diff is what you expected, re-capture ({recapture}) and say why in "
            "the commit message. If it was a refactor, it changed behaviour: fix the code.")


class GoldenMasterTests(unittest.TestCase):
    def _check(self, name, result):
        baseline = json.loads((GOLDEN_DIR / f"{name}.json").read_text())
        current = json.loads(json.dumps(dataclasses.asdict(result), allow_nan=False))
        diffs = _diff(baseline, current, RTOL_OVERRIDES)
        if diffs:
            self.fail(f"{name}.json differs from current run ({len(diffs)} fields):\n"
                      f"{_format(diffs)}\n{_cause(name)}")

    def test_optimum(self):
        result = result_from_saved(OPTIMUM_PATH, allow_stale=True)
        self._check("optimum", result.full_result)

    def test_typical(self):
        # Point TYPICAL_BUILD at an empty temp dir so run() solves afresh, as
        # the fixture was captured, instead of reusing a saved Typical build
        # (which only matches to 1e-7), and so outputs/ is left untouched.
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(typical, "TYPICAL_BUILD", Path(directory) / "typical_build.json"):
            try:
                self._check("typical", SupplyChain().run())
            finally:
                # The fresh solve is cached in-process and would leak into later
                # tests' runs; reseeding replaces it (writing only to the temp dir).
                typical_seed()

    def test_frozen(self):
        optimum = result_from_saved(OPTIMUM_PATH, allow_stale=True)
        frozen = SupplyChain(**SupplyChain.fixed_deployment(optimum.full_result)).run()
        self._check("frozen", frozen)


if __name__ == "__main__":
    unittest.main()
