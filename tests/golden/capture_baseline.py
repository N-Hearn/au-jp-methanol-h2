"""Capture golden-master fixtures.

    python -m tests.golden.capture_baseline                 # all three
    python -m tests.golden.capture_baseline optimum frozen  # only these

Writes tests/golden/<name>.json via dataclasses.asdict() on the full
SupplyChainResult. test_golden_master.py compares later runs against these
field-by-field.

The fixtures are the record of what the model currently produces:

    typical  SupplyChain().run() from the workbook and hourly profiles alone
             (every port at its workbook maximum, fresh solve).
    optimum  outputs/optimised_build/optimum.json replayed through the model.
    frozen   that optimum's installed design re-run as a fixed deployment
             (SupplyChain.fixed_deployment).

When to re-capture
------------------
Only when a change is meant to move the results, and only the fixtures it is
meant to move:

  - workbook values or hourly profiles changed (model_fingerprint changes):
    all three, after re-running the analysis so optimum.json is current;
  - optimum.json re-optimised (--reoptimise, new seed or search settings):
    optimum and frozen; typical does not read the saved build;
  - a deliberate model change (new physics, corrected cost): whichever
    fixtures the test run shows moving.

A refactor must reproduce the existing fixtures -- do not re-capture to make
a failing test pass. Fix the code, or widen one field in RTOL_OVERRIDES with
the reason in the commit message.

Re-capturing
------------
1. Run python -m unittest tests.test_golden_master and read the diff; it
   lists the fields that moved, largest relative change first. Check they
   are the ones the change should move.
2. Capture the affected fixtures with this script.
3. Re-run the tests (they should pass) and commit the fixtures together with
   the change that moved them, saying why in the commit message, so the
   history shows which results each change shifted.

optimum and frozen load optimum.json with allow_stale=True, so they still
replay a build saved under older inputs; that lets them check a code change
without re-optimising first. The fixture is captured against whatever build
is committed, so commit optimum.json and the fixtures together.

captured_from.json
------------------
Each capture also records, per fixture, what it was captured against: the
model_fingerprint() (workbook values and hourly profiles) and, for optimum
and frozen, a hash of optimum.json. When a test fails it compares these with
the current values to say whether the inputs changed, the build changed, or
neither (so the code moved the results). Never edit it by hand; it is
rewritten for each fixture you capture.
"""

import argparse
import dataclasses
import hashlib
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from meoh_pathway import typical
from meoh_pathway.optimisation import model_fingerprint, result_from_saved
from meoh_pathway.supply_chain import SupplyChain

GOLDEN_DIR = Path(__file__).resolve().parent
OPTIMUM_PATH = GOLDEN_DIR.parent.parent / "outputs" / "optimised_build" / "optimum.json"
CAPTURED_FROM = GOLDEN_DIR / "captured_from.json"
NAMES = ("optimum", "typical", "frozen")
# Fixtures that replay the saved build rather than solving from the inputs.
REPLAYS_OPTIMUM = ("optimum", "frozen")


def optimum_sha256():
    # Line endings normalised, as in model_fingerprint(), so a CRLF checkout
    # does not read as a changed build.
    return hashlib.sha256(OPTIMUM_PATH.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def current_source(name):
    """What a fixture captured now would be captured against."""
    source = {"model_fingerprint": model_fingerprint()}
    if name in REPLAYS_OPTIMUM:
        source["optimum_sha256"] = optimum_sha256()
    return source


def load_captured_from():
    try:
        return json.loads(CAPTURED_FROM.read_text())
    except FileNotFoundError:
        return {}


def _write(name, result):
    data = dataclasses.asdict(result)
    (GOLDEN_DIR / f"{name}.json").write_text(json.dumps(data, indent=2, allow_nan=False))
    sources = load_captured_from()
    sources[name] = current_source(name)
    CAPTURED_FROM.write_text(json.dumps(dict(sorted(sources.items())), indent=2) + "\n")
    print(f"wrote {name}.json")


def capture(names=NAMES):
    if "optimum" in names or "frozen" in names:
        optimum = result_from_saved(OPTIMUM_PATH, allow_stale=True)
    if "optimum" in names:
        _write("optimum", optimum.full_result)

    if "typical" in names:
        # Solve afresh rather than reuse a saved Typical build, as test_typical
        # does, without touching outputs/.
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(typical, "TYPICAL_BUILD", Path(directory) / "typical_build.json"):
            try:
                _write("typical", SupplyChain().run())
            finally:
                typical.typical_seed()

    if "frozen" in names:
        frozen = SupplyChain(**SupplyChain.fixed_deployment(optimum.full_result)).run()
        _write("frozen", frozen)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*", choices=NAMES, default=list(NAMES))
    capture(parser.parse_args().names)
