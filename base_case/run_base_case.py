"""Run the grey H2 reference case and save its result.

Usage:
    python -m base_case.run_base_case

The saved JSON sets the total H2 demand the MeOH pathway must deliver, so that
model reads it instead of re-running the grey chain. It carries a fingerprint of
the grey inputs and code, and load_grey() refuses a stale file.

Grey H2 is the reference case (outputs/reference_case/grey.json). The blue H2
base case (LNG + ATR + CCS, CO2 returned to Australia) is saved as
outputs/base_case/lng_ccs.json for reporting only; nothing reads it back.
"""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from .blue_h2 import BlueH2, BlueH2Result
from .delivered_h2 import DeliveredH2, DeliveredH2Result
from .parameters import (
    default_target, load_atr, load_atr_ccs, load_carrier, load_co2_liquefaction, load_discount,
    load_japan_plant, load_lng,
)

ROOT = Path(__file__).resolve().parent.parent
GREY_JSON = ROOT / "outputs" / "reference_case" / "grey.json"
BLUE_JSON = ROOT / "outputs" / "base_case" / "lng_ccs.json"


def run_grey() -> DeliveredH2Result:
    return DeliveredH2(
        load_lng(),
        load_atr(),
        default_target(),
        load_japan_plant(),
    ).run()


def run_blue() -> BlueH2Result:
    return BlueH2(load_lng(), load_atr(), load_atr_ccs(), default_target(), load_japan_plant(),
                  load_co2_liquefaction(), load_carrier(), load_discount()).run()


def inputs_fingerprint() -> str:
    # Inputs only, like model_fingerprint: a code change does not make grey.json stale.
    digest = hashlib.sha256()
    inputs = {
        "lng": asdict(load_lng()),
        "atr": asdict(load_atr()),
        "target": asdict(default_target()),
        "japan": asdict(load_japan_plant()),
    }
    digest.update(json.dumps(inputs, sort_keys=True).encode())
    return digest.hexdigest()


def save_grey(result: DeliveredH2Result, path: Path = GREY_JSON) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"inputs_fingerprint": inputs_fingerprint(), "result": asdict(result)}
    path.write_text(json.dumps(payload, indent=2))


def load_grey(path: Path = GREY_JSON) -> DeliveredH2Result:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No saved grey result at {path}; run python -m base_case.run_base_case"
        )
    payload = json.loads(path.read_text())
    if payload.get("inputs_fingerprint") != inputs_fingerprint():
        raise ValueError(
            f"{path} predates the current grey inputs; "
            "run python -m base_case.run_base_case"
        )
    return DeliveredH2Result(**payload["result"])


def main():
    result = run_grey()
    save_grey(result)
    print(
        f"Grey H2 (reference case): {result.target_h2_tpa / 1e6:.2f} Mt/yr, "
        f"{result.delivered_cost_aud_per_kg_h2:.3f} AUD/kg H2, "
        f"EI {result.delivered_ei_kg_co2_per_kg_h2:.2f} kg CO2/kg H2"
    )
    print(f"Saved {GREY_JSON}")

    blue = run_blue()
    BLUE_JSON.parent.mkdir(parents=True, exist_ok=True)
    BLUE_JSON.write_text(json.dumps(asdict(blue), indent=2))
    print(
        f"Blue H2 (LNG + CCS, base case): {blue.delivered_cost_aud_per_kg_h2:.3f} AUD/kg H2, "
        f"EI {blue.delivered_ei_kg_co2_per_kg_h2:.2f} kg CO2/kg H2, "
        f"{blue.fleet_co2_stored_tpa / 1e6:.1f} Mt CO2/yr returned"
    )
    print(f"Saved {BLUE_JSON}")


if __name__ == "__main__":
    main()
