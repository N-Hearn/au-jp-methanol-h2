"""Vary only where production sits: re-site the optimised build, re-size it to feasibility.

python -m scripts.design_analysis.colocation [--workers 10] [--budget 1500] [--max-removed 3]
python -m scripts.design_analysis.colocation --replot

Every technology and market input stays at its workbook Typical value; the only
thing that changes is the set of production ports. Starting from the saved
optimum, each alternative footprint

  * drops one to --max-removed of the optimum's sites, its share redistributed
    onto the survivors with their capacities (the optimiser's own site closure),
  * adds an unused viable port, taking part of the largest site's share, or
  * swaps one optimum site for an unused port, its whole share moved across,

is repaired to hourly feasibility, and is then locally refined (PEM reduction,
share transfers between the footprint's own sites, coordinate steps) with the
footprint held fixed, until no cheaper move is found within --budget evaluations.

The optimum came from a full global search; the alternatives get local
refinement only, so each alternative's cost is an upper bound on what that
footprint could achieve and the penalties shown are conservative in its favour.
Infeasible footprints are reported in colocation.json and left off the plots.

Writes outputs/optimised_build/design_analysis/colocation/colocation.json,
colocation_cost_ei.png (delivered
cost against emissions intensity, coloured by how the footprint changed and
numbered by cost rank), colocation_footprints.png (every feasible footprint as a
row under the same rank: the ports it uses, sized by share, beside its cost and
emissions, with the delivery-margin builds from outputs/optimised_build/
margin_cases appended below for comparison) and colocation_single_site.png (the cost and emissions change from
removing, adding or swapping each single site).
"""

import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path

import numpy as np

from scripts.common.cli import build_parser
from scripts.common.outputs import MARGIN_DIR, OPT_DIR, design_dir

COLOCATION_DIR = design_dir(OPT_DIR) / "colocation"
_CLOSURE_MARGINS = (0.0, 0.1, 0.25, 0.5)
_MIN_IMPROVEMENT = 1e-3  # USD/kg between refinement rounds


def _footprints(optimum_sites, viable, max_removed):
    """(label kind, removed, added) per alternative, optimum first."""
    unused = sorted(set(viable) - set(optimum_sites))
    plans = [("optimum", (), ())]
    for k in range(1, max_removed + 1):
        plans += [(f"-{k}", removed, ()) for removed in combinations(sorted(optimum_sites), k)]
    for k in range(1, len(unused) + 1):
        plans += [(f"+{k}", (), added) for added in combinations(unused, k)]
    plans += [("swap", (removed,), (added,)) for removed in sorted(optimum_sites) for added in unused]
    return plans


def _site_weights(x):
    from meoh_pathway.optimisation import deployment_from_vector
    return deployment_from_vector(list(x))["site_h2_allocation_weight"]


def _resize_synth(x, x0, bnds):
    """Scale each site's synthesis trains with its share of production.

    The optimiser's closure and transfer moves carry renewables, storage and PEM
    with a share but leave synthesis trains where they were, so a survivor's
    fleet falls short and the run fails outright -- a flat penalty refinement
    cannot climb out of. A site new to the footprint takes the fleet's average
    trains per unit share. Rounded up; coordinate refinement trims any excess.
    """
    from meoh_pathway.optimisation import DEFAULT_BOUNDS
    if x is None:
        return None
    names = [name for name, *_ in DEFAULT_BOUNDS]
    pos = {name: i for i, name in enumerate(names)}
    old, new = _site_weights(x0), _site_weights(x)
    old_total, new_total = sum(old.values()), sum(new.values())
    # Active sites only: an unused port keeps whatever train count the vector last
    # held (over a thousand in a saved optimum), which would inflate the average.
    trains = {p: x0[pos[f"synth_modules_{p}"]] for p, w in old.items()
              if w > 0 and f"synth_modules_{p}" in pos}
    per_share = sum(trains.values()) / old_total
    x = np.asarray(x, dtype=float).copy()
    for port, weight in new.items():
        key = f"synth_modules_{port}"
        if key not in pos or weight <= 0:
            continue
        share = weight / new_total
        if old.get(port, 0) > 0:
            count = trains[port] * share / (old[port] / old_total)
        else:
            count = per_share * share * old_total
        lo, hi = bnds[pos[key]]
        x[pos[key]] = min(hi, max(lo, math.ceil(count - 1e-9)))
    return x


def _start(x0, bnds, removed, added, evaluate, closure_first=False):
    """A starting vector for the footprint, or None if the survivors cannot absorb it."""
    x0 = np.asarray(x0, dtype=float)
    x = _start_moves(x0, bnds, removed, added, evaluate, closure_first)
    return _resize_synth(x, x0, bnds)


def _grown(x, bnds, port, margin):
    """x with the port's capacities over-provisioned by margin, within its bounds."""
    from meoh_pathway.optimisation import CAPACITY_DECISIONS, DEFAULT_BOUNDS
    x = np.asarray(x, dtype=float).copy()
    for i, (name, _, _, integer) in enumerate(DEFAULT_BOUNDS):
        if name in {f"{field}_{port}" for field in CAPACITY_DECISIONS}:
            value = min(bnds[i][1], max(bnds[i][0], x[i] * (1 + margin)))
            x[i] = math.ceil(value) if integer else value
    return x


def _best_grown(x, x0, bnds, port, evaluate):
    """The receiving port's over-provision the evaluation prices best."""
    if x is None:
        return None
    return min((_grown(x, bnds, port, m) for m in _CLOSURE_MARGINS),
               key=lambda t: evaluate(_resize_synth(t, x0, bnds)))


def _start_moves(x0, bnds, removed, added, evaluate, closure_first=False):
    from meoh_pathway.optimisation import _transfer_share
    x = np.asarray(x0, dtype=float)
    if len(removed) == 1 and len(added) == 1 and not closure_first:
        return _best_grown(_transfer_share(x, bnds, removed[0], added[0], 1.0), x0, bnds, added[0], evaluate)
    if closure_first:
        x = _close_all(x, x0, bnds, removed, evaluate)
        removed = ()
        if x is None:
            return None
    for port in added:
        # A new port enters at the smallest active site's share, taken from the
        # largest: a fixed fraction of a dominant site overwhelms a minor port's caps.
        weights = {p: w for p, w in _site_weights(x).items() if w > 0 and p not in removed}
        donor = max(weights, key=weights.get)
        x = _best_grown(_transfer_share(x, bnds, donor, port, min(weights.values()) / weights[donor]),
                        x0, bnds, port, evaluate)
        if x is None:
            return None
    return _close_all(x, x0, bnds, removed, evaluate)


def _close_all(x, x0, bnds, removed, evaluate):
    from meoh_pathway.optimisation import _close_and_redistribute
    for port in removed:
        # Proportional growth alone usually lands just short; take whichever
        # over-provision the evaluation prices best, refinement trims it back.
        trials = [t for t in (_close_and_redistribute(x, bnds, port, m) for m in _CLOSURE_MARGINS)
                  if t is not None]
        if not trials:
            return None
        x = min(trials, key=lambda t: evaluate(_resize_synth(t, x0, bnds)))
    return x


def _grow_step(x, bnds, name):
    """One step up for a capacity: a module at a time while the count is small, else +25%."""
    from meoh_pathway.optimisation import DEFAULT_BOUNDS
    i = [n for n, *_ in DEFAULT_BOUNDS].index(name)
    if DEFAULT_BOUNDS[i][3]:
        value = x[i] + 1 if x[i] < 8 else math.ceil(x[i] * 1.25)
    else:
        value = max(x[i] * 1.25, x[i] + 1)
    y = np.asarray(x, dtype=float).copy()
    y[i] = min(bnds[i][1], value)
    return None if y[i] == x[i] else y


def _repair(x, bnds, added, measure):
    """Grow capacities or hand share back until the footprint is hourly feasible.

    A start is rarely feasible: share moves carry the donor's capacity mix to a
    port whose profiles want another, a new port enters with more share than its
    caps can carry, and integer PEM rounds down at the donor. Left to cost
    refinement, an infeasible start's penalty is cut fastest by shrinking the
    short port's share, so an added port withers to a few tonnes below PEM
    minimum load and never recovers. Repair instead takes, each step, whichever
    move cuts unmet hydrogen most: one capacity increase at a short site, or
    half an added port's share handed back to one of the original sites, which
    takes the capacity with it. Refinement trims the over-provision afterwards.
    An exhausted budget keeps the progress made so far.
    """
    from meoh_pathway.optimisation import CAPACITY_DECISIONS, DEFAULT_BOUNDS, _BudgetExhausted, _transfer_share
    names = {n for n, *_ in DEFAULT_BOUNDS}
    try:
        unmet, short = measure(x)
        while unmet > 0:
            weights = {p: w for p, w in _site_weights(x).items() if w > 0}
            trials = [_grow_step(x, bnds, f"{field}_{port}") for port in short
                      for field in CAPACITY_DECISIONS + ("synth_modules",) if f"{field}_{port}" in names]
            trials += [_transfer_share(x, bnds, port, receiver, 0.5) for port in added if port in weights
                       for receiver in weights if receiver not in added]
            best = None
            for trial in trials:
                if trial is not None:
                    m = measure(trial)
                    if best is None or m[0] < best[0][0]:
                        best = (m, trial)
            if best is None or best[0][0] >= unmet:
                break
            (unmet, short), x = best
    except _BudgetExhausted:
        pass
    return x


def _run_footprint(args):
    kind, removed, added, x0, budget = args
    from meoh_pathway import optimisation as opt
    bnds = [(lo, hi) for _, lo, hi, _ in opt.DEFAULT_BOUNDS]
    target = frozenset(opt._active_sites(x0)) - set(removed) | set(added)
    memo, spent = {}, [0]

    def evaluate(trial, weight=0.0, overrides=None):
        trial = np.asarray(trial, dtype=float)
        key = trial.tobytes()
        if key not in memo:
            if opt._active_sites(trial) != target:
                return 1e9  # a move that changes the footprint is not this experiment
            if spent[0] >= budget:
                raise opt._BudgetExhausted
            memo[key] = opt._evaluate(trial, weight, overrides)
            spent[0] += 1
        return memo[key]

    def measure(trial):
        """Unmet hydrogen (0 when hourly feasible) and the short sites, on the same budget."""
        trial = np.asarray(trial, dtype=float)
        if opt._active_sites(trial) != target:
            return math.inf, {}
        if spent[0] >= budget:
            raise opt._BudgetExhausted
        spent[0] += 1
        try:
            r = opt.build_supply_chain(list(trial)).run(allow_infeasible=True)
        except (ValueError, ArithmeticError, RuntimeError):
            return math.inf, {}
        short = {p: s["unmet_h2_kg"] for p, s in r.site_shortfalls.items() if s["unmet_h2_kg"] > 1}
        return (0.0 if r.hourly_feasible else sum(short.values()) + 1.0), short

    started = time.perf_counter()
    record = dict(kind=kind, removed=list(removed), added=list(added), sites=sorted(target))
    # A swap starts either by moving the dropped port's whole share to the new
    # port, or by closing it onto the survivors first and bringing the new port
    # in small. Neither start's own shortfall predicts which repairs better (a
    # Gladstone swap repairs only from the second), so repair both, keep the closer.
    repaired = []
    for closure_first in (False, True) if removed and added else (False,):
        try:
            x = _start(x0, bnds, removed, added, evaluate, closure_first)
            if x is not None and opt._active_sites(x) == target:
                x = _repair(x, bnds, added, measure)
                repaired.append((measure(x)[0], len(repaired), x))
        except opt._BudgetExhausted:
            pass
    if not repaired:
        return record | dict(feasible=False, reason="survivors cannot absorb the share", evaluations=spent[0])
    x = min(repaired)[2]

    # Pin every variable of a port outside the footprint, so refinement spends
    # nothing on coordinates that cannot change this footprint's build.
    names = [name for name, *_ in opt.DEFAULT_BOUNDS]
    pinned = [(lo, hi) if any(n.endswith("_" + p) for p in target) else (v, v)
              for n, (lo, hi), v in zip(names, bnds, x)]
    best = [np.asarray(x, dtype=float), math.inf]  # valued in the try below: repair may spend the budget

    def tracked(trial, weight=0.0, overrides=None):
        value = evaluate(trial, weight, overrides)
        if value < best[1]:
            best[0], best[1] = np.asarray(trial, dtype=float).copy(), value
        return value

    stop = "round limit"
    try:
        tracked(x)
        for _ in range(20):
            before = best[1]
            for stage in (opt._reduce_pem,
                          lambda *a, **k: opt._refine_shares(*a, fractions=(0.3, 0.1), **k),
                          opt._refine_feasible):
                stage(best[0], pinned, 0.0, evaluate=tracked)
            if before - best[1] <= _MIN_IMPROVEMENT:
                stop = "improvement threshold"
                break
    except opt._BudgetExhausted:
        stop = "evaluation budget"

    try:
        r = opt.build_supply_chain(list(best[0])).run(allow_infeasible=True)
    except (ValueError, ArithmeticError, RuntimeError) as exc:
        return record | dict(feasible=False, reason=str(exc), evaluations=spent[0], stop_reason=stop,
                             x=[float(v) for v in best[0]])
    share = r.site_h2_frac
    return record | dict(
        feasible=bool(r.hourly_feasible),
        reason="" if r.hourly_feasible else "no hourly-feasible build found within the budget",
        delivered_cost_aud_per_kg_h2=r.delivered_cost_aud_per_kg_h2,
        delivered_ei_kg_co2_per_kg_h2=r.delivered_ei_kg_co2_per_kg_h2,
        ei_components={k: getattr(r, k) for k in (
            "cracker_vent_ei", "synth_process_ei", "ship_fuel_ei", "ship_fossil_fuel_ei",
            "meoh_loss_ei", "co2_capture_ei", "pem_electricity_ei", "synth_electricity_ei")},
        cost_breakdown_aud_per_kg_h2=dict(r.cost_breakdown_aud_per_kg_h2),
        circularity_rate=r.circularity_rate,
        share_weighted_voyage_nm=sum(share[p] * r.site_routes[p]["route_distance_nm"] for p in share),
        site_h2_frac=dict(share),
        evaluations=spent[0], stop_reason=stop, seconds=time.perf_counter() - started,
        x=[float(v) for v in best[0]],
    )


class _Serial:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def compute(optimum_path, workers, budget, max_removed):
    from meoh_pathway.optimisation import DEFAULT_BOUNDS, _active_sites, model_fingerprint, warm_start
    x0 = warm_start(optimum_path)
    optimum_sites = sorted(_active_sites(x0))
    viable = sorted(n.removeprefix("h2_allocation_weight_") for n, *_ in DEFAULT_BOUNDS
                    if n.startswith("h2_allocation_weight_"))
    plans = _footprints(optimum_sites, viable, max_removed)
    print(f"{len(plans)} footprints from {len(optimum_sites)} optimum sites "
          f"({len(viable)} viable ports), budget {budget} evaluations each", flush=True)
    jobs = [(kind, removed, added, x0, budget) for kind, removed, added in plans]
    results = []
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    with pool or _Serial():
        for i, record in enumerate((pool.map if pool else map)(_run_footprint, jobs), 1):
            results.append(record)
            cost = record.get("delivered_cost_aud_per_kg_h2", math.nan)
            print(f"  [{i}/{len(jobs)}] {record['kind']:8} -{record['removed']} +{record['added']}: "
                  f"{'feasible' if record['feasible'] else 'INFEASIBLE'} {cost:.3f} AUD/kg", flush=True)
    return dict(
        model_fingerprint=model_fingerprint(), optimum=str(optimum_path),
        optimum_sites=optimum_sites, viable_ports=viable, budget=budget, max_removed=max_removed,
        basis=("Typical inputs throughout; only the production footprint changes. Alternatives are "
               "locally refined from the optimum, so their costs are upper bounds."),
        footprints=results)


# --- plotting ---------------------------------------------------------------

# Colour by direction of change: blues darkening with the number of sites
# removed (the lightest still readable on white), orange for sites added or
# swapped, black for the optimum. Each also has its own marker shape, so no
# category is told apart by colour alone.
_STYLE = {
    "optimum": dict(color="#0b0b0b", marker="*", s=220, label="Optimum footprint", zorder=5),
    "-1": dict(color="#86b6ef", marker="o", s=34, label="1 site removed"),
    "-2": dict(color="#2a78d6", marker="s", s=30, label="2 sites removed"),
    "-3": dict(color="#104281", marker="^", s=36, label="3 sites removed"),
    "add": dict(color="#eb6834", marker="D", s=34, label="Site(s) added"),
    "swap": dict(color="#eb6834", marker="P", s=46, label="One site swapped for an unused port"),
}


def _style_key(kind):
    if kind.startswith("+"):
        return "add"
    return kind if kind in _STYLE else "-3"


def _short(port):
    from scripts.common.plot_common import port_label
    return port_label(port).replace("Port ", "Pt ")


# West to east, with the state each port sits in, so regional clustering reads directly.
_PORT_ORDER = (("Ashburton", "WA"), ("Dampier", "WA"), ("PortHedland", "WA"), ("PortDarwin", "NT"),
               ("PortBonython", "SA"), ("PortHastings", "VIC"), ("Newcastle", "NSW"),
               ("Gladstone", "QLD"), ("HayPoint", "QLD"), ("PortAbbotPoint", "QLD"))


# The delivery-margin builds keep their builds.optimisation colours: the 10% build is
# the proposed build, the greens darkening with the margin (MARGIN_COLOURS).
_MARGIN_STYLE = dict(marker="h", s=60)
_PROPOSED_COLOUR = "#01672d"  # MARGIN_COLOURS[10]


def _margin_rows(optimum_sites):
    """Each delivery-margin build as a footprint row, its ports compared with the optimum's."""
    import re
    rows = []
    for path in MARGIN_DIR.glob("margin_*pct"):
        match = re.fullmatch(r"margin_([\d.]+)pct", path.name)
        if not (match and (path / "results.json").exists()):
            continue
        r = json.loads((path / "results.json").read_text())
        share = {p: s for p, s in r["site_h2_frac"].items() if s > 0}
        pct = float(match.group(1))
        rows.append(dict(
            kind="margin", pct=pct, site_h2_frac=share,
            removed=sorted(set(optimum_sites) - set(share)), added=sorted(set(share) - set(optimum_sites)),
            delivered_cost_aud_per_kg_h2=r["delivered_cost_aud_per_kg_h2"],
            delivered_ei_kg_co2_per_kg_h2=r["delivered_ei_kg_co2_per_kg_h2"]))
    return sorted(rows, key=lambda r: r["pct"])


def _row_label(r, rank):
    if r["kind"] == "margin":
        return f"+{r['pct']:g}% margin" + (" (proposed)" if math.isclose(r["pct"], 10) else "")
    return f"{rank[id(r)]}" + ("  optimum" if r["kind"] == "optimum" else "")


def _plot_footprints(rows, rank, payload, outdir):
    """Which ports each feasible footprint uses, beside its cost and emissions.

    Dot area is the port's share of delivered hydrogen, in the neutral grey: use
    carries no identity, only change does. A blue dash marks an optimum port the
    footprint dropped; an orange ring marks a port it added. The delivery-margin
    builds follow the ranked footprints, below a rule: each is its own full
    optimisation to a larger delivery target, so it takes no cost rank.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from scripts.common.plot_common import NEUTRAL, NEUTRAL_LIGHT, margin_colour, save_fig

    ports = [p for p, _ in _PORT_ORDER if p in payload["viable_ports"]]
    ports += sorted(set(payload["viable_ports"]) - set(ports))
    states = dict(_PORT_ORDER)
    n_ranked = len(rows)
    rows = rows + _margin_rows(payload["optimum_sites"])
    n = len(rows)
    y = np.arange(n)
    fig, axes = plt.subplots(1, 3, figsize=(12, 0.22 * n + 1.9), sharey=True, layout="constrained",
                             gridspec_kw=dict(width_ratios=[len(ports) * 0.42, 2.2, 2.2]))
    ax = axes[0]
    for i, r in enumerate(rows):
        share = r["site_h2_frac"]
        for j, port in enumerate(ports):
            if share.get(port, 0) > 0:
                added = port in r["added"]
                ax.scatter(j, i, s=12 + 380 * share[port], color="white" if added else NEUTRAL,
                           edgecolor=_STYLE["add"]["color"] if added else NEUTRAL,
                           linewidth=1.6 if added else 0, zorder=3)
            elif port in r["removed"]:
                ax.scatter(j, i, marker="_", s=70, color=_STYLE["-2"]["color"], linewidth=2, zorder=3)
            else:
                ax.scatter(j, i, s=4, color=NEUTRAL_LIGHT, zorder=2)
    ax.set_xticks(range(len(ports)), [f"{_short(p)}\n({states.get(p, '')})" for p in ports],
                  rotation=60, ha="right", fontsize=7.5)
    ax.set_yticks(y, [_row_label(r, rank) for r in rows], fontsize=7)
    ax.set_xlim(-0.6, len(ports) - 0.4)
    ax.set_ylabel("Footprint (cost rank)")
    ax.set_title("Ports used (dot area = share of H2)", fontsize=9.5)
    ax.grid(False)
    fig.legend(handles=[
        Line2D([], [], marker="o", ls="", color=NEUTRAL, markersize=7, label="Used"),
        Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=_STYLE["add"]["color"],
               markeredgewidth=1.6, markersize=7, label="Added vs optimum"),
        Line2D([], [], marker="_", ls="", color=_STYLE["-2"]["color"], markersize=9, markeredgewidth=2,
               label="Removed vs optimum"),
        Line2D([], [], marker=_MARGIN_STYLE["marker"], ls="", color=_PROPOSED_COLOUR, markersize=7,
               label="Delivery-margin build (own optimisation, unranked)")],
        frameon=False, fontsize=8, loc="outside lower left", ncol=4)

    for ax, key, xlabel in ((axes[1], "delivered_cost_aud_per_kg_h2", "Delivered cost [AUD/kg H2]"),
                            (axes[2], "delivered_ei_kg_co2_per_kg_h2", "Emissions intensity [kg CO2/kg H2]")):
        ref = next(r for r in rows if r["kind"] == "optimum")[key]
        ax.axvline(ref, color="black", linewidth=0.8, linestyle=":", zorder=1)
        for i, r in enumerate(rows):
            if r["kind"] == "margin":
                colour = margin_colour(r["pct"])
                style = dict(color=colour, **_MARGIN_STYLE)
            else:
                style = _STYLE[_style_key(r["kind"])]
            ax.scatter(r[key], i, color=style["color"], marker=style["marker"],
                       s=120 if r["kind"] == "optimum" else min(style["s"], 60),
                       edgecolor="white", linewidth=0.6, zorder=3)
        ax.set_xlabel(xlabel, fontsize=8.5)
        ax.tick_params(axis="x", labelsize=7.5)
        ax.grid(axis="y", visible=False)
    axes[1].set_title("Cost (dotted line: optimum)", fontsize=9.5)
    axes[2].set_title("Emissions intensity", fontsize=9.5)
    if n > n_ranked:
        for ax in axes:
            ax.axhline(n_ranked - 0.5, color=NEUTRAL, linewidth=0.8, zorder=1)
    axes[0].set_ylim(n - 0.4, -0.6)  # top row first, no dead band above or below
    fig.suptitle("Co-location footprints: ports used, cost and emissions (feasible builds, cheapest first; "
                 "delivery-margin builds below)",
                 fontsize=10.5, fontweight="bold")
    save_fig(fig, outdir / "colocation_footprints.png")


def plot(payload, outdir):
    import matplotlib.pyplot as plt
    from scripts.common.plot_common import save_fig

    rows = [r for r in payload["footprints"] if r["feasible"]]
    optimum = next(r for r in rows if r["kind"] == "optimum")
    n_bad = len(payload["footprints"]) - len(rows)

    # One rank per feasible footprint, cheapest first; the scatter and the
    # footprint matrix share it, so a point can be traced to the ports it uses.
    rows.sort(key=lambda r: r["delivered_cost_aud_per_kg_h2"])
    rank = {id(r): i + 1 for i, r in enumerate(rows)}

    fig, ax = plt.subplots(figsize=(7.5, 5.2), layout="constrained")
    for key, style in _STYLE.items():
        pts = [r for r in rows if _style_key(r["kind"]) == key]
        if not pts:
            continue
        ax.scatter([r["delivered_cost_aud_per_kg_h2"] for r in pts],
                   [r["delivered_ei_kg_co2_per_kg_h2"] for r in pts],
                   edgecolor="white", linewidth=0.8, alpha=0.9, **style)
    for r in rows:
        ax.annotate(str(rank[id(r)]), (r["delivered_cost_aud_per_kg_h2"], r["delivered_ei_kg_co2_per_kg_h2"]),
                    xytext=(4, 3), textcoords="offset points", fontsize=6, color="#6b6a65")
    ax.set(xlabel="Delivered cost [AUD/kg H2]", ylabel="Emissions intensity [kg CO2/kg H2]",
           title="Co-location: cost and emissions of alternative production footprints")
    ax.legend(frameon=False, fontsize=8.5, loc="best")
    ei = [r["delivered_ei_kg_co2_per_kg_h2"] for r in rows]
    ax.text(1.0, -0.13, f"{len(rows)} feasible footprints; {n_bad} infeasible excluded. Numbers are cost ranks "
            "(ports used: colocation_footprints.png)\n"
            f"EI spans only {min(ei):.3f}-{max(ei):.3f} kg CO2/kg H2; alternatives locally refined from "
            "the optimum (costs are upper bounds)",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.5, color="#6b6a65")
    save_fig(fig, outdir / "colocation_cost_ei.png")
    _plot_footprints(rows, rank, payload, outdir)

    singles = [r for r in rows if r["kind"] in ("-1", "+1", "swap")]
    if not singles:
        return
    labels = [", ".join([f"−{_short(p)}" for p in r["removed"]] + [f"+{_short(p)}" for p in r["added"]])
              for r in singles]
    d_cost = [r["delivered_cost_aud_per_kg_h2"] - optimum["delivered_cost_aud_per_kg_h2"] for r in singles]
    d_ei = [r["delivered_ei_kg_co2_per_kg_h2"] - optimum["delivered_ei_kg_co2_per_kg_h2"] for r in singles]
    order = np.argsort(d_cost)
    colors = [_STYLE[_style_key(singles[i]["kind"])]["color"] for i in order]
    height = max(3.5, 0.24 * len(singles) + 1.2)
    fig, axes = plt.subplots(1, 2, figsize=(10, height), sharey=True, layout="constrained")
    y = np.arange(len(order))
    for ax, values, xlabel in ((axes[0], d_cost, "Δ delivered cost vs optimum [AUD/kg H2]"),
                               (axes[1], d_ei, "Δ emissions intensity vs optimum [kg CO2/kg H2]")):
        ax.barh(y, [values[i] for i in order], color=colors, height=0.72, edgecolor="white", linewidth=1)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set(xlabel=xlabel)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [labels[i] for i in order], fontsize=8)
    axes[0].invert_yaxis()
    fig.suptitle("Single-site changes to the optimum footprint (blue: removed; orange: added or swapped)",
                 fontsize=10, fontweight="bold")
    save_fig(fig, outdir / "colocation_single_site.png")


def main():
    parser = build_parser(__doc__, outdir=COLOCATION_DIR, replot=True)
    parser.add_argument("--optimum", type=Path, default=OPT_DIR / "optimum.json")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--budget", type=int, default=2500, help="Evaluations per footprint")
    parser.add_argument("--max-removed", type=int, default=3)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    path = args.outdir / "colocation.json"
    if args.replot:
        payload = json.loads(path.read_text())
    else:
        payload = compute(args.optimum, args.workers, args.budget, args.max_removed)
        path.write_text(json.dumps(payload, indent=2, allow_nan=False))
        print(f"Saved: {path}")
    if not args.no_plot:
        from scripts.common.cli import init_plotting
        init_plotting()
        plot(payload, args.outdir)


if __name__ == "__main__":
    main()
