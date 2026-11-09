"""Seasonal and annual average dispatch profiles for the MeOH supply chain.

Generates a three-panel figure:
  1. Full-year H2 storage level (hourly fill + 7-day rolling mean) and battery SOC
  2. Seasonal average 24-hour diurnal profiles (wind and solar power; PEM H2
     output and synthesis H2 demand)
  3. Monthly average generation and curtailment

Profiles are in UTC as delivered by renewables.ninja; AEST = UTC+10,
AWST = UTC+8, so solar peaks appear around hours 2-4 UTC.

Usage:
    python -m scripts.plots.dispatch_profiles
    python -m scripts.plots.dispatch_profiles --port Gladstone
    python -m scripts.plots.dispatch_profiles --outdir outputs
"""

import argparse
from pathlib import Path

import numpy as np

from scripts.common.cli import init_plotting

init_plotting()

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from constants import KG_PER_TONNE
from meoh_pathway.supply_chain import SupplyChain, SupplyChainResult
from meoh_pathway.renewables_dispatch import (
    load_wind_profile,
    load_solar_profile,
)
from meoh_pathway.parameters import PORT_SITES
from matplotlib.lines import Line2D

from scripts.common.plot_common import (BATTERY_COLOUR, CURTAILED_COLOUR, H2_STORE_COLOUR, PEM_COLOUR,
                                  SOLAR_COLOUR, THESIS_FONT_PT, THESIS_TITLE_PT, THESIS_WIDTH_IN,
                                  SYNTH_COLOUR, WIND_COLOUR, port_label,
                                  save_fig, tint)

# Non-leap year calendar month hour boundaries
_MONTH_HOURS = [
    0,
    744,
    1416,
    2160,
    2880,
    3624,
    4368,
    5112,
    5832,
    6576,
    7296,
    8016,
    8760,
]
_MONTH_NAMES = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]

# Southern hemisphere: DJF=Dec+Jan+Feb, MAM, JJA, SON
_MONTH_SEASON = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0]
_SEASON_LABELS = ["Summer (DJF)", "Autumn (MAM)", "Winter (JJA)", "Spring (SON)"]

# Standard-time UTC offsets per port, ignoring daylight saving. The data is
# hourly, so the +9.5 h ACST ports (SA, NT) round to +10.
_PORT_UTC_OFFSET = {
    "PortHastings": 10,  # AEST (VIC)
    "PortBonython": 10,  # ACST +9.5 -> round to 10 for hourly data
    "Gladstone": 10,  # AEST (QLD, no DST)
    "HayPoint": 10,  # AEST (QLD, no DST)
    "Ashburton": 8,  # AWST (WA, no DST)
    "Dampier": 8,  # AWST (WA, no DST)
    "PortHedland": 8,  # AWST (WA, no DST)
    "PortDarwin": 10,  # ACST +9.5 -> round to 10 for hourly data
    "PortAbbotPoint": 10,  # AEST (QLD, no DST)
    "Newcastle": 10,  # AEST (NSW)
}


def _hour_metadata():
    month = np.zeros(8760, dtype=int)
    for m in range(12):
        month[_MONTH_HOURS[m] : _MONTH_HOURS[m + 1]] = m
    season = np.array([_MONTH_SEASON[m] for m in month])
    hour_of_day = np.arange(8760) % 24
    return month, season, hour_of_day


def _periodic_rolling(x: np.ndarray, window: int) -> np.ndarray:
    """Centered rolling mean with periodic (wrap-around) boundaries.

    simulate_flex_dispatch converges to a periodic fixed point (year N end
    = year N+1 start), so wrapping the window around Jan 1 is physically
    correct. np.convolve(mode='same') would pad with zeros instead, which
    depresses the start-of-year value.
    """
    half = window // 2
    offsets = np.arange(-half, window - half)
    idx = np.arange(len(x)).reshape(-1, 1) + offsets
    return np.take(x, idx, mode="wrap").mean(axis=1)


def _periodic_rolling_max(x: np.ndarray, window: int) -> np.ndarray:
    """Centered rolling max with periodic boundaries (see _periodic_rolling)."""
    half = window // 2
    offsets = np.arange(-half, window - half)
    idx = np.arange(len(x)).reshape(-1, 1) + offsets
    return np.take(x, idx, mode="wrap").max(axis=1)


def _style_right_spine(ax_primary, ax_secondary):
    """Force the twinx()'s right spine to render as a clean solid black line.

    Matplotlib's twin_axes leaves the shared spine ambiguous: the primary
    axes' right spine sits under the twin's, and default styling can leave
    both invisible or faint. Hide the primary's right spine and explicitly
    style the twin's."""
    ax_primary.spines["right"].set_visible(False)
    s = ax_secondary.spines["right"]
    s.set_visible(True)
    s.set_color("black")
    s.set_linewidth(1.2)


def _plot_trace(port, trace, r):
    trace = dict(trace)
    hcap = r.site_h2_storage_kg.get(port, 0)
    trace["h2_soc_frac"] = trace["soc_h2_kg"] / hcap if hcap else np.zeros(8760)
    trace["battery_soc_mwh"] = trace["soc_e_mwh"]
    trace["wind_gen_mw"] = load_wind_profile(port) * r.site_wind_mw.get(port, 0)
    trace["solar_gen_mw"] = load_solar_profile(port) * r.site_solar_mw.get(port, 0)
    return trace


def _season_profile(series, mask, hod_local):
    """Mean of series at each local hour of day over the hours in mask."""
    return np.array([series[mask & (hod_local == h)].mean() for h in range(24)])


def plot_site_dispatch(
    port: str, trace: dict, r: SupplyChainResult, outdir: Path
) -> None:
    """Three-panel dispatch profile for one port of a multi-site result, at the
    thesis page width and type sizes: (a) storage state of charge over the
    year, (b) seasonal average daily profiles, one small panel per season,
    (c) monthly average generation and curtailment."""
    h2_storage_kg = r.site_h2_storage_kg.get(port, 0.0)
    battery_mwh = r.site_battery_mwh.get(port, 0.0)
    if r.site_h2_frac:
        site_frac = r.site_h2_frac.get(port, 0.0)
    else:
        total_energy_mw = sum(
            r.site_wind_mw.get(p, 0.0) * float(load_wind_profile(p).mean())
            + r.site_solar_mw.get(p, 0.0) * float(load_solar_profile(p).mean())
            for p in r.site_wind_mw
        )
        port_energy_mw = r.site_wind_mw.get(port, 0.0) * float(
            load_wind_profile(port).mean()
        ) + r.site_solar_mw.get(port, 0.0) * float(load_solar_profile(port).mean())
        site_frac = port_energy_mw / total_energy_mw if total_energy_mw > 0.0 else 0.0
    site_h2_daily_kg = r.fleet_pem_h2_tpa * KG_PER_TONNE * site_frac / 365.0
    storage_days = h2_storage_kg / site_h2_daily_kg if site_h2_daily_kg > 0 else 0.0
    batt_soc_pct = trace["battery_soc_mwh"] / battery_mwh * 100.0 if battery_mwh > 0.0 else None

    month, season, hod = _hour_metadata()
    fs = THESIS_FONT_PT
    fig = plt.figure(figsize=(THESIS_WIDTH_IN, 9.0), layout="constrained")
    fig.suptitle(f"{port_label(port)} dispatch profile ({site_frac:.0%} of fleet H2)",
                 fontsize=THESIS_TITLE_PT, fontweight="bold")
    sub_a, sub_b, sub_c = fig.subfigures(3, 1, height_ratios=[1, 1.55, 1])

    def _title(target, text):
        if hasattr(target, "set_title"):
            target.set_title(text, loc="left", fontsize=fs, fontweight="bold")
        else:
            target.suptitle(text, x=0.0, ha="left", fontsize=fs, fontweight="bold")

    def _axis_style(ax):
        ax.tick_params(labelsize=fs)
        ax.spines[["top"]].set_visible(False)

    # (a) H2 storage level, with the battery's daily SOC on the right axis.
    ax = sub_a.subplots()
    days = np.arange(8760) / 24.0
    soc_pct = trace["h2_soc_frac"] * 100.0
    ax.fill_between(days, soc_pct, color=tint(H2_STORE_COLOUR, 0.75), linewidth=0,
                    label="H2 storage level, hourly")
    ax.plot(days, _periodic_rolling(soc_pct, 168), color=H2_STORE_COLOUR, linewidth=1.5,
            label="H2 storage level, 7-day mean")
    ax.set_ylabel("H2 storage level (%)", fontsize=fs)
    ax.set_ylim(0, 105)
    ax.set_xlim(0, 365)
    ax.set_xticks([h / 24.0 for h in _MONTH_HOURS[:-1]], _MONTH_NAMES)
    ax.grid(axis="x", visible=False)
    _axis_style(ax)
    handles, labels = ax.get_legend_handles_labels()
    if batt_soc_pct is not None:
        ax2 = ax.twinx()
        ax2.plot(days, _periodic_rolling_max(batt_soc_pct, 24), color=BATTERY_COLOUR,
                 linewidth=0.6, linestyle=":", label="Battery SOC, daily max")
        ax2.plot(days, _periodic_rolling(batt_soc_pct, 24), color=BATTERY_COLOUR,
                 linewidth=0.6, label="Battery SOC, daily mean")
        ax2.set_ylabel("Battery SOC (%)", fontsize=fs)
        ax2.set_ylim(0, 105)
        ax2.grid(False)
        ax2.tick_params(labelsize=fs)
        _style_right_spine(ax, ax2)
        h2, l2 = ax2.get_legend_handles_labels()
        handles, labels = handles + h2, labels + l2
        _title(ax, f"(a) Storage levels: H2 storage {h2_storage_kg / 1e6:.0f} M kg "
                   f"({storage_days:.1f} days of demand), battery {battery_mwh:,.0f} MWh")
    else:
        ax.spines[["right"]].set_visible(False)
        _title(ax, f"(a) Storage levels: H2 storage {h2_storage_kg / 1e6:.0f} M kg "
                   f"({storage_days:.1f} days of demand), no battery")
    # Below the plot: the year is busy from edge to edge.
    # Two columns, H2 entries in the first and battery in the second: four
    # in one row run past the page width.
    sub_a.legend(handles, labels, loc="outside lower center", ncol=2,
                 fontsize=fs, frameon=False)

    # (b) Seasonal average daily profiles in local time, one panel per season,
    # wind and solar power on a shared left axis and H2 flows (PEM output,
    # synthesis demand) on a shared right axis.
    utc_off = _PORT_UTC_OFFSET.get(port, 10)
    hod_local = (hod + utc_off) % 24
    hours = np.arange(24)
    axes_b = sub_b.subplots(2, 2, sharex=True, sharey=True)
    _title(sub_b, f"(b) Seasonal average daily profiles (local time, UTC+{utc_off})")
    h2_flows = (("h2_produced_kg", "PEM H2 output (right axis)", PEM_COLOUR, "-"),
                ("demand_h2_kg", "Synthesis H2 demand (right axis)", SYNTH_COLOUR, "-"))
    peak = h2_peak = 0.0
    h2_axes = []
    for s_idx, (label, ax) in enumerate(zip(_SEASON_LABELS, axes_b.flat)):
        mask = season == s_idx
        for key, name, colour, width in (("wind_gen_mw", "Wind", WIND_COLOUR, 1.2),
                                         ("solar_gen_mw", "Solar", SOLAR_COLOUR, 1.2)):
            profile = _season_profile(trace[key], mask, hod_local) / 1e3
            peak = max(peak, profile.max())
            ax.plot(hours, profile, color=colour, linewidth=width, label=name)
        ax.set_title(f"({('i', 'ii', 'iii', 'iv')[s_idx]}) {label}", fontsize=fs)
        ax.set_xlim(0, 23)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(3))
        _axis_style(ax)
        ax.spines[["right"]].set_visible(False)
        axh = ax.twinx()
        h2_axes.append(axh)
        for key, name, colour, style in h2_flows:
            series = np.broadcast_to(np.asarray(trace[key], dtype=float), (8760,))
            profile = _season_profile(series, mask, hod_local)
            h2_peak = max(h2_peak, profile.max())
            axh.plot(hours, profile, color=colour, linewidth=1.2, linestyle=style, label=name)
        axh.grid(False)
        axh.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
        # Only the right column carries the H2 axis; the left column's would
        # sit between the panels unlabelled.
        if s_idx % 2 == 1:
            axh.tick_params(labelsize=fs)
            axh.set_ylabel("H2 flow (kg/h)", fontsize=fs)
            _style_right_spine(ax, axh)
        else:
            axh.tick_params(right=False, labelright=False)
            axh.spines[["right"]].set_visible(False)
        axh.spines[["top"]].set_visible(False)
    # Shared axes, set once every season is drawn.
    axes_b[0, 0].set_ylim(0, peak * 1.08)
    for axh in h2_axes:
        axh.set_ylim(0, h2_peak * 1.08)
    for ax in axes_b[:, 0]:
        ax.set_ylabel("Power (GW)", fontsize=fs)
    for ax in axes_b[1, :]:
        ax.set_xlabel("Hour of day", fontsize=fs)
    handles, labels = axes_b[0, 0].get_legend_handles_labels()
    for _, name, colour, style in h2_flows:
        handles.append(Line2D([0], [0], color=colour, linewidth=1.2, linestyle=style))
        labels.append(name)
    sub_b.legend(handles, labels, loc="outside lower center", ncol=len(handles),
                 fontsize=fs, frameon=False)

    # (c) Monthly averages, outlined bars with a gap so outlines do not touch.
    ax = sub_c.subplots()
    s, e = _MONTH_HOURS[:-1], _MONTH_HOURS[1:]
    x = np.arange(12)
    w = 0.26
    for i, (key, name, colour) in enumerate((("wind_gen_mw", "Wind", WIND_COLOUR),
                                             ("solar_gen_mw", "Solar", SOLAR_COLOUR),
                                             ("curtailed_mw", "Curtailed", CURTAILED_COLOUR))):
        means = np.array([trace[key][a:b].mean() for a, b in zip(s, e)]) / 1e3
        ax.bar(x + (i - 1) * w, means, w - 0.04, color=tint(colour), edgecolor=colour,
               linewidth=[1.0 if m > 0 else 0 for m in means], label=name)
    ax.set_xticks(x, _MONTH_NAMES)
    ax.set_ylabel("Average power (GW)", fontsize=fs)
    ax.grid(axis="x", visible=False)
    _axis_style(ax)
    ax.spines[["right"]].set_visible(False)
    _title(ax, "(c) Monthly average generation and curtailment")
    sub_c.legend(loc="outside lower center", ncol=3, fontsize=fs, frameon=False)

    outpath = outdir / f"dispatch_{port}.png"
    save_fig(fig, outpath, tight=False)


def plot_optimised_dispatch(r: SupplyChainResult, outdir: Path, traces: dict) -> None:
    """Plot the integrator traces for the verified optimised result."""
    from meoh_pathway.validation import validate_traces
    validate_traces(r, traces)
    outdir.mkdir(parents=True, exist_ok=True)
    if not any(
        r.site_wind_mw.get(p, 0.0) + r.site_solar_mw.get(p, 0.0) > 0.0
        for p in PORT_SITES
    ):
        print("No active sites in result.")
        return
    for port in PORT_SITES:
        if port in r.idle_site_renewable_mwh:
            plot_unused_site(port, outdir, retained_assets=True)
            continue
        if r.site_wind_mw.get(port, 0.0) + r.site_solar_mw.get(port, 0.0) <= 0.0:
            print(f"  [{port}] not used -- placeholder plot")
            plot_unused_site(port, outdir)
            continue
        print(f"  [{port}] running trace ...", end=" ", flush=True)
        try:
            trace = _plot_trace(port, traces[port], r)
            plot_site_dispatch(port, trace, r, outdir)
        except ValueError as exc:
            raise ValueError(f"Cannot reproduce dispatch for {port}: {exc}") from exc


def plot_unused_site(port: str, outdir: Path, retained_assets=False) -> None:
    """Placeholder plot for a port the optimiser left with zero capacity."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        f"{port}\n" + ("Zero production; fixed assets retained and costed"
                       if retained_assets else "Not used in this deployment"),
        ha="center",
        va="center",
        fontsize=16,
        color="#666",
    )
    outpath = outdir / f"dispatch_{port}.png"
    save_fig(fig, outpath, dpi=120)


def main():
    from scripts.common.outputs import TYPICAL_DIR
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--port", default=None, help="Single port to plot (default: all ports)"
    )
    p.add_argument(
        "--outdir",
        default=str(TYPICAL_DIR / "dispatch_profiles"),
        help="Output directory (default: outputs/typical_build/dispatch_profiles/)",
    )
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sc = SupplyChain()
    r = sc.run()
    traces = sc.dispatch_traces()
    ports = [args.port] if args.port else list(PORT_SITES)
    for port in ports:
        if port not in PORT_SITES:
            raise ValueError(f"Unknown port: {port}")
        if port in traces:
            plot_site_dispatch(port, _plot_trace(port, traces[port], r), r, outdir)
        else:
            plot_unused_site(port, outdir, retained_assets=port in r.idle_site_renewable_mwh)


if __name__ == "__main__":
    main()
