"""Shared graphviz and matplotlib styling, and output paths, for the plot scripts."""

import io
import re
import textwrap
import time
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Wedge
import numpy as np
import seaborn as sns

from scripts.common.outputs import OUTPUT_DIR, TYPICAL_DIR, OPT_DIR

DEFAULT_DIR = TYPICAL_DIR


def ensure_output_dirs():
    """Create the top-level output directories. Called from init_plotting(),
    not at import time -- importing this module must not touch the filesystem."""
    for d in (OUTPUT_DIR, DEFAULT_DIR, OPT_DIR):
        d.mkdir(parents=True, exist_ok=True)


# House matplotlib/seaborn theme, applied by every plotting script via apply_theme().
# Dotted axis grid, the convention across the repo's figures.
THESIS_RC = {
    "figure.dpi": 140,
    "savefig.dpi": 400,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.linestyle": ":",
    "grid.alpha": 0.5,
}


def apply_theme():
    """Set the shared whitegrid theme. Call after matplotlib.use('Agg')."""
    sns.set_theme(style="whitegrid", rc=THESIS_RC)


# Outlined fills (cost pie wedges, pathway comparison bars) are drawn this far
# toward white, with a 1.5 pt edge in the full colour like the flow-diagram
# nodes: lighter than a solid fill, well short of the flow-diagram fills (~0.86).
FILL_TINT = 0.30
INK = "#2b2b29"  # dark label ink, legible on the tinted fills
# Thesis figure width. The cost pies and pathway comparison are drawn this wide
# so their title is THESIS_TITLE_PT and other text THESIS_FONT_PT when placed at
# 17 cm on the page.
THESIS_WIDTH_IN = 17.0 / 2.54
THESIS_TITLE_PT = 12
THESIS_FONT_PT = 8


def tint(colour, amount=FILL_TINT):
    """colour mixed amount of the way to white, alpha kept."""
    rgba = np.array(mcolors.to_rgba(colour), dtype=float)
    rgba[:3] += (1 - rgba[:3]) * amount
    return tuple(rgba)


def save_fig(fig, path, dpi=None, tight=True, transparent=False):
    """Save (default: the theme's savefig.dpi) and close, replacing the per-script
    _save helpers. Pass dpi= for a figure that needs a different resolution,
    tight=False to keep the figure size exactly, for figures sized to the page,
    and transparent=True to drop the figure and axes backgrounds (slides)."""
    path = Path(path)
    kwargs = {"bbox_inches": "tight" if tight else None, "format": path.suffix.lstrip(".") or "png",
              "transparent": transparent}
    if dpi is not None:
        kwargs["dpi"] = dpi
    buf = io.BytesIO()
    fig.savefig(buf, **kwargs)
    plt.close(fig)
    # On Windows an image preview, thumbnailer or indexer can briefly hold the old
    # file mapped, and overwriting it then fails with EINVAL or EACCES. The lock
    # clears within seconds, so retry rather than abort a long run over one plot.
    for attempt in range(6):
        try:
            path.write_bytes(buf.getvalue())
            break
        except OSError as exc:
            if attempt == 5:
                print(f"WARNING: could not write {path} ({exc}); close any program "
                      "showing it and regenerate the plot")
                return
            time.sleep(2)
    print(f"Saved: {path}")


# One accent for single-series figures and a small ordered palette for multi-series
# ones, replacing the four greens / three blues that had accumulated one per script.
ACCENT = "#2a78d6"
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#9a9992")
# Marks that carry no identity (context points, unused slots, "other") take the
# neutral grey, never a hue; NEUTRAL_LIGHT for the faintest background marks.
# Reference and zero lines stay black.
NEUTRAL = SERIES_COLORS[3]
NEUTRAL_LIGHT = "#d6d5d0"
# Low/high bar colours for signed quantities (e.g. a tornado of correlations).
DIVERGING = ("#4a7bc8", "#c84a4a")


# Delivery-margin builds: greens darkening with the margin. 0% is the e-MeOH
# green (FLOW_COLORS["h2"]) and 10% the proposed-build green of the Monte Carlo
# (COST_COMPONENTS["pem_capital"]); 5% sits between them, 20% darker still.
MARGIN_COLOURS = {0: "#2ca02c", 5: "#16842d", 10: "#01672d", 20: "#00441b"}


def margin_colour(pct):
    """MARGIN_COLOURS green of a delivery margin in percent; other margins take the nearest."""
    return MARGIN_COLOURS[min(MARGIN_COLOURS, key=lambda m: abs(m - pct))]


def port_label(port):
    """Space out a camel-case port name (e.g. PortAbbotPoint) for tick labels."""
    return re.sub(r"(?<!^)(?=[A-Z])", " ", port)


FLOW_COLORS = {
    "h2": "#2ca02c",
    "meoh": "#1f77b4",
    "co2": "#ff7f0e",
    "vent": "#d62728",
    "energy": "#c8a604",
    "store": "#7f4cb8",
    "util": "#7f7f7f",
    # Methanol reformer (cracker): a darker H2 green, as its cost-pie wedge.
    "reformer": "#17661c",
}

# Per-site build and dispatch figures: one hue per quantity, wind light blue,
# solar yellow, and the flow diagram's colours for the rest.
WIND_COLOUR = "#4a90d9"
SOLAR_COLOUR = "#f2c200"
PEM_COLOUR = FLOW_COLORS["h2"]
BATTERY_COLOUR = FLOW_COLORS["store"]
H2_STORE_COLOUR = FLOW_COLORS["h2"]
CURTAILED_COLOUR = FLOW_COLORS["vent"]
# Navy: the flow diagram's mid methanol blue reads as the light-blue wind line.
SYNTH_COLOUR = "#0b3a7e"

FLOW_FILLS = {
    "h2": "#e7f3e3",
    "meoh": "#e0ecf5",
    "co2": "#ffe7d1",
    "vent": "#fcdcdc",
    "energy": "#faf4da",
    "store": "#ece1f3",
    "util": "#ededed",
    "reformer": "#d5e6d6",
}

FLOW_LABELS = {
    "energy": "Electricity",
    "h2": "Hydrogen",
    "meoh": "Methanol",
    "co2": "CO2",
    "store": "Storage",
    "vent": "Vented CO2",
    "util": "Water / fuel",
}


def format_html_node(title, lines, color):
    rows = f'<TR><TD><B><FONT COLOR="{color}">{title}</FONT></B></TD></TR>'
    for line in lines:
        rows += f"<TR><TD>{line}</TD></TR>"
    return f'<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="2">{rows}</TABLE>>'


def format_legend(keys=None):
    rows = '<TR><TD COLSPAN="2" BORDER="0"><B>Legend</B></TD></TR>'
    for key in keys or FLOW_LABELS:
        label = FLOW_LABELS[key]
        rows += (
            f"<TR>"
            f'<TD BGCOLOR="{FLOW_FILLS[key]}" COLOR="{FLOW_COLORS[key]}" '
            f'WIDTH="24" HEIGHT="18"></TD>'
            f'<TD BORDER="0" ALIGN="LEFT">  {label}</TD></TR>'
        )
    return (
        f'<<TABLE BORDER="1" CELLBORDER="1" CELLSPACING="4" '
        f'CELLPADDING="2">{rows}</TABLE>>'
    )


# Delivered-cost component key -> (legend label, colour). The Monte Carlo
# cost-component figure colours each component on its own; the pies group them
# into COST_STAGES. Hue follows the stage (PEM green, electricity gold, storage
# purple, methanol blue, CO2 orange, cracker teal, LNG gold, ATR green) and the
# single-component stages take their stage colour. Within a stage the shades
# alternate dark/light so neighbouring bars stay distinguishable.
COST_COMPONENTS = {
    # LNG and ATR: the delivered LNG price's three parts, each A$/GJ times the
    # ATR's feed GJ/kg H2, and the ATR's production cost ex feed and ASU power.
    "lng_feed_gas": ("Feed gas", "#ce9708"),
    "lng_liquefaction": ("Liquefaction (AU)", "#f0c43c"),
    "lng_shipping": ("LNG shipping AU to JP", "#a07505"),
    "atr_production": ("ATR capital + O&M (ex feed)", "#1f7a1f"),
    "atr_electricity": ("ASU electricity (JP)", "#5cb85c"),
    "pem_capital": ("PEM capital", "#01672d"),
    "pem_om": ("PEM O&M", "#329b4b"),
    "pem_stack": ("PEM stack replacement", "#7ecb6c"),
    "pem_water": ("PEM water", "#3e6822"),
    "renewables": ("Renewables + transmission", "#e8c000"),
    "battery": ("Battery", "#643b9a"),
    "storage": ("H2 storage", "#ae8cd5"),
    "transport": ("H2 pipelines + compressors", "#b75131"),
    "synth": ("MeOH synthesis", "#17559b"),
    "shipping": ("MeOH + CO2 carriers", "#70b6e7"),
    "terminal": ("AU + JP methanol storage", "#3585c9"),
    "co2_capture": ("CO2 liquefaction + storage", "#c14804"),
    "co2_makeup": ("CO2 make-up (DAC)", "#f09638"),
    "cracker": ("MeOH cracker", "#01aeae"),
    # Blue-H2 base case: the CO2 chain in CO2 orange, as separate keys from the
    # e-MeOH co2_* rows so each pathway's legend reads on its own.
    "ccs_capture": ("CO2 capture at ATR (JP)", "#9d3106"),
    "ccs_liquefaction": ("CO2 liquefaction + storage (JP)", "#ee9b58"),
    "ccs_shipping": ("LCO2 shipping JP to AU", "#c55204"),
    "ccs_storage": ("CO2 geological storage (AU)", "#dda151"),
}

# Pie wedges: each stage's components summed, in process order clockwise from
# 12 o'clock. Each stage takes the FLOW_COLORS colour of its node in the flow
# diagrams (LNG and ATR as in the grey base-case diagram), tinted FILL_TINT
# toward white so the share labels are dark ink, not white. The e-MeOH pie has
# three hydrogen stages, so they take shades of the H2 green: the cracker, next
# to PEM at 12 o'clock, darker, and the pipelines lighter.
COST_STAGES = (
    ("PEM electrolysis", FLOW_COLORS["h2"], ("pem_capital", "pem_om", "pem_stack", "pem_water")),
    ("Electricity", FLOW_COLORS["energy"], ("renewables",)),
    ("Battery + H2 storage", FLOW_COLORS["store"], ("battery", "storage")),
    ("H2 pipelines + compressors", "#74c476", ("transport",)),
    ("Methanol synthesis, shipping + storage", FLOW_COLORS["meoh"], ("synth", "shipping", "terminal")),
    ("CO2 recycle + make-up", FLOW_COLORS["co2"], ("co2_capture", "co2_makeup")),
    ("MeOH cracker", FLOW_COLORS["reformer"], ("cracker",)),
    ("Delivered LNG feed", FLOW_COLORS["energy"],
     ("lng_feed_gas", "lng_liquefaction", "lng_shipping")),
    ("ATR production", FLOW_COLORS["h2"], ("atr_production", "atr_electricity")),
    ("CO2 capture, return + storage", FLOW_COLORS["co2"],
     ("ccs_capture", "ccs_liquefaction", "ccs_shipping", "ccs_storage")),
)

# Slide pie: the e-MeOH stages only, H2 pipelines folded into electrolysis and
# the cracker named as the reformer, matching the slide flow diagram.
SLIDE_COST_STAGES = (
    ("PEM electrolysis + H2 pipelines", FLOW_COLORS["h2"],
     ("pem_capital", "pem_om", "pem_stack", "pem_water", "transport")),
    ("Renewables + transmission", FLOW_COLORS["energy"], ("renewables",)),
    ("Battery + H2 storage", FLOW_COLORS["store"], ("battery", "storage")),
    ("MeOH synthesis, shipping + storage", FLOW_COLORS["meoh"], ("synth", "shipping", "terminal")),
    ("CO2 recycle + make-up", FLOW_COLORS["co2"], ("co2_capture", "co2_makeup")),
    ("Reformer", FLOW_COLORS["reformer"], ("cracker",)),
)
# The LNG benchmarks' slide pie: the COST_STAGES LNG stages.
# ATR takes the reformer green of the slide flow diagrams' H2 Reforming node.
LNG_SLIDE_COST_STAGES = tuple(
    (label, FLOW_COLORS["reformer"] if label == "ATR production" else colour, keys)
    for label, colour, keys in COST_STAGES
    if label in ("Delivered LNG feed", "ATR production", "CO2 capture, return + storage"))
SLIDE_FONT_PT = 16

# The two LNG benchmarks: grey H2 is the reference case (no capture), blue H2
# (ATR + CCS, CO2 returned to Australia) is the base case e-MeOH is judged against.
REFERENCE_CASE = ("Grey H2 (LNG)", "reference case", "#868686")
BASE_CASE = ("Blue H2 (LNG + CCS)", "base case", "#0153a4")


def lng_case_costs(grey, lng, blue=None):
    """Cost components [AUD/kg H2] of the reference (grey) or base (blue) case.

    lng (LngSupplyParameters) splits the feed cost into the delivered LNG
    price's parts, each scaled by the ATR's feed GJ/kg H2.
    """
    feed = grey.feed_gj_per_kg_h2
    costs = {"lng_feed_gas": feed * lng.lng_feed_gas_cost_aud_per_gj,
             "lng_liquefaction": feed * lng.lng_liquefaction_cost_aud_per_gj,
             "lng_shipping": feed * lng.lng_shipping_cost_aud_per_gj,
             "atr_production": grey.prod_cost_ex_feed_aud_per_kg_h2,
             "atr_electricity": grey.atr_electricity_cost_aud_per_kg_h2}
    if blue is not None:
        costs.update(ccs_capture=blue.capture_cost_aud_per_kg_h2,
                     ccs_liquefaction=blue.co2_liquefaction_cost_aud_per_kg_h2,
                     ccs_shipping=blue.co2_shipping_cost_aud_per_kg_h2,
                     ccs_storage=blue.co2_storage_cost_aud_per_kg_h2)
    return costs


def plot_cost_pie(costs, title, path, label_share=0.04):
    """Delivered-cost pie, one wedge per COST_STAGES stage, clockwise from 12 o'clock.

    Every wedge carries its share: inside when at least label_share, else just
    outside on a short leader line. The legend, right of the pie, names every
    stage with its AUD/kg H2 value and share, and lists the components behind a
    multi-component stage, so the detail is still on the figure. Drawn at
    THESIS_WIDTH_IN and saved uncropped, so the title is THESIS_TITLE_PT and other
    text THESIS_FONT_PT when the image is placed 17 cm wide.
    """
    staged = {k for _, _, keys in COST_STAGES for k in keys}
    unknown = sorted(k for k, v in costs.items() if v > 0 and k not in staged)
    extra = iter(sns.color_palette("husl", max(1, len(unknown))))
    stages = [(label, colour, keys) for label, colour, keys in COST_STAGES]
    stages += [(k.replace("_", " ").capitalize(), next(extra), (k,)) for k in unknown]
    values, colours, outlines, labels = [], [], [], []
    for label, colour, keys in stages:
        parts = [(k, costs[k]) for k in keys if costs.get(k, 0) > 0]
        if not parts:
            continue
        values.append(sum(v for _, v in parts))
        colours.append(tint(colour))
        outlines.append(colour)
        # A one-component stage is just that component, under its own name.
        labels.append((COST_COMPONENTS.get(parts[0][0], (label,))[0] if len(parts) == 1 else label,
                       parts if len(parts) > 1 else []))
    total = sum(values)
    rows = sum(1 + len(parts) for _, parts in labels)

    # Tall enough for the pie, or for the legend's one row per stage and component.
    fig, ax = plt.subplots(figsize=(THESIS_WIDTH_IN, max(3.4, 1.1 + 0.155 * rows)),
                           layout="constrained")
    _outlined_pie(ax, values, colours, outlines, label_share, THESIS_FONT_PT)
    # Components get their own blank-swatch rows, so each colour sits level with
    # its stage name rather than centred on a multi-line block.
    handles, entries = [], []
    for colour, outline, (label, parts), value in zip(colours, outlines, labels, values):
        handles.append(Patch(facecolor=colour, edgecolor=outline, linewidth=1.5))
        entries.append(f"{label}  A${value:.2f} ({value/total:.1%})")
        for k, v in parts:
            handles.append(Patch(alpha=0))
            entries.append(f"    {COST_COMPONENTS[k][0]}  A${v:.2f}")
    fig.legend(handles, entries, loc="outside right center", frameon=False,
               fontsize=THESIS_FONT_PT, labelspacing=0.35)
    # A title too long for one line breaks after its colon, else wherever it fits.
    if len(title) > 80:
        title = "\n".join(textwrap.fill(t, 80) for t in title.replace(": ", ":\n", 1).split("\n"))
    # Over the whole figure: an axes title is centred on the pie alone and runs
    # off the left edge.
    fig.suptitle(f"{title}\nDelivered cost A${total:.2f} / kg H2",
                 fontsize=THESIS_TITLE_PT, fontweight="bold")
    # Room around the pie for the outside labels, which layout does not reserve.
    ax.set(xlim=(-1.4, 1.4), ylim=(-1.15, 1.15), aspect="equal")
    save_fig(fig, path, tight=False)


def plot_cost_pie_slide(costs, path, label_share=0.04, stages=SLIDE_COST_STAGES, thesis=False):
    """Slide version of plot_cost_pie: one wedge per stages stage, no component
    rows, all text SLIDE_FONT_PT when placed at its drawn size (12 x 5.6 in).

    thesis=True draws the thesis copy: an opaque white background, no shares
    in the legend (the wedges carry them) and the delivered cost moved from the
    legend title into a bold "(b) Cost breakdown" heading top left."""
    staged = {k for _, _, keys in stages for k in keys}
    unknown = sorted(k for k, v in costs.items() if v > 0 and k not in staged)
    if unknown:
        raise ValueError(f"No slide cost stage for {unknown}")
    values, colours, outlines, labels = [], [], [], []
    for label, colour, keys in stages:
        value = sum(costs.get(k, 0) for k in keys)
        if value > 0:
            values.append(value)
            colours.append(tint(colour))
            outlines.append(colour)
            labels.append(label)
    total = sum(values)

    fig, ax = plt.subplots(figsize=(12, 5.6), layout="constrained")
    # Wider edges than the thesis pie: the pie is drawn about twice as large.
    _outlined_pie(ax, values, colours, outlines, label_share, SLIDE_FONT_PT,
                  edge_pt=2.5, gap_pt=3)
    handles = [Patch(facecolor=c, edgecolor=o, linewidth=2) for c, o in zip(colours, outlines)]
    entries = [f"{label}  A${v:.2f}" + ("" if thesis else f" ({v/total:.0%})")
               for label, v in zip(labels, values)]
    # The total heads the legend rather than the figure.
    fig.legend(handles, entries, loc="outside right center", frameon=False,
               fontsize=SLIDE_FONT_PT, labelspacing=0.6, handleheight=1.2,
               title=None if thesis else f"Delivered cost A${total:.2f} / kg H2", alignment="left",
               title_fontproperties=dict(size=SLIDE_FONT_PT + 2, weight="bold"))
    ax.set(xlim=(-1.4, 1.4), ylim=(-1.15, 1.15), aspect="equal")
    if thesis:
        fig.text(0.01, 0.99, f"(b) Cost breakdown, A${total:.2f} / kg H2 delivered", ha="left",
                 va="top", fontsize=SLIDE_FONT_PT + 2, fontweight="bold", color=INK)
    save_fig(fig, path, tight=False, transparent=not thesis)


# Components plot_flow_cost lists as one legend row, to keep its legend short.
FLOW_COST_MERGED = (("PEM electrolysis", ("pem_capital", "pem_om", "pem_stack", "pem_water")),)


def plot_flow_cost(flow_png, costs, path, stages=SLIDE_COST_STAGES, beside=True,
                   flow_font_pt=16, flow_dpi=300, flow_share=0.70):
    """Thesis-width figure: "(a) Flow chart", a rendered slide flow diagram,
    with "(b) Cost breakdown", its cost pie, either beside it with the legend
    across the foot of both, one column per stage, or with beside=False
    centred below it with the legend beside the pie. The legend lists the
    components behind a multi-component stage (FLOW_COST_MERGED ones as one
    row). The pie and legend text is set at the size the flow diagram's
    flow_font_pt text prints at once scaled to the page."""
    staged = {k for _, _, keys in stages for k in keys}
    unknown = sorted(k for k, v in costs.items() if v > 0 and k not in staged)
    if unknown:
        raise ValueError(f"No slide cost stage for {unknown}")
    merged = {k: label for label, keys in FLOW_COST_MERGED for k in keys}
    rows = []
    for label, colour, keys in stages:
        parts = {}
        for k in keys:
            if costs.get(k, 0) > 0:
                name = merged.get(k) or COST_COMPONENTS[k][0]
                parts[name] = parts.get(name, 0) + costs[k]
        if parts:
            rows.append((label, colour, sum(parts.values()),
                         list(parts.items()) if len(parts) > 1 else []))
    total = sum(v for _, _, v, _ in rows)

    img = plt.imread(flow_png)
    native_w, native_h = img.shape[1] / flow_dpi, img.shape[0] / flow_dpi
    W = THESIS_WIDTH_IN
    fw = W * (flow_share if beside else 1.0)
    fh = fw * native_h / native_w
    pt = flow_font_pt * fw / native_w
    head_pt = pt * 1.15
    hh = 1.5 * head_pt / 72  # one heading line, in
    colours = [tint(c) for _, c, _, _ in rows]
    outlines = [c for _, c, _, _ in rows]

    # Components get blank-swatch rows, so each colour sits level with its
    # stage. Across the foot, each stage is padded to the longest so that,
    # filled column by column, every stage starts a column of its own.
    def legend(fig, **kw):
        blocks = []
        for (label, _, value, parts), colour, outline in zip(rows, colours, outlines):
            block = [(Patch(facecolor=colour, edgecolor=outline, linewidth=1),
                      f"{label}  A${value:.2f}")]
            block += [(Patch(alpha=0), f"    {name}  A${v:.2f}") for name, v in parts]
            blocks.append(block)
        if beside:
            n = max(len(b) for b in blocks)
            blocks = [b + [(Patch(alpha=0), "")] * (n - len(b)) for b in blocks]
            kw["ncol"], kw["columnspacing"] = len(blocks), 2.5
        items = [item for block in blocks for item in block]
        return fig.legend(*zip(*items), frameon=False, fontsize=pt, labelspacing=0.3,
                          handlelength=1.6, borderaxespad=0, borderpad=0, **kw)

    # Measured sizes set the layout: the legend, and whether the bold cost
    # heading fits the cost column on one line.
    heading = f"(b) Cost breakdown - Delivered A${total:.2f} / kg H2"
    probe = plt.figure()
    probe.canvas.draw()
    renderer = probe.canvas.get_renderer()
    box = legend(probe).get_window_extent(renderer)
    lw, lh = box.width / probe.dpi, box.height / probe.dpi
    head_w = probe.text(0, 0, heading, fontsize=head_pt, fontweight="bold")
    head_w = head_w.get_window_extent(renderer).width / probe.dpi
    plt.close(probe)

    if beside:
        x0 = fw + 0.1  # the cost column
        if head_w > W - x0 - 0.02:
            heading = heading.replace(" - ", " -\n")
        pie_d = (W - x0) / 1.4
        pie_x, pie_top = x0 + (W - x0) / 2, (heading.count("\n") + 1) * hh + 0.1
        top_h = max(hh + fh, pie_top + pie_d)
        H = top_h + 0.12 + lh + 0.03
    else:
        x0, y1 = 0, hh + fh + 0.15
        # As large a pie as fits beside the legend, the pair centred on the
        # page; 0.3 in to the left of the disc holds its outside share labels.
        pie_d = min(2.6, W - lw - 0.9)
        left = (W - (0.3 + pie_d + 0.25 + lw)) / 2
        pie_x, pie_y = left + 0.3 + pie_d / 2, y1 + hh + 0.1
        block = max(pie_d, lh)
        H = pie_y + block + 0.05
        pie_top = pie_y + (block - pie_d) / 2

    fig = plt.figure(figsize=(W, H))

    def axes(x, y, w, h):  # inches from the top left
        return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])

    def text(x, y, s):
        fig.text(x / W, 1 - y / H, s, ha="left", va="top", color=INK, fontsize=head_pt,
                 fontweight="bold", linespacing=1.3)

    text(0.02, 0.02, "(a) Flow chart")
    ax = axes(0, hh, fw, fh)
    ax.imshow(img, interpolation="lanczos")
    ax.axis("off")

    text(x0 + 0.02, 0.02 if beside else y1, heading)
    # The axes is 1.4 x the disc wide, leaving room for outside share labels.
    pax = axes(pie_x - 0.7 * pie_d, pie_top, 1.4 * pie_d, pie_d)
    _outlined_pie(pax, [v for _, _, v, _ in rows], colours, outlines, 0.06, pt,
                  edge_pt=1.0, gap_pt=1)
    pax.set(xlim=(-1.4, 1.4), ylim=(-1.0, 1.0), aspect="equal")

    if beside:
        legend(fig, loc="upper center", bbox_to_anchor=(0.5, 1 - (top_h + 0.12) / H))
    else:
        legend(fig, loc="center left", bbox_to_anchor=((pie_x + pie_d / 2 + 0.25) / W,
                                                       1 - (pie_top + pie_d / 2) / H))
    save_fig(fig, path, dpi=600, tight=False)


def _outlined_pie(ax, values, colours, outlines, label_share, fontsize, edge_pt=1.5, gap_pt=1):
    """Pie clockwise from 12 o'clock, outlined wedges (edge_pt wide, parted by
    a gap_pt white line), share labels inside or on a leader line outside when
    below label_share."""
    wedges, _ = ax.pie(values, colors=colours, startangle=90, counterclock=False,
                       wedgeprops=dict(edgecolor="none"))
    # Outlined like the flow-diagram nodes: tinted fill, 1.5 pt edge in the full
    # colour. Each edge is drawn 3 pt wide and clipped to its own wedge, so only
    # the inner 1.5 pt shows and neighbours never paint over one another; a thin
    # white line on top then parts them along the shared radius. Round joins: a
    # mitred edge on a thin wedge spikes past the centre into the wedge opposite.
    for wedge, outline in zip(wedges, outlines):
        edge = Wedge(wedge.center, wedge.r, wedge.theta1, wedge.theta2, facecolor="none",
                     edgecolor=outline, linewidth=2 * edge_pt, joinstyle="round",
                     zorder=wedge.zorder + 1)
        ax.add_patch(edge)
        edge.set_clip_path(wedge)
        ax.add_patch(Wedge(wedge.center, wedge.r, wedge.theta1, wedge.theta2, facecolor="none",
                           edgecolor="white", linewidth=gap_pt, joinstyle="round",
                           zorder=wedge.zorder + 2))
    total = sum(values)
    for wedge, value in zip(wedges, values):
        angle = np.deg2rad((wedge.theta1 + wedge.theta2) / 2)
        x, y = np.cos(angle), np.sin(angle)
        share = f"{value/total:.0%}" if value / total >= 0.01 else "<1%"
        if value / total >= label_share:
            ax.text(0.68*x, 0.68*y, share, ha="center", va="center",
                    fontsize=fontsize, fontweight="bold", color=INK)
        else:
            ax.plot([1.0*x, 1.1*x], [1.0*y, 1.1*y], color=INK, linewidth=0.8)
            ax.text(1.14*x, 1.14*y, share, ha="left" if x >= 0 else "right", va="center",
                    fontsize=fontsize, fontweight="bold", color=INK)
