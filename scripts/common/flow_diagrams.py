"""Graphviz flow diagrams for the grey reference case and the MeOH pathway.

Presentation code, kept out of the runner scripts so it can be
imported without pulling in a CLI entry point.

Both diagrams are flowsheets laid out by hand: the product chain runs along one
straight line, each unit's vent sits above it and its utilities below it.
Positions are pinned and neato only routes the edges, because dot reorders
nodes within a column to cut crossings and the layout then changes from build
to build.
"""

from pathlib import Path

import numpy as np

from meoh_pathway.parameters import PORT_SITES
from constants import DAYS_PER_YEAR, M_H2O, M_MEOH

# Thesis page vs a 16:9 slide: the slide version is drawn at 16 pt so it can
# be placed at 100% scale, so it needs tighter gaps to fit 13.3 in across.
_THESIS = dict(font="14", edge_font="12", title_font="18", col_gap=1.7, row_gap=0.45)
_SLIDE = dict(font="16", edge_font="16", title_font="18", col_gap=0.55, row_gap=0.35,
              margin="0.1,0.05")


def _node(name, flow, title, lines, **kwargs):
    return (name, flow, title, lines, kwargs)


def _vent(name, mtpa, source):
    return _node(name, "vent", "Vented CO2", [f"{mtpa:,.2f} Mtpa {source}"])


def _column(main=None, above=(), below=(), mirror=None, lift=False):
    """One flowsheet column; `above` and `below` are listed top to bottom.
    `mirror` names one node above and one below to set equidistant from the
    main line. `lift=True` stacks a column with no main node from the top of
    the tallest main node rather than from the main line; a number instead
    widens the gap between the main node and the first node above it (in)."""
    return (list(above), main, list(below), mirror, lift)


def _graph(dpi, style=_THESIS):
    import graphviz

    g = graphviz.Digraph(format="png", engine="neato")
    g.attr(splines="spline", fontname="Helvetica", dpi=dpi, esep="+4")
    g.attr(
        "node",
        shape="box",
        style="rounded,filled",
        fontname="Helvetica",
        fontsize=style["font"],
        penwidth="1.5",
    )
    if "margin" in style:
        g.attr("node", margin=style["margin"])
    g.attr("edge", fontname="Helvetica", fontsize=style["edge_font"], penwidth="1.5")
    return g


def _add_nodes(g, columns, extras, pos=None):
    from scripts.common.plot_common import FLOW_COLORS, FLOW_FILLS, format_html_node

    def at(name):
        return {"pos": "%.3f,%.3f!" % pos[name]} if pos else {}

    for above, main, below, *_ in columns:
        for name, flow, title, lines, kwargs in [*above, *([main] if main else []), *below]:
            # An `html` kwarg replaces the standard title-and-lines label.
            kwargs = dict(kwargs)
            kwargs.pop("span", None)
            kwargs.pop("row", None)
            kwargs.pop("drop", None)
            kwargs.pop("dx", None)
            g.node(
                name,
                kwargs.pop("html", None) or format_html_node(title, lines, FLOW_COLORS[flow]),
                color=FLOW_COLORS[flow],
                fillcolor=FLOW_FILLS[flow],
                **{"style": "rounded,filled", **kwargs, **at(name)},
            )
    for name, label in extras:
        g.node(name, label, shape="plaintext", style="solid", **at(name))


def _node_sizes(columns, extras, style):
    g = _graph("72", style)
    _add_nodes(g, columns, extras)
    sizes = {}
    for row in g.pipe(format="plain", engine="neato").decode().splitlines():
        parts = row.split()
        if parts and parts[0] == "node":
            sizes[parts[1].strip('"')] = (float(parts[4]), float(parts[5]))
    return sizes


def _positions(columns, extras, sizes, style):
    """Columns left to right, main nodes on y = 0. `extras` go under the last
    column (the legend) and a title node, if any, above the whole diagram."""
    col_gap, row_gap = style["col_gap"], style["row_gap"]
    pos = {}
    x = 0.0
    top = 0.0
    main_half = max((sizes[c[1][0]][1] / 2 for c in columns if c[1]), default=0.0)
    for i, (above, main, below, mirror, lift) in enumerate(columns):
        above = [n[0] for n in above]
        below = [n[0] for n in below]
        if i == len(columns) - 1:
            below += [name for name, _ in extras if name != "Title"]
        width = max(sizes[n][0] for n in [*above, *below, *([main[0]] if main else [])])
        x += width / 2
        half = sizes[main[0]][1] / 2 if main else (main_half if lift is True else 0.0)
        if lift is not True and lift:
            half += lift
        if main:
            pos[main[0]] = (x, 0.0)
        y = half
        for name in reversed(above):
            y += row_gap + sizes[name][1] / 2
            pos[name] = (x, y)
            y += sizes[name][1] / 2
        top = max(top, y)
        y = -half
        for name in below:
            y -= row_gap + sizes[name][1] / 2
            pos[name] = (x, y)
            y -= sizes[name][1] / 2
        if mirror:
            upper, lower = mirror
            d = max(pos[upper][1], -pos[lower][1])
            shift = {n: d - pos[upper][1] for n in above}
            shift.update({n: -d - pos[lower][1] for n in below})
            for name, dy in shift.items():
                pos[name] = (x, pos[name][1] + dy)
            top = max(top, *(pos[n][1] + sizes[n][1] / 2 for n in above))
        x += width / 2 + col_gap
    if "Title" in sizes:
        pos["Title"] = ((x - col_gap) / 2, top + row_gap + sizes["Title"][1] / 2)
    return pos


def _render(columns, edges, path, extras=(), title=None, footer=None, style=_THESIS, groups=(),
            panel=None):
    """Draw pinned columns plus `edges` as (tail, head, flow, label, kwargs).
    `groups` are (name, title, members, flow): a titled box drawn behind the
    members, which edges can leave from like any node. `panel` is a bold
    subfigure heading such as "(a) Flow chart", set above the top left."""
    from scripts.common.plot_common import FLOW_COLORS, FLOW_FILLS

    extras = list(extras)
    if title:
        extras.append(("Title", f"<<FONT POINT-SIZE=\"{style['title_font']}\">{title}</FONT>>"))
    sizes = _node_sizes(columns, extras, style)
    pos = _positions(columns, extras, sizes, style)
    nodes = [n for above, main, below, *_ in columns for n in [*above, *([main] if main else []), *below]]
    # A `drop` kwarg lowers a node that far (in) below its place in the column.
    for name, *_, kwargs in nodes:
        if kwargs.get("drop"):
            pos[name] = (pos[name][0], pos[name][1] - kwargs["drop"])
    # A `row` kwarg then sets a node level with the node it names, and a `dx`
    # kwarg shifts it that far (in) sideways, so nodes can sit side by side.
    for name, *_, kwargs in nodes:
        if kwargs.get("row"):
            pos[name] = (pos[name][0], pos[kwargs["row"]][1])
        if kwargs.get("dx"):
            pos[name] = (pos[name][0] + kwargs["dx"], pos[name][1])
    pad, head = 0.12, 0.45
    for name, _, members, _ in groups:
        left = min(pos[n][0] - sizes[n][0] / 2 for n in members) - pad
        right = max(pos[n][0] + sizes[n][0] / 2 for n in members) + pad
        bottom = min(pos[n][1] - sizes[n][1] / 2 for n in members) - pad
        top = max(pos[n][1] + sizes[n][1] / 2 for n in members) + head
        sizes[name] = (right - left, top - bottom)
        pos[name] = ((left + right) / 2, (top + bottom) / 2)
    # A `span` kwarg stretches a node across the outer edges of the nodes it
    # names, so lines into it from each of them can rise straight up. It is
    # raised, if need be, to clear the tallest of them by the slide vent gap.
    spans = set()
    for name, *_, kwargs in nodes:
        if kwargs.get("span"):
            over = kwargs["span"]
            left = min(pos[n][0] - sizes[n][0] / 2 for n in over)
            right = max(pos[n][0] + sizes[n][0] / 2 for n in over)
            clear = max(pos[n][1] + sizes[n][1] / 2 for n in over) + 0.7 + sizes[name][1] / 2
            sizes[name] = (right - left, sizes[name][1])
            pos[name] = ((left + right) / 2, max(pos[name][1], clear))
            kwargs["width"] = "%.3f" % (right - left)
            spans.add(name)
    g = _graph("300", style)
    if footer:
        g.attr(label=footer, labelloc="b")
    # Groups are declared first so they are drawn behind their members.
    for name, gtitle, _, flow in groups:
        g.node(name, f'<<B>{gtitle}</B>>', fontcolor=FLOW_COLORS[flow], color=FLOW_COLORS[flow],
               fillcolor=FLOW_FILLS[flow], labelloc="t", fixedsize="true",
               width="%.3f" % sizes[name][0], height="%.3f" % sizes[name][1],
               pos="%.3f,%.3f!" % pos[name])
    _add_nodes(g, columns, extras, pos)
    if panel:
        # A graph label sits outside the pinned nodes, so it never overlaps one.
        g.attr(label=f"<<B>{panel}</B>>", labelloc="t", labeljust="l",
               fontsize=style["title_font"])
    for tail, head, flow, label, kwargs in edges:
        kwargs = dict(kwargs)
        if kwargs.pop("curve_under", False):
            _route_curve(g, tail, head, FLOW_COLORS[flow], pos, sizes)
            continue
        vent_label = kwargs.pop("vent_label", None)
        if vent_label:
            # Pin a vent line's number beside the midpoint of the line, offset
            # square to it on the upper (or, for a vertical line, right) side.
            # Graphviz edge labels land on the line itself here.
            src, _, port = tail.partition(":")
            w, h = sizes[src]
            ox = {"n": 0, "ne": w / 2, "nw": -w / 2, "e": w / 2, "w": -w / 2}.get(port, 0)
            oy = {"e": 0, "w": 0}.get(port, h / 2)
            tx, ty = pos[src][0] + ox, pos[src][1] + oy
            hw, hh = sizes[head]
            hx = min(max(tx, pos[head][0] - hw / 2), pos[head][0] + hw / 2)
            hy = pos[head][1] - hh / 2
            if head in spans:
                # Neato aims an edge at the head's centre, so end the line on
                # a hidden point on the spanning node's bottom edge instead.
                anchor = f"{src}_{head}_anchor"
                g.node(anchor, "", shape="point", width="0.01", height="0.01", style="invis",
                       pos="%.3f,%.3f!" % (hx, hy))
                head = anchor
            dx, dy = hx - tx, hy - ty
            norm = (dx * dx + dy * dy) ** 0.5 or 1.0
            nx, ny = -dy / norm, dx / norm
            if ny < -1e-6 or (abs(ny) <= 1e-6 and nx < 0):
                nx, ny = -nx, -ny
            # Clear half the number's width, so longer numbers sit further out.
            off = 0.1 + 0.06 * len(vent_label)
            t = 0.75 if abs(dx) < 1e-6 and ty < 0 < hy else 0.5
            g.node(f"{src}_{head}_vlabel", vent_label, shape="plaintext", style="solid",
                   fontsize=style["edge_font"], fontcolor=FLOW_COLORS[flow], width="0", height="0",
                   # A vertical line rising from below the main line crosses it,
                   # so its number sits three quarters of the way up.
                   margin="0", pos="%.3f,%.3f!" % (tx + t * dx + nx * off, ty + t * dy + ny * off))
        g.edge(tail, head, color=FLOW_COLORS[flow], **kwargs)

        if label:
            # Labelled edges are the main line. neato puts an edge label wherever
            # the spline's midpoint lands, so pin it centred above the gap instead.
            left = pos[tail][0] + sizes[tail][0] / 2
            right = pos[head][0] - sizes[head][0] / 2
            g.node(f"{tail}_{head}_label", label, shape="plaintext", style="solid",
                   fontsize=style["edge_font"], fontcolor=FLOW_COLORS[flow], width="0", height="0",
                   margin="0", pos="%.3f,%.3f!" % ((left + right) / 2, 0.2))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Render in a scratch directory and copy the PNG across, so no Graphviz
    # source file is ever written (or deleted) beside the outputs.
    import shutil
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        png = g.render(filename=path.name, directory=tmp, cleanup=True, format="png")
        shutil.copyfile(png, path.with_name(path.name + ".png"))
    print(f"Saved: {path}.png")


def _route_curve(g, tail, head, colour, pos, sizes, gap=0.12, radius=0.2):
    """Draw tail -> head beneath the nodes between them: out of the tail's
    right side, down the gap before the next column, along a lane under every
    node on or below the main line, and up into the foot of the head, with
    rounded corners. Neato falls back to straight edges once any nodes touch
    (the slide's shipping box always encloses its legs), and a straight edge
    would run through them, so the route is drawn as a fine polyline through
    pinned points."""
    (tx, ty), (tw, th) = pos[tail], sizes[tail]
    (hx, hy), (_, hh) = pos[head], sizes[head]
    x0, y0 = tx + tw / 2, ty - th / 4
    # Every node the route could cross: on or below the main line between the ends.
    between = [n for n in pos if x0 < pos[n][0] < hx and pos[n][1] < sizes[n][1] / 2
               and n in sizes]
    left = min((pos[n][0] - sizes[n][0] / 2 for n in between), default=(x0 + hx) / 2)
    xv = (x0 + left) / 2
    lane = min([y0 - 2 * radius]
               + [pos[n][1] - sizes[n][1] / 2 - 2 * gap for n in between
                  if pos[n][0] + sizes[n][0] / 2 > xv and pos[n][0] - sizes[n][0] / 2 < hx])
    corners = [np.array(c) for c in ((x0, y0), (xv, y0), (xv, lane), (hx, lane),
                                     (hx, hy - hh / 2 - 0.15))]
    points = [corners[0]]
    for q, c, r in zip(corners, corners[1:], corners[2:]):
        u_in = (c - q) / np.linalg.norm(c - q)
        u_out = (r - c) / np.linalg.norm(r - c)
        # Each corner a quadratic Bezier with the corner as its control point.
        a, b = c - radius * u_in, c + radius * u_out
        for t in np.linspace(0, 1, 9):
            points.append((1 - t) ** 2 * a + 2 * (1 - t) * t * c + t ** 2 * b)
    points.append(corners[-1])
    # The last stretch is a normal edge so neato draws the arrowhead into the head.
    names = [f"{tail}_{head}_via{i}" for i in range(len(points))]
    for name, xy in zip(names, points):
        g.node(name, "", shape="point", width="0.01", height="0.01", style="invis",
               pos="%.3f,%.3f!" % tuple(xy))
    # Unclipped, each segment runs to the point's centre and meets the next.
    for a, b in zip(names, names[1:]):
        g.edge(a, b, color=colour, arrowhead="none", tailclip="false", headclip="false")
    g.edge(names[-1], head, color=colour, tailclip="false")


def _edge(tail, head, flow, label=None, **kwargs):
    return (tail, head, flow, label, kwargs)


def _round_to_total(values, decimals=2):
    """Round `values` so the rounded parts add up to the rounded total
    (largest remainder), so a split vent reads as the sum it feeds."""
    scale = 10 ** decimals
    total = round(sum(values) * scale)
    floors = [int(v * scale) for v in values]
    order = sorted(range(len(values)), key=lambda i: values[i] * scale - floors[i], reverse=True)
    for i in order[:max(0, total - sum(floors))]:
        floors[i] += 1
    return [f / scale for f in floors], total / scale


def _vent_total(name, sources, note=None):
    """Slide vent: one total node, spread across its sources, and a dashed,
    numbered line from each. `sources` is [(tail, mtpa)]; returns (node, edges)."""
    parts, total = _round_to_total([v for _, v in sources])
    node = _node(name, "vent", "Vented CO2", [f"{total:,.2f} Mtpa", *([note] if note else [])],
                 span=[tail.partition(":")[0] for tail, _ in sources])
    edges = [_edge(tail, name, "vent", style="dashed", vent_label=f"{part:,.2f}")
             for (tail, _), part in zip(sources, parts) if part > 0]
    return node, edges


def _ship_node(subtitle, halves):
    """Slide shipping block: one grey box holding a stacked cell per leg, each
    a port (`PORT:<name>:w/e`) so the legs' lines enter and leave it.
    `halves` is [(port, flow, title, lines)], listed top to bottom."""
    from scripts.common.plot_common import FLOW_COLORS, FLOW_FILLS, format_html_node

    cells = ""
    for port, flow, title, lines in halves:
        inner = format_html_node(title, lines, FLOW_COLORS[flow])[1:-1]
        cells += (f'<TR><TD PORT="{port}" STYLE="rounded" BGCOLOR="{FLOW_FILLS[flow]}" '
                  f'COLOR="{FLOW_COLORS[flow]}" CELLPADDING="4">{inner}</TD></TR>')
    html = (
        f'<<TABLE BORDER="1" STYLE="rounded" COLOR="{FLOW_COLORS["util"]}" '
        f'BGCOLOR="{FLOW_FILLS["util"]}" CELLBORDER="1" CELLSPACING="3" CELLPADDING="2">'
        f'<TR><TD BORDER="0"><B><FONT COLOR="{FLOW_COLORS["util"]}">Shipping</FONT></B></TD></TR>'
        f'<TR><TD BORDER="0">{subtitle}</TD></TR>{cells}</TABLE>>'
    )
    return _node("Ship", "util", None, None, html=html, shape="plain", style="solid")


def plot_base_case_flow(p, lng_params, outdir=None):
    from scripts.common.plot_common import DEFAULT_DIR, format_legend

    target = p.target_h2_tpa
    n = p.fleet_plants
    plant_scale = target / (n * p.annual_h2_per_plant_tpa)
    capacity_tpd = p.plant_capacity_tpd_h2 * plant_scale
    fleet_h2_kg = target * 1000.0
    ng_mass_mt = (
        fleet_h2_kg
        * (p.feed_gj_per_kg_h2 * 1000.0 / lng_params.lng_lhv_mj_per_kg)
        / 1e9
    )
    co2_vent = p.plant_gate_ei_kg_co2_per_kg_h2 * fleet_h2_kg / 1e9
    co2_up = p.upstream_ei_kg_co2_per_kg_h2 * fleet_h2_kg / 1e9

    columns = [
        _column(
            _node("LNG", "energy", "LNG Export", [f"{lng_params.lng_capacity_mtpa:.1f} MTPA"]),
            above=[_vent("LngVent", co2_up, "upstream")],
        ),
        _column(
            _node("ATR", "h2", "ATR Plant", [f"{n} x {capacity_tpd:,.0f} t/d", "No CCS"]),
            above=[_vent("AtrVent", co2_vent, "plant")],
        ),
        _column(_node("Target", "h2", "Delivered H2", [f"{target / 1e6:,.2f} Mtpa"],
                      style="bold,rounded,filled")),
    ]
    edges = [
        _edge("LNG", "ATR", "energy", f"LNG {ng_mass_mt:,.2f} Mtpa"),
        _edge("ATR", "Target", "h2", f"H2 {target / 1e6:,.2f} Mtpa"),
        _edge("LNG", "LngVent", "vent", style="dashed"),
        _edge("ATR", "AtrVent", "vent", style="dashed"),
    ]
    legend = [("Legend", format_legend(("energy", "h2", "vent")))]
    _render(columns, edges, Path(outdir or DEFAULT_DIR) / "base_case_flow_grey", legend)


def _site_table_label(p):
    cols = ["Port", "Wind MW", "Solar MW", "PEM modules", "n Synth", "Batt MWh", "H2 Store kt"]
    hdr = (
        "<TR>"
        + "".join(f'<TD BORDER="0"><B>{c}</B></TD>' for c in cols)
        + "</TR>"
    )
    body = ""
    for _port in PORT_SITES:
        _w = p.site_wind_mw.get(_port, 0.0)
        _s = (p.site_solar_mw or {}).get(_port, 0.0)
        if _port not in p.site_h2_frac and _port not in p.idle_site_renewable_mwh:
            # Keep the port on the diagram so the option set stays visible.
            body += (
                f"<TR>"
                f'<TD BORDER="0"><FONT COLOR="#888888">{_port}</FONT></TD>'
                f'<TD BORDER="0" COLSPAN="{len(cols) - 1}"><FONT COLOR="#888888"><I>Not used</I></FONT></TD>'
                f"</TR>"
            )
            continue
        cells = [
            _port + (" (idle)" if _port in p.idle_site_renewable_mwh else ""),
            f"{_w:,.0f}",
            f"{_s:,.0f}",
            p.pem_modules(_port),
            p.site_n_synth.get(_port, 0),
            f"{(p.site_battery_mwh or {}).get(_port, 0.0):,.0f}",
            f"{(p.site_h2_storage_kg or {}).get(_port, 0.0) / 1e6:,.1f}",
        ]
        body += (
            "<TR>"
            + "".join(f'<TD BORDER="0">{c}</TD>' for c in cells)
            + "</TR>"
        )
    return (
        '<<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="2" CELLPADDING="3">'
        f'<TR><TD COLSPAN="{len(cols)}" BORDER="0"><B>Per-Site Allocation</B></TD></TR>'
        + hdr
        + body
        + "</TABLE>>"
    )


def _cracker_heat_lines(p):
    c = p.cracker
    feed = p.required_cracker_meoh_tpa / p.n_cracker
    lines = [
        f"Reformer duty {c.gross_thermal_gj_per_year / feed:.1f} GJ/t MeOH (endothermic)",
        f"PSA tail gas {c.tail_gas_heat_credit_gj_per_year / c.gross_thermal_gj_per_year:.0%}",
    ]
    if c.thermal_gj_per_year > 0:
        # Electric reformer heat is all electric heat less capture regeneration,
        # whether a resistance heater or the heat pump delivered it.
        electric = (
            (c.resistance_heat_mwh + c.heat_pump_heat_mwh) * 3.6
            - c.capture_regen_heat_gj_per_year
        ) / c.thermal_gj_per_year
        fuels = [f"{p.fleet_h2_fuel_tpa / 1e3:,.0f} kt/yr H2"] if p.fleet_h2_fuel_tpa > 0 else []
        if p.fleet_heat_meoh_tpa > 0:
            fuels.append(f"{p.fleet_heat_meoh_tpa / 1e6:.2f} Mtpa MeOH")
        balance = [f"{electric:.0%} electric"] if electric > 0.005 else []
        if fuels:
            balance.append(f"{1 - electric:.0%} fired ({' + '.join(fuels)})")
        if balance:
            lines.append("Balance " + ", ".join(balance))
    if c.heat_pump_heat_mwh > 0:
        lines.append(f"Heat pump {c.heat_pump_heat_mwh * p.n_cracker / 1e3:,.0f} GWh/yr heat")
    if c.reformer_electricity_mwh > 0:
        lines.append(f"Reformer power {c.reformer_electricity_mwh * p.n_cracker / 1e3:,.0f} GWh/yr")
    lines.append(
        f"Capture regeneration power {c.regeneration_electricity_mwh * p.n_cracker / 1e3:,.0f} GWh/yr")
    lines.append(
        ("Zero-carbon" if c.cracker_power_source else "Japan grid")
        + f" power, {c.power_ei_kg_co2_per_kwh:.2f} kg CO2/kWh"
    )
    lines.append(
        f"{c.effective_capture_rate:.0%} of "
        + ("flue CO2 captured" if c.co2_capture_config else "reaction CO2 captured before PSA")
    )
    return lines


def _cracker_heat_short(p):
    """Slide version of _cracker_heat_lines: heat source split and capture."""
    c = p.cracker
    if c.thermal_gj_per_year > 0:
        electric = (
            (c.resistance_heat_mwh + c.heat_pump_heat_mwh) * 3.6
            - c.capture_regen_heat_gj_per_year
        ) / c.thermal_gj_per_year
        fuel = "H2" if p.fleet_h2_fuel_tpa > 0 else "MeOH"
        heat = ("Electric" if electric > 0.995
                else f"{fuel}-fired" if electric < 0.005
                else f"{electric:.0%} electric")
    else:
        heat = "Tail gas"
    return ["<I>" + heat + (" + heat pump" if c.heat_pump_heat_mwh > 0 else "") + "</I>",
            f"{c.effective_capture_rate:.0%} CO2 capture"
            + (" (amine)" if c.co2_capture_config else " (PSA)")]


def plot_meoh_flow(p, outdir: Path = None, build_note=None, slide=False,
                   panel="(a) Flow chart"):
    """Flowsheet of the delivered MeOH pathway. `slide` draws a 16 pt version
    with one or two short lines per node and no per-site table, sized for a
    16:9 slide, saved as flow_diagram_thesis.png with an "(a) Flow chart" heading."""
    if not p.hourly_feasible:
        raise ValueError("Cannot plot an infeasible candidate as a delivered pathway")

    from scripts.common.plot_common import DEFAULT_DIR, FLOW_LABELS, format_legend

    def pick(full, short):
        return short if slide else full

    def vent(name, mtpa, source):
        # The slide puts the source on its own line to keep the node narrow.
        if slide:
            return _node(name, "vent", "CO2 vent", [f"{mtpa:,.2f} Mtpa", source])
        return _vent(name, mtpa, source)

    title = build_note

    target = p.fleet_h2_delivered_tpa
    # H2 compression is drawn with the PEM load: both sit at the renewables site.
    fleet_pem_gwh = (
        p.fleet_pem_electricity_kwh_per_year
        + p.fleet_transport_electricity_kwh_per_year
    ) / 1e6
    show_batt = p.battery_power_capacity_mw > 0
    show_h2store = p.h2_storage_capacity_kg > 0
    landed_meoh_mtpa = (p.required_cracker_meoh_tpa + p.fleet_heat_meoh_tpa) / 1e6
    pem_water_mtpa = p.pem.water_kl_per_year * p.n_pem / 1e6
    pem_water_l_per_kg = p.pem.water_kl_per_year * 1e3 / p.pem.annual_h2_kg
    synth_water_mtpa = p.fleet_annual_meoh_tpa * (M_H2O / M_MEOH) / 1e6
    fleet_pem_h2_mtpa = p.pem.annual_h2_kg * p.n_pem / 1e9
    ship_vent = (p.ship_fuel_ei + p.meoh_loss_ei) * target / 1e6
    crack_vent = (p.cracker_vent_ei + p.cracker_electricity_ei + p.cracker_heat_ei) * target / 1e6

    columns = []
    edges = []
    # The slide sets the power block and its sources below the main line, so
    # the electricity line to synthesis can curve under electrolysis.
    drop = dict(drop=0.35) if slide else {}
    # The slide sets wind and solar side by side over the power block,
    # level with the make-up CO2, rather than in a column of their own.
    sources = ((dict(row="MakeupCO2", dx=-1.05), dict(row="MakeupCO2", dx=0.45))
               if slide else (drop, drop))
    wind = _node("Wind", "energy", "Wind", pick([
        f"{p.wind_capacity_mw:,.0f} MW",
        f"{p.wind_annual_energy_mwh / 1e3:,.0f} GWh/yr",
    ], [f"{p.wind_capacity_mw / 1e3:,.1f} GW"]), **sources[0])
    solar = _node("Solar", "energy", pick("Solar PV", "Solar"), pick([
        f"{p.solar_capacity_mw:,.0f} MW",
        f"{p.solar_annual_energy_mwh / 1e3:,.0f} GWh/yr",
    ], [f"{p.solar_capacity_mw / 1e3:,.1f} GW"]), **sources[1])
    if not slide:
        columns.append(_column(above=[wind], below=[solar]))
    edges += [_edge("Wind", "Dist", "energy"), _edge("Solar", "Dist", "energy")]

    dist_lines = [
        f"{p.fleet_total_electricity_kwh_per_year / 1e6:,.0f} GWh/yr to loads",
        f"PEM + H2 compression {fleet_pem_gwh:,.0f} GWh/yr",
        f"Synthesis {p.fleet_synth_electricity_kwh_per_year / 1e6:,.0f} GWh/yr",
        f"Port line loss {p.transmission_loss_mwh / 1e3:,.0f} GWh/yr",
    ]
    dist_lines = pick(dist_lines, [f"{p.fleet_total_electricity_kwh_per_year / 1e9:,.0f} TWh/yr"])
    batt = []
    if show_batt:
        batt_h = p.battery_energy_capacity_mwh / p.battery_power_capacity_mw
        batt = [_node("Batt", "store", "Battery", pick([
            f"{p.battery_power_capacity_mw:,.0f} MW", f"{batt_h:.0f} h duration"],
            [f"{p.battery_power_capacity_mw / 1e3:,.1f} GW", f"{batt_h:.0f} h"]), **drop)]
        edges.append(_edge("Dist", "Batt", "energy", dir="both"))
    columns.append(_column(_node("Dist", "energy", pick("Power Distribution", "Power"),
                                 dist_lines, **drop), below=batt,
                           above=[wind, solar] if slide else []))

    makeup = _node("MakeupCO2", "co2", "Make-up CO2", [
        pick("", "<I>DAC</I>, ") + f"{p.fleet_annual_co2_makeup_tpa / 1e6:.2f} Mtpa"])
    h2store = _node("H2Store", "store", "H2 Storage", pick(
        [f"{p.h2_storage_capacity_kg / 1e6:,.1f} kt working gas"],
        ["<I>Geological</I>", f"{p.h2_storage_capacity_kg / 1e6:,.0f} kt working"]),
        **(dict(row="Batt") if slide and show_batt else {}))
    columns.append(_column(
        _node("PEM", "h2", pick("PEM Electrolyser", "Electrolysis"), pick([
            f"{p.n_pem * p.pem.small_module_mw:,.0f} MW",
            f"{p.pem_actual_op_hrs:,.0f} h/yr (CF {p.pem_capacity_factor:.1%})",
        ], [
            "<I>PEM</I>",
            f"{p.n_pem * p.pem.small_module_mw / 1e3:,.1f} GW",
            f"CF {p.pem_capacity_factor:.0%}",
        ])),
        # The slide leaves out both water nodes, and sets the make-up CO2 over
        # electrolysis and H2 storage under it, level with the battery.
        above=[makeup] if slide else [],
        below=([h2store] if show_h2store else []) if slide else [
            _node("PemWater", "util", "Process water", [
                f"{pem_water_mtpa:,.1f} Mt/yr demineralised",
                f"{pem_water_l_per_kg:.1f} L/kg H2",
            ])],
    ))
    edges.append(_edge("Dist", "PEM", "energy"))
    if not slide:
        edges.append(_edge("PemWater", "PEM", "util"))

    # Slide edges carry no labels: the narrow column gaps leave no room.
    h2_label = pick(f"H2 {fleet_pem_h2_mtpa:.2f} Mtpa", None)
    if slide:
        # Storage only buffers the difference between production and demand,
        # so electrolysis feeds synthesis directly as well as through storage.
        edges.append(_edge("PEM", "Synth", "h2"))
        if show_h2store:
            edges += [_edge("PEM", "H2Store", "h2"), _edge("H2Store", "Synth", "h2")]
    elif show_h2store:
        columns.append(_column(h2store, above=[makeup]))
        edges += [
            _edge("PEM", "H2Store", "h2", h2_label),
            _edge("H2Store", "Synth", "h2"),
        ]
    else:
        columns.append(_column(above=[makeup]))
        edges.append(_edge("PEM", "Synth", "h2", h2_label))

    columns.append(_column(
        _node("Synth", "meoh", "MeOH Synthesis", pick([
            f"{p.n_synth} x {p.synth.plant_capacity_tpy_meoh / DAYS_PER_YEAR:,.0f} t/d",
            f"{p.synth_actual_op_hrs:,.0f} h/yr (CF {p.synth_capacity_factor:.1%})",
        ], [
            "<I>CO2 hydrogenation</I>",
            f"{p.n_synth} x {p.synth.plant_capacity_tpy_meoh / DAYS_PER_YEAR:,.0f} t/d",
            f"CF {p.synth_capacity_factor:.0%}",
        ])),
        above=[] if slide else [vent("SynthVent", p.synth_process_ei * p.target_h2_tpa / 1e6,
                                     "purge gas")],
        below=[] if slide else [_node("SynthWater", "util", "Water byproduct", [
            f"{synth_water_mtpa:,.1f} Mt/yr", "CO2 + 3 H2 -&gt; CH3OH + H2O"])],
    ))
    edges += [
        # The slide's edges are straight (see _route_curve), so this one is
        # curved under electrolysis and storage rather than through them.
        _edge("Dist", "Synth", "energy", curve_under=slide),
        _edge("MakeupCO2", "Synth", "co2"),
    ]
    if not slide:
        edges.append(_edge("Synth", "SynthVent", "vent", style="dashed"))
    if not slide:
        edges.append(_edge("Synth", "SynthWater", "util"))

    co2_ret_vent = p.co2_capture_ei * target / 1e6
    if slide:
        # One shipping block on the main line, split methanol out (top) and
        # CO2 back (bottom): both legs sail on the same carriers. Its single
        # vent covers fuel, cargo losses and CO2 liquefaction for the loop.
        ship_fuel = "MeOH-fuelled" if p.fleet_ship_fuel_meoh_tpa > 0 else "Fuel-oil fuelled"
        ship = _ship_node(f"{p.fleet_ship_total} carriers<BR/><I>{ship_fuel}</I>", [
            ("meoh", "meoh", "MeOH supply", [f"{landed_meoh_mtpa:.1f} Mtpa MeOH"]),
            ("co2", "co2", "CO2 return", [f"{p.fleet_annual_co2_returned_tpa / 1e6:.1f} Mtpa LCO2"]),
        ])
        # One vented-CO2 total over the shipping column, fed by a numbered line
        # from each source: it equals the DAC make-up that closes the loop.
        vent_node, vent_edges = _vent_total("Vent", [
            ("Synth:n", p.synth_process_ei * target / 1e6),
            ("Ship:n", ship_vent + co2_ret_vent),
            ("Cracker:n", crack_vent)])
        columns.append(_column(
            ship, above=[vent_node], lift=0.35,
        ))
        edges += vent_edges + [
            _edge("Synth", "Ship:meoh:w", "meoh"),
            _edge("Ship:co2:w", "Synth", "co2"),
            _edge("Ship:meoh:e", "Cracker", "meoh"),
            _edge("Cracker", "Ship:co2:e", "co2"),
        ]
    else:
        # The two shipping legs are drawn as a loop, methanol out above the
        # main line and CO2 back below it.
        columns.append(_column(
            above=[
                vent("ShipVent", ship_vent, "fuel + losses"),
                _node("Ship", "meoh", "MeOH Shipping", [
                    f"{p.fleet_ship_total} MeOH/LCO2 carriers, route-specific voyages",
                    f"Loaded {p.fleet_annual_meoh_tpa / 1e6:.2f} Mtpa incl. bunkers",
                    f"Cargo {p.fleet_meoh_shipped_tpa / 1e6:.2f} Mtpa, landed {landed_meoh_mtpa:.2f} Mtpa",
                ]),
                _node("Bunker", "util", "Ship bunkers", [
                    f"{p.fleet_ship_fuel_meoh_tpa / 1e6:.2f} Mtpa MeOH",
                    "Part of AU synthesis output,",
                    "loaded at the export port",
                ]),
            ],
            mirror=("Ship", "CO2Return"),
            below=[
                _node("CO2Return", "co2", "CO2 Return Leg", [
                    "Same carriers, liquefied in JP",
                    f"{p.fleet_annual_co2_returned_tpa / 1e6:.2f} Mtpa captured CO2",
                ]),
                vent("CO2RetVent", co2_ret_vent, "liquefaction + losses"),
            ],
        ))
        edges += [
            _edge("Synth", "Ship", "meoh"),
            _edge("Ship", "ShipVent", "vent", style="dashed"),
            _edge("Ship", "Bunker", "util"),
            _edge("CO2Return", "Synth", "co2"),
            _edge("CO2Return", "CO2RetVent", "vent", style="dashed"),
            _edge("Ship", "Cracker", "meoh"),
            _edge("Cracker", "CO2Return", "co2"),
        ]

    columns.append(_column(
        _node("Cracker", "reformer", pick("MeOH Cracker (JP)", "H2 Reforming"), pick([
            f"{p.n_cracker} x {p.cracker.gross_capacity_tpd_h2:,.0f} t/d gross H2",
            f"{p.cracker_actual_op_hrs:,.0f} h/yr (CF {p.cracker_capacity_factor:.1%})",
            f"Reformer feed {p.required_cracker_meoh_tpa / 1e6:.2f} Mtpa MeOH",
        ], [
            "<I>MSR + PSA</I>",
            f"{p.n_cracker} x {p.cracker.gross_capacity_tpd_h2:,.0f} t/d H2",
            f"CF {p.cracker_capacity_factor:.0%}",
        ])),
        above=[] if slide else [vent("CrackerVent", crack_vent, "process, heat + power")],
        below=[_node("CrackerHeat", "util", pick("Reformer and capture heat", "Supplemental heat"),
                     pick(_cracker_heat_lines(p), _cracker_heat_short(p)))],
    ))
    edges.append(_edge("CrackerHeat", "Cracker", "util"))
    if not slide:
        edges.append(_edge("Cracker", "CrackerVent", "vent", style="dashed"))

    columns.append(_column(_node("Target", "h2", "Delivered H2", [f"{target / 1e6:.2f} Mtpa"],
                                 style="bold,rounded,filled")))
    edges.append(_edge("Cracker", "Target", "reformer", pick(f"H2 {target / 1e6:.2f} Mtpa", None)))

    # Node titles name each flow, so the slide leaves the legend out for width.
    legend = [] if slide else [
        ("Legend", format_legend(list(FLOW_LABELS)))]
    footer = _site_table_label(p) if len(p.site_wind_mw) > 1 and not slide else None
    name = "flow_diagram_thesis" if slide else "meoh_flow_diagram"
    _render(columns, edges, Path(outdir or DEFAULT_DIR) / name,
            legend, title, footer, style=_SLIDE if slide else _THESIS,
            panel=panel if slide else None)




# Shipping's share of the well-to-terminal LNG chain behind
# supply_chain_ghg_kg_co2_per_gj: 51 of 719 kg CO2/t LNG (well head 85,
# processing 88, water 0.8, dehydration 0.3, pipeline 129, liquefaction 257,
# shipping 51, regasification 108). Schandl et al.
# (2019), CSIRO GISERA, Table 11. Used only to show the voyage on the slides.
_LNG_SHIPPING_SHARE = 51 / 719.1


def plot_lng_flow_slide(blue, lng_params, outdir, ccs, panel="(a) Flow chart"):
    """Slide-style flowsheet of the grey (ccs=False) or blue (ccs=True) LNG
    pathway, drawn to match the MeOH flow_diagram_thesis: short nodes, no edge
    labels, one vented-CO2 total with a numbered line from each source."""
    grey = blue.grey
    target = grey.target_h2_tpa
    h2_kg = target * 1000.0
    lng_mtpa = h2_kg * grey.feed_gj_per_kg_h2 * 1000.0 / lng_params.lng_lhv_mj_per_kg / 1e9
    capacity_tpd = grey.plant_capacity_tpd_h2 * target / (grey.fleet_plants * grey.annual_h2_per_plant_tpa)
    upstream = grey.upstream_ei_kg_co2_per_kg_h2 * h2_kg / 1e9
    lng_ship = upstream * _LNG_SHIPPING_SHARE
    stored = blue.fleet_co2_stored_tpa / 1e6
    extraction = _node("LNG", "energy", "Extraction + Liquefaction", [
        "<I>Surat Basin to Curtis Island</I>", f"{lng_mtpa:,.1f} Mtpa LNG"])
    cargo = _node("LngShip", "energy", "LNG Shipping", [
        "<I>Dual-fuel (BOG + MGO)</I>", f"{lng_mtpa:,.1f} Mtpa LNG"])
    atr = _node("ATR", "reformer", "H2 Reforming", [
        "<I>ATR + ASU</I>", f"{grey.fleet_plants} x {capacity_tpd:,.0f} t/d H2",
        *([f"{blue.captured_co2_kg_per_kg_h2 / (blue.captured_co2_kg_per_kg_h2 + blue.vented_co2_kg_per_kg_h2):.0%} CO2 capture"]
          if ccs else [])])
    target_node = _node("Target", "h2", "Delivered H2", [f"{target / 1e6:.2f} Mtpa"],
                        style="bold,rounded,filled")
    edges = [_edge("LNG", "LngShip", "energy"), _edge("LngShip", "ATR", "energy"),
             _edge("ATR", "Target", "h2")]
    if ccs:
        # Two fleets, one per row: the LNG voyage on the product line and the
        # LCO2 return level with storage, grouped so one line carries both
        # voyages' CO2. The rest of the LNG chain stays on extraction.
        lco2 = (blue.co2_liquefaction_ei_kg_co2_per_kg_h2
                + blue.co2_shipping_ei_kg_co2_per_kg_h2) * h2_kg / 1e9
        plant = (blue.vented_co2_kg_per_kg_h2 + grey.atr_electricity_ei_kg_co2_per_kg_h2
                 + blue.capture_power_ei_kg_co2_per_kg_h2) * h2_kg / 1e9
        sources = [("LNG:n", upstream - lng_ship), ("Shipping:n", lng_ship + lco2), ("ATR:n", plant)]
        groups = [("Shipping", "Shipping", ["LngShip", "CO2Ret"], "util")]
        columns = [
            _column(extraction, below=[_node("Store", "co2", "CO2 Storage", [
                "<I>Geological</I>", f"{stored:,.1f} Mtpa"], row="CO2Ret")]),
            _column(cargo, below=[_node("CO2Ret", "co2", "CO2 Return", [
                "<I>Fuel oil</I>", f"{stored:,.1f} Mtpa LCO2"])]),
        ]
        edges += [_edge("ATR", "CO2Ret", "co2"), _edge("CO2Ret", "Store", "co2")]
    else:
        # The supply-chain intensity runs well to terminal, voyage included,
        # so one line leaves the box around the whole LNG chain.
        plant = (grey.plant_gate_ei_kg_co2_per_kg_h2 + grey.atr_electricity_ei_kg_co2_per_kg_h2) * h2_kg / 1e9
        sources = [("Supply:n", upstream), ("ATR:n", plant)]
        groups = [("Supply", "LNG Supply", ["LNG", "LngShip"], "util")]
        columns = [_column(extraction), _column(cargo)]
    vent_node, vent_edges = _vent_total("Vent", sources)
    columns += [_column(atr, above=[vent_node]), _column(target_node)]
    _render(columns, vent_edges + edges, Path(outdir) / "flow_diagram_thesis", style=_SLIDE,
            groups=groups, panel=panel)
