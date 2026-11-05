"""Shared argparse and plotting-init preamble for the scripts/ CLI layer."""

import argparse
from pathlib import Path


def build_parser(description, *, outdir=None, plot=True, replot=False):
    """A base ArgumentParser with the --outdir / --no-plot / --replot flags
    the runner scripts share. Pass outdir=<default Path> to add --outdir,
    plot=True to add --no-plot, replot=True to add --replot. Callers add any
    script-specific arguments to the returned parser before parse_args()."""
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    if outdir is not None:
        parser.add_argument("--outdir", type=Path, default=outdir)
    if plot:
        parser.add_argument("--no-plot", action="store_true", help="Skip plots")
    if replot:
        parser.add_argument("--replot", action="store_true",
                             help="Regenerate plots from saved outputs without recomputing")
    return parser


def init_plotting():
    """matplotlib.use('Agg') then the shared theme, replacing the copy-pasted
    preamble at the top of every plotting script. Also creates the output
    directories -- importing scripts.common.plot_common alone must not touch disk."""
    import matplotlib
    matplotlib.use("Agg")
    from scripts.common.plot_common import apply_theme, ensure_output_dirs
    ensure_output_dirs()
    apply_theme()
