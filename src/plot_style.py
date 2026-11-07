"""Shared style of the report's figures, after the figures of recent ML conference papers.

Times New Roman as in the text, bold axis labels, boxed axes with inward ticks and a light dashed
grid, framed legends and sub-captions such as "(a) ..." under the panels. One color ramp for the
update rules: BP is the warm baseline, and the blue gets darker the more of each step the proximal
operator sets (NM-static, NM-dynamic, ProxProp, ProxProp-all). The five colors pass the palette
checks for adjacent pairs under normal vision and color-vision deficiency; markers and line styles
encode the same identity.
"""
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42  # embed TrueType, not Type 3
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402

INK, INK2, MUTED, GRID, BAND = "#1a1a1a", "#4d4d4d", "#8c8c8c", "#cdcdcd", "#dedede"
METHOD = {  # name, color, marker, linestyle
    "bp_sgd": ("BP+SGD", "#c8553d", "o", "-"),
    "nm_static": ("NM-static", "#86bce0", "v", ":"),
    "nm_dynamic": ("NM-dynamic", "#3d8fcf", "^", "--"),
    "prox_sgd": ("ProxProp", "#1a5ca8", "s", "-"),
    "prox_all": ("ProxProp-all", "#0a2a55", "D", "-"),
    "bp_adam": ("BP+Adam", "#9a9a9a", "x", (0, (1, 1))),
}
RAMP3 = ["#86bce0", "#3d8fcf", "#0a2a55"]  # three ordered steps of the same blue, light to dark
BOX_FILL = "#d6e7f5"
DSN = {"drybean": "Dry Bean", "optdigits": "Optdigits", "har": "HAR"}


def apply(size=7.0):
    """Set the rcParams of the report style; `size` is the base font size in points."""
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
        "font.size": size, "axes.labelsize": size + 0.5, "axes.labelweight": "bold",
        "axes.titlesize": size + 0.5, "axes.titleweight": "bold", "axes.titlepad": 4,
        "legend.fontsize": size - 0.7, "xtick.labelsize": size - 0.8, "ytick.labelsize": size - 0.8,
        "axes.edgecolor": INK, "axes.linewidth": 1.0, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.direction": "in", "ytick.direction": "in",
        "xtick.major.size": 2.6, "ytick.major.size": 2.6, "xtick.minor.size": 1.4, "ytick.minor.size": 1.4,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.45, "ytick.minor.width": 0.45,
        "axes.grid": True, "grid.color": GRID, "grid.linestyle": (0, (3, 3)), "grid.linewidth": 0.45,
        "axes.axisbelow": True, "legend.frameon": True, "legend.framealpha": 0.95, "legend.edgecolor": "#b5b5b5",
        "legend.fancybox": False, "legend.borderpad": 0.35, "legend.handletextpad": 0.4,
        "lines.linewidth": 1.1, "lines.markersize": 3.0, "patch.linewidth": 0.6,
    })


def subcaptions(fig, axes, texts, gap=0.01):
    """Bold sub-captions such as '(a) First-layer input spectrum', centered under each panel and aligned
    just below the lowest tick or axis label of all panels. Call after the layout is final."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_fig = fig.transFigure.inverted()
    y = min(to_fig.transform(ax.get_tightbbox(renderer).p0)[1] for ax in axes) - gap
    for ax, text in zip(axes, texts):
        box = ax.get_position()
        fig.text((box.x0 + box.x1) / 2, y, text, ha="center", va="top", weight="bold",
                 fontsize=plt.rcParams["font.size"] + 0.5, color=INK)
