"""Shared figure design system for the TRB / journal papers.

One import, one look. Nature conventions, tightened to postdoc production quality:
  * lowercase bold panel letters (a, b, c) top-left  -- Nature standard
  * short scientific panel TITLES (sentence fragment, not a headline)
  * axis labels bold, first letter uppercase, units in parentheses, fixed labelpad
  * despined, no heavy gridlines, direct labelling over legends
  * one fixed, colourblind-safe palette with FIXED semantic roles across every figure
  * vector PDF, fonts embedded (Type 42)

Every figure module calls fs.setup() once, uses fs.C / fs.EVENT / fs.COND / fs.MODEL for colour,
fs.panel_label(), fs.title(), fs.axis(), and fs.save().
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ------------------------------------------------------------------ palette
# Wong colourblind-safe base, curated into a cohesive cool/warm theme.
# ---------------------------------------------------------------- palette
# Sourced from palette.py, which searches for and then PROVES the colour choices: every pair that
# can share a panel clears dE 18 under deuteranopia, protanopia and tritanopia, and the object
# identities sit on a lightness ladder so the raster survives a greyscale print. Do not hardcode a
# colour in a figure module; add a semantic key there instead.
from palette import C as _P, OBJECT, ramp, verify as verify_palette   # noqa: F401

C = dict(_P)
C["cv"] = _P["human"]              # motion / computer vision reads as a third voice
C["accentA"], C["accentA0"], C["accentB"] = _P["condA"], _P["condA0"], _P["condB"]

# Fixed spacing rules.  These constants keep axis labels and panel headings on the
# same visual rhythm when a figure combines maps, bars, heat maps, and line plots.
AXIS_LABEL_PAD = 6
PANEL_TITLE_PAD = 8

# fixed event colours (used identically everywhere)
EVENT = {"alarm": C["engine"], "movement": C["vlm"], "egress": C["human"]}
EVENT_LABEL = {"alarm": "Alarm onset", "movement": "Movement initiation",
               "egress": "Egress completion"}
INTERVAL_LABEL = {"pre_movement": "Pre-movement time", "travel": "Travel time",
                  "total": "Total evacuation time"}

# condition colours
COND = {"A": C["condA"], "A0": C["condA0"], "B": C["condB"]}
COND_LABEL = {"A": "Video + audio", "A0": "Video, muted", "B": "Frames only"}

# a stable, distinguishable colour per model (assigned by family)
MODEL = {
    "gemini-3.1-pro-preview":             "#005376",
    "gemini-3-flash-preview":             "#007BA0",
    "gemini-3.1-flash-lite":              "#2DA8C9",
    "openai/gpt-5.2":                     "#408555",
    "anthropic/claude-opus-4.8":          "#A23B47",
    "anthropic/claude-sonnet-5":          "#D4757A",
    "qwen/qwen3-vl-235b-a22b-instruct":   "#956CAD",
    "meta-llama/llama-4-maverick":        "#CDAD5B",
}
MODEL_SHORT = {
    "gemini-3-flash-preview":          "Gemini 3 Flash",
    "gemini-3.1-pro-preview":          "Gemini 3.1 Pro",
    "gemini-3.1-flash-lite":           "Gemini 3.1 Flash-Lite",
    "openai/gpt-5.2":                  "GPT-5.2",
    "anthropic/claude-opus-4.8":       "Claude Opus 4.8",
    "anthropic/claude-sonnet-5":       "Claude Sonnet 5",
    "qwen/qwen3-vl-235b-a22b-instruct":"Qwen3-VL 235B",
    "meta-llama/llama-4-maverick":     "Llama 4 Maverick",
}
MODEL_VENDOR = {
    "gemini-3-flash-preview": "Google", "gemini-3.1-pro-preview": "Google",
    "gemini-3.1-flash-lite": "Google", "openai/gpt-5.2": "OpenAI",
    "anthropic/claude-opus-4.8": "Anthropic", "anthropic/claude-sonnet-5": "Anthropic",
    "qwen/qwen3-vl-235b-a22b-instruct": "Alibaba", "meta-llama/llama-4-maverick": "Meta",
}

# TRB / Nature widths (inches)
COL_W, MID_W, FULL_W = 3.46, 5.0, 7.16


def _pick_sans():
    for name in ("Arial", "Helvetica", "Helvetica Neue", "Nimbus Sans", "DejaVu Sans"):
        try:
            font_manager.findfont(name, fallback_to_default=False)
            return name
        except Exception:
            continue
    return "DejaVu Sans"


def setup():
    fam = _pick_sans()
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 600,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.015,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none",
        "font.family": "sans-serif", "font.sans-serif": [fam, "DejaVu Sans"],
        "font.size": 8.2, "text.color": C["ink"],
        "axes.edgecolor": C["ink"], "axes.labelcolor": C["ink"],
        "axes.titlesize": 9.2, "axes.titleweight": "regular", "axes.titlecolor": C["ink"],
        "axes.labelsize": 8.8, "axes.labelweight": "bold",
        "xtick.labelsize": 7.8, "ytick.labelsize": 7.8,
        "xtick.color": C["ink"], "ytick.color": C["ink"],
        "legend.fontsize": 7.6, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.major.size": 2.6, "ytick.major.size": 2.6,
        "xtick.direction": "out", "ytick.direction": "out",
        "lines.linewidth": 1.4, "lines.markersize": 4,
        "lines.solid_capstyle": "round",
        "grid.color": C["grid"], "grid.linewidth": 0.5,
        "patch.linewidth": 0.7,
    })


def panel_label(ax, letter, dx=-0.02, dy=1.0, fontsize=9):
    """Nature-style lowercase bold panel letter, in figure-relative position."""
    ax.annotate(letter, xy=(dx, dy), xycoords="axes fraction",
                fontsize=fontsize, fontweight="bold", va="bottom", ha="right",
                color=C["ink"], annotation_clip=False)


def _capitalize(text):
    """Use a capital first letter without changing approved scientific terms."""
    return text[:1].upper() + text[1:] if text else text


def title(ax, text, pad=PANEL_TITLE_PAD):
    """Use a short, left-aligned academic title for a subplot."""
    ax.set_title(_capitalize(text), fontsize=9.2, fontweight="regular", color=C["ink"],
                 pad=pad, loc="left")


def panel_title(ax, letter, text, pad=PANEL_TITLE_PAD):
    """Put a panel letter and its academic title on one aligned baseline."""
    ax.set_title(f"({letter}) {_capitalize(text)}", fontsize=9.2, fontweight="regular",
                 color=C["ink"], pad=pad, loc="left")


def axis(ax, xlabel=None, ylabel=None, xpad=AXIS_LABEL_PAD, ypad=AXIS_LABEL_PAD):
    """Bold, first-letter-uppercase axis labels with consistent padding."""
    if xlabel is not None:
        ax.set_xlabel(_capitalize(xlabel), labelpad=xpad, fontweight="bold", fontsize=8.8)
    if ylabel is not None:
        ax.set_ylabel(_capitalize(ylabel), labelpad=ypad, fontweight="bold", fontsize=8.8)


def colorbar_label(colorbar, text):
    """Apply the shared capitalization, weight, and spacing to a colorbar label."""
    colorbar.set_label(_capitalize(text), fontsize=8.8, fontweight="bold", labelpad=AXIS_LABEL_PAD)


def despine(ax, left=True, bottom=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)


def faint_grid(ax, axis="y"):
    ax.grid(axis=axis, color=C["grid"], linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def save(fig, name, outdir=None, superseded=False):
    """Write the figure as vector PDF and 300 dpi PNG, and mirror the PDF into the paper build.

    `superseded=True` diverts a figure to figures/superseded/ and does NOT mirror it into
    paper/figures. The journal allows five figures, and four of the original set carry claims
    that were withdrawn after review; regenerating them into the main folder would silently
    reinstate them and break the venue check in Q1_numbers_audit.py.
    """
    import pathlib
    base = pathlib.Path(outdir) if outdir else (pathlib.Path(__file__).resolve().parent / "figures")
    d = base / "superseded" if superseded else base
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / f"{name}.pdf")
    fig.savefig(d / f"{name}.png", dpi=300)
    if not superseded:
        pd = pathlib.Path(__file__).resolve().parent.parent / "paper" / "figures"
        pd.mkdir(parents=True, exist_ok=True)
        fig.savefig(pd / f"{name}.pdf")
    plt.close(fig)
    print("figure ->", name)
