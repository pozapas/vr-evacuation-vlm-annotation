"""Inline micro-plots for the Q1 tables (TikZ), in the manner of sparklines rather than fills.

Every glyph is drawn from data, sits on one text line, and uses ink and gray with a single accent,
so the table reads as a quantitative display and not as a colored card. Each helper returns a TikZ
snippet. Axis extents are fixed per glyph type so that glyphs in one column share a scale.

  rug          event positions of every unit on a common axis (e.g. alarm and cut per recording)
  strip        one dot per unit on an axis with its median marked
  ptci         point estimate with interval on a labeled axis, with optional reference lines
  hist         a histogram sparkline of unit values
  bars         a row of tiny bars, one per group, each filled to its proportion
  coverage     a vertical stacked bar of full / partial / absent counts (positioning table)

Colors come from palette.py through `preamble()`, so no hex value is typed here.
"""
import sys, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from palette import SLATE, EMBER, VIOLET, INK, SUB, WHISPER

W = 2.8          # default glyph width, cm
FONT = r"\rmfamily\fontsize{6.5}{7}\selectfont"   # Times, as the rest of the manuscript


def preamble():
    cols = dict(tvInk=INK, tvSub=SUB, tvFaint=WHISPER, tvSlate=SLATE, tvEmber=EMBER,
                tvViolet=VIOLET)
    return "\n".join(rf"\providecolor{{{k}}}{{HTML}}{{{v.lstrip('#')}}}" for k, v in cols.items())


def _wrap(body, baseline="-0.6ex"):
    return r"\tikz[baseline=" + baseline + r",x=1cm,y=1cm,line cap=round]{" + body + "}"


def _x(v, lo, hi, width):
    return (min(max(v, lo), hi) - lo) / (hi - lo) * width


def _axis(width, lo, hi, ticks=(), labels=True, y=0.0):
    s = rf"\draw[tvSub,line width=0.25pt] (0,{y}) -- ({width},{y});"
    for t in ticks:
        x = _x(t, lo, hi, width)
        s += rf"\draw[tvSub,line width=0.25pt] ({x:.3f},{y}) -- ({x:.3f},{y - 0.05});"
        if labels:
            s += (rf"\node[font={FONT},text=tvSub,anchor=north,inner sep=0.5pt] at ({x:.3f},{y - 0.05})"
                  rf" {{{t:g}}};")
    return s


def rug(rows, lo, hi, width=W, ticks=None, height=0.13):
    """rows: list of (values, color, label). One tick per unit, one row per event type."""
    s, n = "", len(rows)
    for k, (vals, col, lab) in enumerate(rows):
        y = (n - 1 - k) * (height + 0.05) + 0.06
        for v in vals:
            x = _x(v, lo, hi, width)
            s += rf"\draw[{col},line width=0.35pt,opacity=0.85] ({x:.3f},{y:.3f}) -- ({x:.3f},{y + height:.3f});"
        s += (rf"\node[font={FONT},text={col},anchor=west,inner sep=0.5pt] at ({width + 0.04},"
              rf"{y + height / 2:.3f}) {{{lab}}};")
    s += _axis(width, lo, hi, ticks or (lo, hi))
    return _wrap(s, "0.2ex")


def strip(vals, lo, hi, width=W, color="tvInk", ticks=None, unit="", r_dot=0.03, median=True):
    """One small dot per unit, jittered deterministically in two rows, median as a vertical bar."""
    s = ""
    order = np.argsort(vals)
    for r, i in enumerate(order):
        x = _x(vals[i], lo, hi, width)
        y = 0.09 + 0.07 * (r % 2)
        s += rf"\fill[{color},opacity=0.8] ({x:.3f},{y:.3f}) circle ({r_dot});"
    if median:
        m = _x(float(np.median(vals)), lo, hi, width)
        s += rf"\draw[tvInk,line width=0.7pt] ({m:.3f},0.03) -- ({m:.3f},0.25);"
    s += _axis(width, lo, hi, ticks or (lo, hi))
    if unit:
        s += rf"\node[font={FONT},text=tvSub,anchor=west,inner sep=0.5pt] at ({width + 0.04},0) {{{unit}}};"
    return _wrap(s, "0.2ex")


def ptci(est, lo_ci, hi_ci, lo, hi, width=W, color="tvInk", refs=(), ticks=None):
    """Point and interval on a fixed axis. refs: list of (value, label) drawn as faint verticals."""
    s = _axis(width, lo, hi, ticks or (lo, hi))
    for v, lab in refs:
        x = _x(v, lo, hi, width)
        s += rf"\draw[tvSub,line width=0.3pt,densely dotted] ({x:.3f},0) -- ({x:.3f},0.26);"
        if lab:
            s += rf"\node[font={FONT},text=tvSub,anchor=south,inner sep=0.3pt] at ({x:.3f},0.26) {{{lab}}};"
    a, b, e = (_x(v, lo, hi, width) for v in (lo_ci, hi_ci, est))
    s += rf"\draw[{color},line width=0.6pt] ({a:.3f},0.13) -- ({b:.3f},0.13);"
    s += rf"\draw[{color},line width=0.6pt] ({a:.3f},0.08) -- ({a:.3f},0.18);"
    s += rf"\draw[{color},line width=0.6pt] ({b:.3f},0.08) -- ({b:.3f},0.18);"
    s += rf"\fill[{color}] ({e:.3f},0.13) circle (0.045);"
    return _wrap(s, "0.2ex")


def point(est, lo, hi, width=W, color="tvInk", ticks=None, hollow=False):
    """A single estimate on the axis, for rows without an interval."""
    s = _axis(width, lo, hi, ticks or (lo, hi))
    e = _x(est, lo, hi, width)
    style = f"draw={color},fill=white,line width=0.6pt" if hollow else f"fill={color}"
    s += rf"\path[{style}] ({e:.3f},0.13) circle (0.045);"
    return _wrap(s, "0.2ex")


def hist(vals, lo, hi, bins, width=W, color="tvInk", ticks=None, height=0.3):
    counts, edges = np.histogram([v for v in vals if v == v], bins=bins, range=(lo, hi))
    top = max(counts.max(), 1)
    s = ""
    bw = width / bins
    for k, c in enumerate(counts):
        if c:
            h = c / top * height
            s += rf"\fill[{color},opacity=0.8] ({k * bw + 0.01:.3f},0) rectangle ({(k + 1) * bw - 0.01:.3f},{h:.3f});"
    s += _axis(width, lo, hi, ticks or (lo, hi))
    return _wrap(s, "0.2ex")


def bars(props, width=W, color="tvInk", height=0.28, labels=None):
    """One thin bar per group, filled to its proportion, with a faint frame for the missing part."""
    n = len(props)
    bw = width / n
    s = ""
    for k, p in enumerate(props):
        x0, x1 = k * bw + 0.02, (k + 1) * bw - 0.02
        s += rf"\draw[tvSub,line width=0.2pt] ({x0:.3f},0) rectangle ({x1:.3f},{height});"
        s += rf"\fill[{color}] ({x0:.3f},0) rectangle ({x1:.3f},{p * height:.3f});"
        if labels:
            s += (rf"\node[font={FONT},text=tvSub,anchor=north,inner sep=0.5pt] at ({(x0 + x1) / 2:.3f},0)"
                  rf" {{{labels[k]}}};")
    return _wrap(s, "0.2ex")


def coverage(full, partial, absent, height=0.55, width=0.16):
    """Vertical stacked bar for one criterion across prior studies: full at the bottom in ink,
    partial in gray, absent as an open frame, so an empty column reads as a gap at a glance."""
    n = full + partial + absent
    u = height / max(n, 1)
    s = rf"\draw[tvSub,line width=0.2pt] (0,0) rectangle ({width},{height});"
    if full:
        s += rf"\fill[tvInk] (0,0) rectangle ({width},{full * u:.3f});"
    if partial:
        s += rf"\fill[tvSub!55] (0,{full * u:.3f}) rectangle ({width},{(full + partial) * u:.3f});"
    return _wrap(s, "0pt")


def group(label, ncol):
    """A section label in small caps with space above, and no fill or rule."""
    return (r"\addlinespace[5pt]" "\n"
            rf"\multicolumn{{{ncol}}}{{@{{}}l}}{{\textbf{{\textit{{{label}}}}}}} \\[1pt]")


def stack(parts, width=W, height=0.16):
    """One 100% composition bar. parts: list of (share, label); shades step from ink to light."""
    shades = ["tvInk", "tvSub", "tvSub!45", "tvSub!25"]
    s, x = "", 0.0
    for k, (sh, lab) in enumerate(parts):
        w = sh * width
        s += rf"\fill[{shades[k % 4]}] ({x:.3f},0.06) rectangle ({x + w - 0.012:.3f},{0.06 + height:.3f});"
        # segments alternate labels above and below the bar so narrow ones cannot collide
        ya, anc = (0.08 + height, "south") if k % 2 else (0.05, "north")
        s += (rf"\node[font={FONT},text=tvSub,anchor={anc},inner sep=0.5pt] at ({x + w / 2:.3f},{ya:.3f})"
              rf" {{{lab} {100 * sh:.0f}\%}};")
        x += w
    return _wrap(s, "0.2ex")
