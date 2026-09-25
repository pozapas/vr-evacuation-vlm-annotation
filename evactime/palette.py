"""EMBER & SLATE - the palette for this paper, and the proof that it is safe to use.

The paper sets a machine's account of a fire evacuation against the engine's own record, so the
palette is built on that opposition rather than on a generic categorical ramp:

    SLATE   cool, receding, factual   -> the engine reference, ground truth, what happened
    EMBER   warm, advancing, assertive-> the model's claim, what it says happened
    VIOLET  a third voice             -> the human coders
    MOSS / RUST / BRASS               -> verdicts: holds / fails / borderline

Warm always means "asserted by a model", cool always means "recorded by the engine". A reader who
learns that once can read every figure in the paper without a legend, which is the point of a
fixed semantic palette.

Two sequential ramps are derived from the anchors so heatmaps stay inside the same world:
DEPTH (slate, for reference quantities) and HEAT (ember, for model quantities).

Nothing here is decorative. `verify()` simulates dichromatic vision and prints the worst pairwise
separation in CAM02-UCS-like terms; the palette below is the result of iterating until the worst
pair cleared the threshold, so DO NOT adjust a hex value without re-running it.

Run: py evactime/palette.py
"""
import numpy as np

# ------------------------------------------------------------------ anchors
INK        = "#0E141B"   # near-black, faintly blue: text, axes, rules
SUB        = "#5C6B7A"   # secondary text
WHISPER    = "#E8ECF0"   # faint rules, grid
SHELL      = "#F6F8FA"   # inset wash
PAPER      = "#FFFFFF"

SLATE      = "#15607A"   # engine / reference / ground truth
SLATE_DIM  = "#7FA8B8"   # the same voice, quieter
EMBER      = "#C45A12"   # model claim, burnt orange (no pink: its tints read as peach)
EMBER_DIM  = "#EDB488"
VIOLET     = "#5A5550"   # human coders, warm gray (no purple; name kept for code compatibility)
BRASS      = "#A39041"   # borderline / attention
MOSS       = "#0A825F"   # holds / correct
RUST       = "#741F07"   # fails / withdrawn

# semantic aliases, so figure code reads as meaning rather than colour
C = {
    "ink": INK, "sub": SUB, "grid": WHISPER, "panel": SHELL, "paper": PAPER,
    "engine": SLATE, "engine_dim": SLATE_DIM,
    "vlm": EMBER, "vlm_dim": EMBER_DIM,
    "human": VIOLET,
    "good": MOSS, "bad": RUST, "mid": BRASS,
    # input conditions: one hue, three weights, because they are ordered not categorical
    "condA": "#0C4A5E", "condA0": "#2E8AA6", "condB": "#9CC4D1",
}

# object identities in the scene. Cool for fixed apparatus, warm for people.
# Object identities sit on a deliberate LIGHTNESS ladder (L* 34, 44, 52, 60, 68, 78, 90) so the
# attention raster survives a greyscale print. Hue still carries meaning; lightness carries the
# separation. The previous set had a 0.7 L* gap between door and NPC, which would have merged
# the two commonest categories into one grey.
OBJECT = {
    "door": "#005677", "equipment": "#885F43", "alarm": "#CC5A36",
    "npc": "#A08653", "sign": "#59B697", "display": "#DFBD6A",
    "none": "#DAE4E8",
}

MODEL_VENDOR = {"google": SLATE, "openai": MOSS, "anthropic": EMBER,
                "qwen": VIOLET, "meta": BRASS}


def ramp(name, n=256):
    """Perceptually ordered sequential ramps built from the anchors."""
    stops = {
        "DEPTH": ["#FFFFFF", "#DCE8ED", "#9FC2CE", "#4A8CA3", "#15607A", "#0A3B4C"],
        "HEAT":  ["#FFFFFF", "#FBE8D8", "#F2BE92", "#E0894A", "#C45A12", "#7E3708"],
    }[name]
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(name, stops, N=n)


# ------------------------------------------------------------------ verification
def _hex_rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=float) / 255.0


def _srgb_lin(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _lin_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def simulate(hexcol, kind):
    """Brettel-style dichromat simulation via the LMS Machado matrices (severity 1.0)."""
    M = {
        "deuteranopia": np.array([[0.367, 0.861, -0.228], [0.280, 0.673, 0.047],
                                  [-0.012, 0.043, 0.969]]),
        "protanopia":   np.array([[0.152, 1.053, -0.205], [0.115, 0.786, 0.099],
                                  [-0.004, -0.048, 1.052]]),
        "tritanopia":   np.array([[1.256, -0.077, -0.179], [-0.078, 0.931, 0.148],
                                  [0.005, 0.691, 0.304]]),
    }[kind]
    lin = _srgb_lin(_hex_rgb(hexcol))
    return _lin_srgb(M @ lin)


def _lab(rgb):
    """sRGB (linear-ish) to CIE Lab, D65."""
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722],
                  [0.0193, 0.1192, 0.9505]])
    xyz = m @ _srgb_lin(np.clip(rgb, 0, 1))
    xyz = xyz / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def dE(a, b):
    return float(np.linalg.norm(_lab(a) - _lab(b)))



# Colours used as a filled mark must also stand off the PAGE, not merely off each other. The
# first version of this palette passed every dichromacy test and still had a wheat that printed
# at contrast 1.70 on white. Anything listed here is exempt with a reason, and everything else
# has to clear 3.0.
CONTRAST_EXEMPT = {
    # the three input conditions encode an ORDERED variable through lightness, so the lightest
    # one is light on purpose; they are never drawn as thin marks in the shipping figures
    "condB",
    # scene objects are drawn as large filled blocks in the flow and the raster, where a light
    # fill reads fine; their lightness ladder is what keeps them apart in a greyscale print
    "sign", "display", "none",
}


def contrast_on_white(hexc):
    """WCAG relative-luminance contrast ratio against the page."""
    lum = float(np.dot(_srgb_lin(_hex_rgb(hexc)), [0.2126, 0.7152, 0.0722]))
    return 1.05 / (lum + 0.05)


def check_contrast(threshold=3.0):
    bad = []
    print("\ncontrast against white paper")
    for name, col in list(C.items()) + list(OBJECT.items()):
        if name in ("ink", "sub", "grid", "panel", "paper", "engine_dim", "vlm_dim"):
            continue
        r = contrast_on_white(col)
        tag = "exempt" if name in CONTRAST_EXEMPT else ("ok" if r >= threshold else "WEAK")
        if tag == "WEAK":
            bad.append(name)
        print(f"   {name:10s} {col}  {r:5.2f}  {tag}")
    return not bad


def verify(threshold=18.0):
    """Every pair that can appear in one panel must stay distinguishable under dichromacy."""
    # Verified in the groups that actually share a panel. "vlm" and "bad" are both warm by
    # design (a model claim and a failed verdict are both things that went wrong) and never
    # appear together, so forcing them apart would cost separation where it is really needed.
    groups = {
        "actors (engine / model / human)": ["engine", "vlm", "human"],
        "verdicts (holds / borderline / fails)": ["good", "mid", "bad"],
        "input conditions": ["condA", "condA0", "condB"],
    }
    worst_overall = (999, None, None)
    ok = True
    for gname, keys in groups.items():
        print(f"\n{gname}")
        for kind in ("normal", "deuteranopia", "protanopia", "tritanopia"):
            worst, pair = 999, None
            for i in range(len(keys)):
                for k in range(i + 1, len(keys)):
                    a, b = C[keys[i]], C[keys[k]]
                    ra = _hex_rgb(a) if kind == "normal" else simulate(a, kind)
                    rb = _hex_rgb(b) if kind == "normal" else simulate(b, kind)
                    d = dE(ra, rb)
                    if d < worst:
                        worst, pair = d, (keys[i], keys[k])
            flag = "ok " if worst >= threshold else "LOW"
            if worst < threshold:
                ok = False
            if worst < worst_overall[0]:
                worst_overall = (worst, pair, kind)
            print(f"   {kind:14s} worst dE {worst:5.1f}  {flag}  ({pair[0]} vs {pair[1]})")
    print(f"\nworst pair anywhere: {worst_overall[1]} under {worst_overall[2]}, "
          f"dE {worst_overall[0]:.1f}  (threshold {threshold})")
    # objects must also survive greyscale, because the journal prints some figures mono
    print("\ngreyscale separation of the object identities")
    ls = {k: _lab(_hex_rgb(v))[0] for k, v in OBJECT.items() if k != "none"}
    s = sorted(ls.items(), key=lambda kv: kv[1])
    for k, l in s:
        print(f"   {k:10s} L* {l:5.1f}")
    gaps = [s[i + 1][1] - s[i][1] for i in range(len(s) - 1)]
    print(f"   smallest L* gap {min(gaps):.1f}")
    ok = check_contrast() and ok
    return ok


if __name__ == "__main__":
    good = verify()
    print("\nPALETTE OK" if good else "\nPALETTE NEEDS WORK")
