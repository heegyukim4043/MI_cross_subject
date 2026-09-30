"""Fig. 1 draft: study design overview (schematic; no data plotted).

All counts and settings are taken from the analysis records (MI_test/progress.md):
development datasets Cho2017 52, Lee2019 54, BNCI2014-001 9, PhysionetMI 109 (224 subjects);
Dreyer2023 87 (pre-specified confirmation); Stieger2021 S21-S62, 41 analysed (sealed validation).
Provisional size 180 x 115 mm; CMPB figure requirements not yet verified.
Colors: Okabe-Ito; every box also carries a text label, so color is not the only cue.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent
C = {"data": "#56B4E9", "prep": "#E69F00", "rq": "#009E73", "ext": "#CC79A7", "tier": "#0072B2"}
plt.rcParams.update({"font.family": "Arial", "font.size": 7, "svg.fonttype": "none", "pdf.fonttype": 42})


def box(ax, x, y, w, h, text, color, title=None, fs=6.6, alpha=0.18):
    for fc, a in ((color, alpha), ("none", 1.0)):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
                                    facecolor=fc, alpha=a, edgecolor=color, linewidth=1.0))
    if title:
        ax.text(x + w / 2, y + h - 0.017, title, ha="center", va="top", fontsize=fs + 0.7, fontweight="bold")
        ax.text(x + w / 2, y + h - 0.05, text, ha="center", va="top", fontsize=fs, linespacing=1.35)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.35)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7,
                                 color="#444444", linewidth=0.8, shrinkA=0, shrinkB=0))


def main():
    fig = plt.figure(figsize=(180 / 25.4, 115 / 25.4))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    for x, t in ((0.12, "A  Data"), (0.395, "B  Harmonization and alignment"), (0.6675, "C  Research questions"),
                 (0.8925, "D  External tests")):
        ax.text(x, 0.975, t, ha="center", va="top", fontsize=8, fontweight="bold")

    # A: development datasets
    ax.text(0.12, 0.92, "Development (LODO and LOSO)", ha="center", va="top", fontsize=6.8, style="italic")
    ys = [0.795, 0.7, 0.605, 0.51]
    for y, (name, n) in zip(ys, (("Cho2017", 52), ("Lee2019", 54), ("BNCI2014-001", 9), ("PhysionetMI", 109))):
        box(ax, 0.03, y, 0.18, 0.075, f"{name}\n{n} subjects", C["data"])
    ax.text(0.12, 0.49, "224 subjects in total", ha="center", va="top", fontsize=6.6)

    # B: harmonization and alignment
    box(ax, 0.265, 0.745, 0.26, 0.15,
        "21 channels shared by all datasets\n8–30 Hz, 0.5–2.5 s after cue, 100 Hz\nvolts (Cho2017 unit-corrected), no ICA",
        C["prep"], title="Common preprocessing")
    box(ax, 0.265, 0.53, 0.26, 0.175,
        "SubjectEA, or DatasetEA → SubjectEA\n(label-free, transductive)\nregularization: scale-consistent\nvs absolute eigenvalue floor",
        C["prep"], title="Covariance alignment")
    for y in ys:
        arrow(ax, 0.215, y + 0.0375, 0.26, 0.82)
    arrow(ax, 0.395, 0.745, 0.395, 0.71)
    ax.text(0.395, 0.505, "Classifiers: CSP-LDA, TS-LR, MDM;\nCSPNet (5 seeds), EEGNet", ha="center", va="top", fontsize=6.4)

    # C: research questions
    rqs = (("RQ1  Loss decomposition", "numerical effect, alignment-removable\nshift, cross-dataset residual,\nlabel-dependent residual"),
           ("RQ2  Scale equivariance", "same EEG at $\\times10^{-6}$ to $\\times10^{6}$;\nregularizers; Riemannian controls"),
           ("RQ3  Dataset-level alignment", "DatasetEA → SubjectEA\nvs SubjectEA (equivalence, ±1 pp)"),
           ("RQ4  Target-data access", "unlabeled n, causal online,\nbatch transductive, few labels"))
    ry = [0.775, 0.625, 0.475, 0.325]
    for y, (t, d) in zip(ry, rqs):
        box(ax, 0.56, y, 0.215, 0.135, d, C["rq"], title=t, fs=6.2)
        arrow(ax, 0.53, 0.62, 0.555, y + 0.0675)

    # D: external tests
    box(ax, 0.805, 0.615, 0.175, 0.23,
        "87 subjects\nplan fixed before\ndata access\nH1–H6 (RQ1–RQ3,\nfew-shot check)", C["ext"], title="Dreyer2023", fs=6.3)
    box(ax, 0.805, 0.325, 0.175, 0.23,
        "S21–S62, session 1\n41 analysed\nrule frozen on\ndevelopment data\nA1–A3 (RQ4)", C["ext"], title="Stieger2021 (sealed)", fs=6.3)
    arrow(ax, 0.78, 0.8425, 0.8, 0.79); arrow(ax, 0.78, 0.6925, 0.8, 0.73); arrow(ax, 0.78, 0.5425, 0.8, 0.68)
    arrow(ax, 0.78, 0.3925, 0.8, 0.44)
    ax.text(0.8925, 0.598, "confirmation", ha="center", va="top", fontsize=6.2, style="italic")
    ax.text(0.8925, 0.308, "validation", ha="center", va="top", fontsize=6.2, style="italic")

    # E: target-access tiers
    ax.text(0.03, 0.235, "E  Target-data access tiers (RQ4)", ha="left", va="center", fontsize=8, fontweight="bold")
    tiers = ("source only", "unlabeled\nn = 5–100 trials", "causal online\n(trial by trial)",
             "batch transductive\n(all trials, AdaBN)", "few labeled trials\nk = 5–30 per class")
    x0, w, gap = 0.03, 0.166, 0.025
    for i, t in enumerate(tiers):
        x = x0 + i * (w + gap)
        box(ax, x, 0.085, w, 0.1, t, C["tier"], fs=6.4, alpha=0.08 + 0.07 * i)
        if i:
            arrow(ax, x - gap + 0.004, 0.135, x - 0.004, 0.135)
    ax.text(0.5, 0.045, "increasing target information →", ha="center", va="center", fontsize=6.6, color="#333333")

    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig1_study_design_draft.{ext}", dpi=600, facecolor="white")
    print("saved", OUT / "fig1_study_design_draft.pdf")


if __name__ == "__main__":
    main()
