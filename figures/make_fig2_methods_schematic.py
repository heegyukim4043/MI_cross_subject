"""Fig. 2 (Methods): (a) design of the loss decomposition, (b) why an absolute eigenvalue floor is not scale-equivariant.

(a) Condition grid: transfer domain (LOSO within dataset, LODO across datasets) x alignment (none, absolute floor,
    relative floor), plus the label-informed reference; arrows mark the subject-paired contrasts of Eqs. (5)-(6).
(b) Eigenvalues of one Lee2019 subject reference (21 channels, float32, as stored in volts) and of the same data
    in microvolts (x1e6, eigenvalues x1e12). The absolute floor (1e-8) clips most eigenvalues in volts and none in
    microvolts; the relative floor (1e-8 * tr/C) moves with the data.
Output: fig2_methods_schematic_draft.pdf/png (180 x 80 mm).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # repository root
CH = "FC3 FC1 C1 C3 C5 CP3 CP1 P1 POz Pz CPz Fz FC4 FC2 Cz C2 C4 C6 CP4 CP2 P2".split()
OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00", "grey": "#555555"}
plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.linewidth": 0.6})


def box(ax, x, y, w, h, text, fc, ec):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.01,rounding_size=0.02",
                                fc=fc, ec=ec, lw=0.9))
    ax.text(x, y, text, ha="center", va="center", fontsize=6.6)


def arrow(ax, p, q, label, color, off=(0, 0), rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=0.9, color=color,
                                 connectionstyle=f"arc3,rad={rad}"))
    ax.text((p[0] + q[0]) / 2 + off[0], (p[1] + q[1]) / 2 + off[1], label, color=color, fontsize=7,
            ha="center", va="center")


def panel_a(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    xs = {"none": 0.2, "abs": 0.5, "rel": 0.8}
    ys = {"LOSO": 0.62, "LODO": 0.25}
    ax.text(0.5, 0.97, "alignment of each subject", ha="center", va="top", fontsize=7, style="italic")
    for k, lab in (("none", "none"), ("abs", "SubjectEA,\nabsolute floor"), ("rel", "SubjectEA,\nrelative floor")):
        ax.text(xs[k], 0.87, lab, ha="center", va="center", fontsize=6.6, fontweight="bold")
    for r, lab in (("LOSO", "within dataset\n(LOSO)"), ("LODO", "across datasets\n(LODO)")):
        ax.text(0.015, ys[r], lab, ha="left", va="center", fontsize=6.6, fontweight="bold", rotation=90)
        for k in xs:
            box(ax, xs[k], ys[r], 0.2, 0.13, "$B^{\\mathrm{%s}}$(%s)" % (r, {"none": "none", "abs": "abs", "rel": "rel"}[k]),
                "#EEF4FA", OI["blue"])
    # numerical contrast (absolute -> relative) and EA contrast (none -> relative, curved below), both rows
    for r in ys:
        arrow(ax, (xs["abs"] + 0.1, ys[r]), (xs["rel"] - 0.1, ys[r]), "", OI["red"])
        arrow(ax, (xs["none"], ys[r] - 0.065), (xs["rel"] - 0.02, ys[r] - 0.065), "", OI["green"], rad=0.25)
    ax.text(0.65, ys["LOSO"] + 0.03, r"$\Delta^{\mathrm{num}}$", color=OI["red"], ha="center", va="bottom")
    ax.text(0.5, ys["LOSO"] - 0.175, r"$\Delta^{\mathrm{EA}}$", color=OI["green"], ha="center")
    # cross-dataset-specific contrast
    arrow(ax, (xs["rel"] + 0.06, ys["LODO"] + 0.065), (xs["rel"] + 0.06, ys["LOSO"] - 0.065), "", OI["orange"])
    ax.text(xs["rel"] + 0.08, (ys["LOSO"] + ys["LODO"]) / 2 + 0.02, r"$\Delta^{\mathrm{cd}}$", color=OI["orange"])
    # label-informed reference (source + own labels)
    ax.text(0.5, 0.0, "label-informed reference: source + own labels, "
            r"$\Delta^{\mathrm{lab}} = B^{\mathrm{src+own}} - B^{\mathrm{LOSO}}(\mathrm{rel})$", ha="center", va="bottom",
            fontsize=5.8, color=OI["grey"])
    ax.text(-0.02, 1.0, "a", fontsize=9, fontweight="bold", transform=ax.transAxes, va="top")


def panel_b(ax):
    d = np.load(ROOT / "MI_test" / "preprocessed_sfreq100" / "lee2019.npz", allow_pickle=True)
    ch = [str(c) for c in d["ch_names"]]
    x = d["X"][d["subjects"] == d["subjects"][0]][:, [ch.index(c) for c in CH], :].astype(np.float32)
    R = np.einsum("nct,ndt->cd", x, x) / len(x)
    lam = np.sort(np.linalg.eigvalsh(R.astype(np.float64)))[::-1]
    k = np.arange(1, len(lam) + 1)
    for a, lab, col, mk in ((1.0, "volts", OI["blue"], "o"), (1e6, "microvolts (×10$^{6}$)", OI["orange"], "s")):
        l2 = lam * a ** 2
        rel = 1e-8 * l2.sum() / len(l2)
        ax.semilogy(k, l2, mk + "-", color=col, ms=2.6, lw=0.8, label=f"eigenvalues, {lab}")
        ax.axhline(rel, color=col, ls=":", lw=0.8)
        ax.text(21.3, rel, "relative floor", color=col, fontsize=5.8, va="center", ha="left")
        n_abs = int((l2 < 1e-8).sum())
        ax.text(1.2, l2[0] * 3, f"{n_abs} of 21 below absolute floor", color=col, fontsize=6.0, va="bottom")
    ax.axhline(1e-8, color="black", lw=0.9)
    ax.text(21.3, 1e-8, "absolute floor\n($10^{-8}$)", fontsize=5.8, va="center", ha="left")
    ax.set_ylim(1e-17, 1e7); ax.set_xlim(0.5, 21.5); ax.set_xlabel("eigenvalue index"); ax.set_ylabel("eigenvalue of $\\bar{R}_s$")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=6, loc="center right", bbox_to_anchor=(1.0, 0.62))
    ax.text(-0.2, 1.0, "b", fontsize=9, fontweight="bold", transform=ax.transAxes, va="top")


def main():
    fig = plt.figure(figsize=(180 / 25.4, 80 / 25.4))
    panel_a(fig.add_axes([0.01, 0.03, 0.5, 0.94]))
    panel_b(fig.add_axes([0.6, 0.15, 0.3, 0.8]))
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"fig2_methods_schematic_draft.{ext}", dpi=600 if ext == "png" else None)
    print("saved", HERE / "fig2_methods_schematic_draft.pdf")


if __name__ == "__main__":
    main()
