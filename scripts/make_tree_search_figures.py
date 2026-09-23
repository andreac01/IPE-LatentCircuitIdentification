import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = "/home/nicolobrunello/Documents/Projects/IPE-LatentCircuitIdentification"
SRC = os.path.join(ROOT, "experiments/faithfulness_completeness_v2")
IMG = os.path.join(ROOT, "img")
os.makedirs(IMG, exist_ok=True)

df = pd.read_csv(os.path.join(SRC, "faithfulness_completeness_v2.csv"))
cls = pd.read_csv(os.path.join(SRC, "incompleteness_by_class_v2.csv"))

F_MODEL = float(df[df.method == "full model"].iloc[0].F)
GT = df[df.method == "ground truth"].iloc[0]
EMPTY = df[df.method == "empty"].iloc[0]
df["gap"] = (df.F - F_MODEL).abs()

# Okabe-Ito, fixed order; identity carried by colour AND marker AND dash.
S = {"path":   dict(color="#0072B2", marker="o", ls="-",  label="path search"),
     "tree":   dict(color="#D55E00", marker="s", ls="-",  label="tree search"),
     "random": dict(color="#999999", marker="x", ls=":",  label="random, size-matched")}
GT_C, GT_M = "#009E73", "*"
GRID = dict(alpha=0.25, lw=0.6)
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "figure.dpi": 180})


def rows(m, nonempty=True):
    s = df[df.method == m]
    if nonempty:
        s = s[s.n_elements > 0]
    return s.sort_values("threshold")


def draw(ax, x, y, m, **kw):
    st = dict(S[m]); st.update(kw)
    ax.plot(x, y, lw=1.9, ms=8, markeredgewidth=1.2, **st)


# ---------------------------------------------------------------- Figure 1
fig, ax = plt.subplots(figsize=(7.2, 4.6))
for m in ("path", "tree"):
    s = rows(m)
    draw(ax, s.threshold, s.element_f1, m)
ax.axhline(1.0, color=GT_C, lw=1.4, ls="-.", label="Wang et al. (26/26 elements)")

dead = rows("path", nonempty=False)
dead = dead[dead.n_elements == 0]
if len(dead):
    t0 = float(dead.threshold.min())
    ax.axvspan(t0, float(df.threshold.max()), color=S["path"]["color"], alpha=0.07, lw=0)
    ax.annotate("path search returns\nno complete paths", xy=(t0 * 1.15, 0.30),
                fontsize=8, color=S["path"]["color"], ha="left")
ax.set_xscale("log")
ax.set_ylim(0, 1.05)
ax.set_xlabel("admission threshold  (min_contribution)")
ax.set_ylabel("element F1 vs. Wang et al.")
ax.set_title("Agreement with the ground truth is far more stable for the tree\n"
             "GPT-2 small / IOI, positional, (head, position) elements", loc="left")
ax.grid(**GRID)
ax.legend(fontsize=8, loc="lower left")
fig.tight_layout()
fig.savefig(os.path.join(IMG, "f1_vs_threshold.png"), bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- Figure 2
fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5))

ax = axes[0]
for m in ("path", "tree", "random"):
    s = rows(m)
    draw(ax, s.n_elements, s.gap, m)
ax.plot([GT.n_elements], [abs(GT.F - F_MODEL)], marker=GT_M, ms=17, ls="none",
        color=GT_C, label="Wang et al.")
ax.axhline(abs(EMPTY.F - F_MODEL), color="black", lw=1.1, ls="--",
           label=f"empty circuit ({abs(EMPTY.F - F_MODEL):.2f})")
ax.axhline(0.0, color="black", lw=1.1, ls=":", label="perfectly faithful")
ax.set_xscale("log")
ax.set_xlabel("elements in the circuit  (head, position)")
ax.set_ylabel(r"$|F(C) - F(M)|$   (logit difference)")
ax.set_title("Faithfulness gap — lower is better", loc="left")
ax.grid(**GRID)
ax.legend(fontsize=7.5)

ax = axes[1]
for m in ("path", "tree", "random"):
    s = rows(m).dropna(subset=["incompl_mean"])
    draw(ax, s.n_elements, s.incompl_mean, m)
ax.plot([GT.n_elements], [GT.incompl_mean], marker=GT_M, ms=17, ls="none",
        color=GT_C, label="Wang et al.")
ax.axhline(0.0, color="black", lw=1.1, ls=":", label="complete")
ax.set_xscale("log")
ax.set_xlabel("elements in the circuit  (head, position)")
ax.set_ylabel(r"$\mathbb{E}_K\,|F(C\setminus K) - F(M\setminus K)|$")
ax.set_title("Incompleteness — lower is better", loc="left")
ax.grid(**GRID)
ax.legend(fontsize=7.5)

fig.suptitle("Both axes in raw logit-difference units, Wang et al. protocol "
             "(MLPs intact, ABC mean-ablation)", fontsize=10, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(IMG, "gap_and_incompleteness.png"), bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- Figure 3
fig, ax = plt.subplots(figsize=(6.6, 6.2))
lo = min(cls["F(C\\K)"].min(), cls["F(M\\K)"].min()) - 0.8
hi = max(cls["F(C\\K)"].max(), cls["F(M\\K)"].max()) + 0.8
ax.plot([lo, hi], [lo, hi], color="#666666", lw=1.3, ls="--",
        label=r"complete:  $F(C\setminus K) = F(M\setminus K)$")
ax.scatter(cls["F(C\\K)"], cls["F(M\\K)"], s=150, marker="*", color=GT_C, zorder=3,
           label="Wang et al., one circuit class removed")

# hand-placed offsets: the three points near (4, 3) collide with the default placement
OFF = {"Negative Name Mover": (-132, -16), "none": (16, 18), "Name Mover": (10, 6),
       "Backup Name Mover": (-34, -22), "Previous Token": (-100, 0),
       "Duplicate Token": (10, -14), "Induction": (10, 4), "S-Inhibition": (10, -14)}
for _, r in cls.iterrows():
    ax.annotate(r["K"], (r["F(C\\K)"], r["F(M\\K)"]), fontsize=8, color="#333333",
                textcoords="offset points", xytext=OFF.get(r["K"], (8, 6)))
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel(r"$F(C\setminus K)$   — the broken circuit")
ax.set_ylabel(r"$F(M\setminus K)$   — the cobbled-together model")
ax.set_title("Completeness of the ground-truth circuit\n"
             "distance from the diagonal is incompleteness", loc="left")
ax.grid(**GRID)
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(IMG, "completeness_by_class.png"), bbox_inches="tight")
plt.close(fig)

print("wrote:")
for f in sorted(os.listdir(IMG)):
    print(f"  img/{f}  ({os.path.getsize(os.path.join(IMG, f)) // 1024} KB)")
