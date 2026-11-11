"""Generate the figures and the results tables from the saved results."""

import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

RESULTS_DIR = sys.argv[1] if len(sys.argv) > 1 else "results"
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "figures"
os.makedirs(OUT_DIR, exist_ok=True)

rcParams.update({
    "font.size": 10,
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "0.35",
    "axes.labelcolor": "0.15",
    "xtick.color": "0.25",
    "ytick.color": "0.25",
    "figure.dpi": 150,
})

MODEL_ORDER = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "google/gemma-2-2b-it",
    "Qwen/Qwen2.5-3B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "microsoft/Phi-3.5-mini-instruct",
]
SHORT = {
    "Qwen/Qwen2.5-1.5B-Instruct": "Qwen2.5-1.5B",
    "google/gemma-2-2b-it": "Gemma-2-2B",
    "Qwen/Qwen2.5-3B-Instruct": "Qwen2.5-3B",
    "meta-llama/Llama-3.2-3B-Instruct": "Llama-3.2-3B",
    "microsoft/Phi-3.5-mini-instruct": "Phi-3.5-mini",
}
BENCHES = ["arc", "mmlu", "csqa"]
BENCH_LABEL = {"arc": "ARC-Challenge", "mmlu": "MMLU", "csqa": "CommonsenseQA"}

# Categorical slots 1-3 (validated together) for the three benchmarks.
BENCH_COLOR = {"arc": "#2a78d6", "mmlu": "#eb6834", "csqa": "#1baf7a"}
# Single-hue sequential ramp (blue), light to dark.
SEQ = ["#9ec5f4", "#3987e5", "#104281"]
GAP = dict(edgecolor="white", linewidth=1.0)

# Dutta et al. (arXiv:2407.09141v1), BnB W4A4, MMLU 5-shot.
# Flips range from their Table 5.
DUTTA_4BIT_MMLU_MIN = 5.65
DUTTA_4BIT_MMLU_MAX = 10.90
# Per model: (16-bit accuracy, change in accuracy, Flips) from Table 5, and
# (% of correct, % of incorrect answers that changed) from Table 2.
DUTTA_MMLU_4BIT = {
    "Llama2-7B":  (47.21, -1.51, 10.65, 12.8, 19.9),
    "Llama2-13B": (53.54, -1.31, 8.09, 8.7, 16.9),
    "Llama2-70B": (63.17, -0.78, 5.65, 5.0, 13.2),
    "Yi-6B":      (62.95, -2.07, 10.90, 10.3, 23.9),
    "Yi-34B":     (74.89, -1.57, 7.44, 6.0, 19.4),
}

# fp16 top-two margin bins (same edges as analyze_results.py).
MARGIN_BINS = [(0.0, 0.25), (0.25, 0.8), (0.8, 1.0001)]
MARGIN_LABEL = ["below 0.25", "0.25 to 0.80", "0.80 and above"]
PREC_LABEL = {"int8": "8-bit", "fp4": "4-bit FP4", "nf4": "4-bit NF4"}


def load():
    with open(os.path.join(RESULTS_DIR, "_bench_items.json"), encoding="utf-8") as f:
        bench_items = json.load(f)
    passes = {}
    for fn in os.listdir(RESULTS_DIR):
        if fn.startswith("_") or not fn.endswith(".json"):
            continue
        with open(os.path.join(RESULTS_DIR, fn), encoding="utf-8") as f:
            d = json.load(f)
        passes[(d["model"], d["precision"], d["benchmark"])] = d
    return bench_items, passes


def breakdown(base, quant, keys):
    """Counts of correct->incorrect, incorrect->correct, wrong->wrong, and of
    initially correct / incorrect items."""
    ci = ic = ww = nc = ni = 0
    for b, q, k in zip(base, quant, keys):
        if b == k:
            nc += 1
            ci += q != k
        else:
            ni += 1
            ic += q == k
            ww += (q != k) and (q != b)
    return ci, ic, ww, nc, ni


def acc(preds, keys):
    return sum(p == k for p, k in zip(preds, keys)) / len(keys)


def style_axis(ax):
    ax.yaxis.grid(True, color="0.9", linewidth=0.6)
    ax.set_axisbelow(True)


def main():
    bench_items, passes = load()
    keys = {b: [it["answer"] for it in bench_items[b]] for b in BENCHES}

    metrics = {}
    for m in MODEL_ORDER:
        for b in BENCHES:
            base = passes.get((m, "fp16", b))
            if base is None:
                continue
            n = len(keys[b])
            for p in ("int8", "fp4", "nf4"):
                q = passes.get((m, p, b))
                if q is None:
                    continue
                ci, ic, ww, nc, ni = breakdown(base["preds"], q["preds"], keys[b])
                metrics[(m, b, p)] = {
                    "flips": (ci + ic) / n, "allflips": (ci + ic + ww) / n,
                    "ci": ci, "ic": ic, "ww": ww, "nc": nc, "ni": ni,
                    "acc16": acc(base["preds"], keys[b]),
                    "accq": acc(q["preds"], keys[b]), "n": n,
                }

    # Figure 1: 4-bit MMLU Flips against the original range.
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    xs = np.arange(len(MODEL_ORDER))
    w = 0.38
    ax.axhspan(DUTTA_4BIT_MMLU_MIN, DUTTA_4BIT_MMLU_MAX, color="0.88",
               label="Dutta et al., 4-bit default (2023 models)")
    for j, (p, col) in enumerate((("fp4", "#104281"), ("nf4", "#9ec5f4"))):
        vals = [metrics[(m, "mmlu", p)]["flips"] * 100 for m in MODEL_ORDER]
        ax.bar(xs + (j - 0.5) * w, vals, w, color=col, label=f"This study, {PREC_LABEL[p]}", **GAP)
    ax.set_xticks(xs)
    ax.set_xticklabels([SHORT[m] for m in MODEL_ORDER], rotation=20, ha="right")
    style_axis(ax)
    ax.set_ylabel("Flips (%), 4-bit MMLU")
    ax.set_title("Reproduction", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.28), ncol=3)
    fig.savefig(os.path.join(OUT_DIR, "fig1_reproduction.pdf"), bbox_inches="tight")
    plt.close(fig)

    # Figure 2: recovery of changed incorrect answers against chance.
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    # Original recovery: (I->C as % of N) / (% of N that were incorrect and changed).
    dutta_rec = [100 * ((flips + dacc) / 2) / ((100 - a16) * pi / 100)
                 for a16, dacc, flips, _, pi in DUTTA_MMLU_4BIT.values()]
    ax.axhspan(min(dutta_rec), max(dutta_rec), color="0.88", label="Dutta et al., 4-bit MMLU")
    ax.axhline(100 / 3, color="0.3", linestyle="--", linewidth=1.2,
               label="Chance, four options")
    ax.axhline(25, color="0.3", linestyle=":", linewidth=1.4,
               label="Chance, five options (CommonsenseQA)")
    xs = np.arange(len(MODEL_ORDER))
    w = 0.26
    for j, b in enumerate(BENCHES):
        rec = [100 * metrics[(m, b, "fp4")]["ic"]
               / (metrics[(m, b, "fp4")]["ic"] + metrics[(m, b, "fp4")]["ww"])
               for m in MODEL_ORDER]
        ax.bar(xs + (j - 1) * w, rec, w, color=BENCH_COLOR[b], label=BENCH_LABEL[b], **GAP)
    ax.set_xticks(xs)
    ax.set_xticklabels([SHORT[m] for m in MODEL_ORDER], rotation=20, ha="right")
    ax.set_ylabel("Changed incorrect answers\nthat became correct (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Recovery under default 4-bit quantization", fontsize=11)
    style_axis(ax)
    h, lab = ax.get_legend_handles_labels()
    order = [0, 3, 1, 4, 2, 5]   # reference lines in the first row, benchmarks in the second
    ax.legend([h[i] for i in order], [lab[i] for i in order], frameon=False, fontsize=8,
              loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=3)
    fig.savefig(os.path.join(OUT_DIR, "fig2_recovery.pdf"), bbox_inches="tight")
    plt.close(fig)

    # Figure 3: dose-response, mean Flips over the three benchmarks.
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    xs = np.arange(len(MODEL_ORDER))
    w = 0.26
    for j, p in enumerate(("int8", "nf4", "fp4")):
        ys = [np.mean([metrics[(m, b, p)]["flips"] for b in BENCHES]) * 100 for m in MODEL_ORDER]
        ax.bar(xs + (j - 1) * w, ys, w, label=PREC_LABEL[p], color=SEQ[j], **GAP)
    ax.set_xticks(xs)
    ax.set_xticklabels([SHORT[m] for m in MODEL_ORDER], rotation=20, ha="right")
    ax.set_ylabel("Flips (%), mean over benchmarks")
    ax.set_title("Dose-response", fontsize=11)
    style_axis(ax)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(os.path.join(OUT_DIR, "fig3_dose_response.pdf"), bbox_inches="tight")
    plt.close(fig)

    # Figure 4: flip rate of initially correct answers by fp16 top-two margin,
    # default 4-bit, three benchmarks pooled.
    rates = {m: [] for m in MODEL_ORDER}
    for m in MODEL_ORDER:
        for lo, hi in MARGIN_BINS:
            tot = chg = 0
            for b in BENCHES:
                base = passes[(m, "fp16", b)]
                q = passes[(m, "fp4", b)]
                for f, x, pr, k in zip(base["preds"], q["preds"], base["probs"], keys[b]):
                    top2 = sorted(pr)[-2:]
                    if f == k and lo <= top2[1] - top2[0] < hi:
                        tot += 1
                        chg += x != k
            rates[m].append(100 * chg / tot)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    xs = np.arange(len(MODEL_ORDER))
    w = 0.26
    for j, lab in enumerate(MARGIN_LABEL):
        ys = [rates[m][j] for m in MODEL_ORDER]
        ax.bar(xs + (j - 1) * w, ys, w, color=SEQ[j], label=lab, **GAP)
        if j == len(MARGIN_LABEL) - 1:
            # Direct labels on the large-margin bars, which are short.
            for x, y in zip(xs + (j - 1) * w, ys):
                ax.text(x, y + 1.2, f"{y:.1f}", ha="center", va="bottom",
                        fontsize=7, color="0.2")
    ax.set_xticks(xs)
    ax.set_xticklabels([SHORT[m] for m in MODEL_ORDER], rotation=20, ha="right")
    ax.set_ylabel("Initially correct answers\nchanged at 4-bit (%)")
    ax.set_title("Flip rate by full-precision top margin", fontsize=11)
    style_axis(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper center", title="fp16 top margin",
              title_fontsize=8, bbox_to_anchor=(0.5, -0.3), ncol=3)
    fig.savefig(os.path.join(OUT_DIR, "fig4_margin_flip_rate.pdf"), bbox_inches="tight")
    plt.close(fig)

    # Table 1: full per-condition results.
    lines = []
    header = (f"{'Model':14s} {'Bench':14s} {'Prec':5s} {'fp16Acc':>8s} {'QAcc':>7s} "
              f"{'AccD':>6s} {'Flips':>7s} {'AllF':>7s} {'C->I':>5s} {'I->C':>5s} {'W->W':>5s}")
    lines.append(header)
    lines.append("-" * len(header))
    tex = ["% Requires \\usepackage{booktabs}.",
           "\\begin{tabular}{llrrrrrrrr}", "\\toprule",
           "Model & Benchmark & fp16 Acc & Q Acc & $\\Delta$Acc & Flips & AllFlips"
           " & C$\\to$I & I$\\to$C & W$\\to$W \\\\"]
    for p in ("int8", "fp4", "nf4"):
        tex += ["\\midrule",
                f"\\multicolumn{{10}}{{l}}{{\\emph{{{PREC_LABEL[p]}"
                + (", default (replication condition)" if p == "fp4" else "")
                + (", extension" if p == "nf4" else "") + "}} \\\\"]
        for m in MODEL_ORDER:
            for j, b in enumerate(BENCHES):
                mm = metrics.get((m, b, p))
                if not mm:
                    continue
                lines.append(
                    f"{SHORT[m]:14s} {BENCH_LABEL[b]:14s} {p:5s} "
                    f"{mm['acc16']*100:7.1f}% {mm['accq']*100:6.1f}% "
                    f"{(mm['accq']-mm['acc16'])*100:+5.1f} {mm['flips']*100:6.1f}% "
                    f"{mm['allflips']*100:6.1f}% {mm['ci']:5d} {mm['ic']:5d} {mm['ww']:5d}")
                tex.append(
                    f"{SHORT[m] if j == 0 else ''} & {BENCH_LABEL[b]} & {mm['acc16']*100:.1f} & "
                    f"{mm['accq']*100:.1f} & {(mm['accq']-mm['acc16'])*100:+.1f} & "
                    f"{mm['flips']*100:.1f} & {mm['allflips']*100:.1f} & "
                    f"{mm['ci']} & {mm['ic']} & {mm['ww']} \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(OUT_DIR, "table1_full_results.txt"), "w") as f:
        f.write("\n".join(lines))
    with open(os.path.join(OUT_DIR, "table1_full_results.tex"), "w") as f:
        f.write("\n".join(tex))

    # Table 2: directional comparison with the original on 4-bit MMLU.
    def row(name, a16, dacc, pc, pi, ratio, rec, chance):
        return (f"{name} & {a16:.1f} & {dacc:+.1f} & {pc:.1f} & {pi:.1f} & "
                f"{ratio:.2f} & {rec:.1f} & {chance:.2f} \\\\")
    t2 = ["% Requires \\usepackage{booktabs}.",
          "\\begin{tabular}{lrrrrrrr}", "\\toprule",
          "Model & fp16 Acc & $\\Delta$Acc & Correct chg. & Incorrect chg. & "
          "C$\\to$I : I$\\to$C & Recovery & Chance ratio \\\\",
          "\\midrule",
          "\\multicolumn{8}{l}{\\emph{Dutta et al., BnB W4A4, 5-shot, full MMLU}} \\\\"]
    for name, (a16, dacc, flips, pc, pi) in DUTTA_MMLU_4BIT.items():
        ci = (flips - dacc) / 2
        ic = (flips + dacc) / 2
        rec = 100 * ic / ((100 - a16) * pi / 100)
        t2.append(row(name, a16, dacc, pc, pi, ci / ic, rec, 3 * a16 / (100 - a16)))
    t2 += ["\\midrule",
           "\\multicolumn{8}{l}{\\emph{This study, bitsandbytes 4-bit default (FP4), zero-shot, 975-item subset}} \\\\"]
    pci = pic = pww = 0
    for m in MODEL_ORDER:
        mm = metrics[(m, "mmlu", "fp4")]
        pci += mm["ci"]; pic += mm["ic"]; pww += mm["ww"]
        a = mm["acc16"]
        t2.append(row(SHORT[m], a * 100, (mm["accq"] - a) * 100,
                      100 * mm["ci"] / mm["nc"], 100 * (mm["ic"] + mm["ww"]) / mm["ni"],
                      mm["ci"] / mm["ic"], 100 * mm["ic"] / (mm["ic"] + mm["ww"]),
                      3 * a / (1 - a)))
    t2.append(f"Pooled, FP4 & & & & & {pci/pic:.2f} & {100*pic/(pic+pww):.1f} & \\\\")
    nci = sum(metrics[(m, "mmlu", "nf4")]["ci"] for m in MODEL_ORDER)
    nic = sum(metrics[(m, "mmlu", "nf4")]["ic"] for m in MODEL_ORDER)
    nww = sum(metrics[(m, "mmlu", "nf4")]["ww"] for m in MODEL_ORDER)
    t2.append(f"Pooled, NF4 & & & & & {nci/nic:.2f} & {100*nic/(nic+nww):.1f} & \\\\")
    t2 += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(OUT_DIR, "table2_directional_comparison.tex"), "w") as f:
        f.write("\n".join(t2))

    print(f"Figures and tables written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
