"""Generate the figures and the results table from the saved results."""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

RESULTS_DIR = "results"
OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

rcParams.update({
    "font.size": 10,
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
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

DUTTA_4BIT_MMLU_MIN = 8.08
DUTTA_4BIT_MMLU_MAX = 16.63


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
    ci = ic = ww = 0
    for b, q, k in zip(base, quant, keys):
        if b == q:
            continue
        if b == k and q != k:
            ci += 1
        elif b != k and q == k:
            ic += 1
        else:
            ww += 1
    return ci, ic, ww


def acc(preds, keys):
    return sum(p == k for p, k in zip(preds, keys)) / len(keys)


def get(passes, m, p, b):
    return passes.get((m, p, b))


def main():
    bench_items, passes = load()
    keys = {b: [it["answer"] for it in bench_items[b]] for b in BENCHES}

    metrics = {}
    for m in MODEL_ORDER:
        for b in BENCHES:
            base = get(passes, m, "fp16", b)
            if base is None:
                continue
            n = len(keys[b])
            a16 = acc(base["preds"], keys[b])
            for p in ("int8", "int4"):
                q = get(passes, m, p, b)
                if q is None:
                    continue
                ci, ic, ww = breakdown(base["preds"], q["preds"], keys[b])
                metrics[(m, b, p)] = {
                    "flips": (ci + ic) / n, "allflips": (ci + ic + ww) / n,
                    "ci": ci, "ic": ic, "ww": ww,
                    "acc16": a16, "accq": acc(q["preds"], keys[b]), "n": n,
                }

    # Figure 1: reproduction. Legend below plot.
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    present = [m for m in MODEL_ORDER if (m, "mmlu", "int4") in metrics]
    vals = [metrics[(m, "mmlu", "int4")]["flips"] * 100 for m in present]
    xs = np.arange(len(present))
    ax.axhspan(DUTTA_4BIT_MMLU_MIN, DUTTA_4BIT_MMLU_MAX, color="0.85",
               label="Dutta et al. range (2023 models)")
    ax.bar(xs, vals, color="#3b6ea5", width=0.6, label="This study (2025 models)")
    ax.set_xticks(xs)
    ax.set_xticklabels([SHORT[m] for m in present], rotation=20, ha="right")
    ax.set_ylabel("Flips (%), 4-bit MMLU")
    ax.set_title("Reproduction", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.28), ncol=2)
    fig.savefig(os.path.join(OUT_DIR, "fig1_reproduction.pdf"), bbox_inches="tight")
    plt.close(fig)

    # Figure 2: directional imbalance.
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ci_means, ic_means, present2 = [], [], []
    for m in MODEL_ORDER:
        cis = [metrics[(m, b, "int4")]["ci"] for b in BENCHES if (m, b, "int4") in metrics]
        ics = [metrics[(m, b, "int4")]["ic"] for b in BENCHES if (m, b, "int4") in metrics]
        if cis:
            present2.append(m)
            ci_means.append(np.mean(cis))
            ic_means.append(np.mean(ics))
    xs = np.arange(len(present2))
    w = 0.38
    ax.bar(xs - w/2, ci_means, w, label="correct $\\rightarrow$ incorrect", color="#b5463f")
    ax.bar(xs + w/2, ic_means, w, label="incorrect $\\rightarrow$ correct", color="#3b6ea5")
    ax.set_xticks(xs)
    ax.set_xticklabels([SHORT[m] for m in present2], rotation=20, ha="right")
    ax.set_ylabel("Mean flip count per benchmark")
    ax.set_title("Directional imbalance (4-bit)", fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(os.path.join(OUT_DIR, "fig2_directional_imbalance.pdf"), bbox_inches="tight")
    plt.close(fig)

    # Figure 3: dose-response.
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    present3, f8, f4 = [], [], []
    for m in MODEL_ORDER:
        e8 = [metrics[(m, b, "int8")]["flips"] for b in BENCHES if (m, b, "int8") in metrics]
        e4 = [metrics[(m, b, "int4")]["flips"] for b in BENCHES if (m, b, "int4") in metrics]
        if e8 and e4:
            present3.append(m)
            f8.append(np.mean(e8) * 100)
            f4.append(np.mean(e4) * 100)
    xs = np.arange(len(present3))
    w = 0.38
    ax.bar(xs - w/2, f8, w, label="8-bit", color="#9ec3e0")
    ax.bar(xs + w/2, f4, w, label="4-bit", color="#3b6ea5")
    ax.set_xticks(xs)
    ax.set_xticklabels([SHORT[m] for m in present3], rotation=20, ha="right")
    ax.set_ylabel("Flips (%), mean over benchmarks")
    ax.set_title("Dose-response", fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(os.path.join(OUT_DIR, "fig3_dose_response.pdf"), bbox_inches="tight")
    plt.close(fig)

    # Figure 4: confidence profile.
    m = "Qwen/Qwen2.5-1.5B-Instruct"
    base = get(passes, m, "fp16", "mmlu")
    q = get(passes, m, "int4", "mmlu")
    if base and q:
        k = keys["mmlu"]
        confs = [base["confs"][i] for i in range(len(k))
                 if base["preds"][i] == k[i] and q["preds"][i] != k[i]]
        fig, ax = plt.subplots(figsize=(6.5, 3.6))
        ax.hist(confs, bins=np.linspace(0.25, 1.0, 16), color="#b5463f", edgecolor="white")
        ax.axvline(0.9, color="0.3", linestyle="--", linewidth=1)
        ymax = ax.get_ylim()[1]
        ax.text(0.905, ymax * 0.9, "0.90", fontsize=8, color="0.3")
        ax.set_xlabel("Full-precision confidence on the destroyed correct answer")
        ax.set_ylabel("Count")
        ax.set_title("Confidence of destroyed correct answers", fontsize=11)
        fig.savefig(os.path.join(OUT_DIR, "fig4_confidence_profile.pdf"), bbox_inches="tight")
        plt.close(fig)

    # Table
    lines = []
    header = f"{'Model':14s} {'Bench':14s} {'Prec':5s} {'fp16Acc':>8s} {'QAcc':>7s} {'AccD':>6s} {'Flips':>7s} {'AllF':>7s} {'C->I':>5s} {'I->C':>5s} {'W->W':>5s}"
    lines.append(header)
    lines.append("-" * len(header))
    tex = ["% Requires \\usepackage{booktabs}.",
           "\\begin{tabular}{lllrrrrrrrr}", "\\toprule",
           "Model & Benchmark & Prec & fp16 Acc & Q Acc & $\\Delta$Acc & Flips & AllFlips & C$\\to$I & I$\\to$C & W$\\to$W \\\\",
           "\\midrule"]
    for m in MODEL_ORDER:
        for b in BENCHES:
            for p in ("int8", "int4"):
                mm = metrics.get((m, b, p))
                if not mm:
                    continue
                lines.append(
                    f"{SHORT[m]:14s} {BENCH_LABEL[b]:14s} {p:5s} "
                    f"{mm['acc16']*100:7.1f}% {mm['accq']*100:6.1f}% "
                    f"{(mm['accq']-mm['acc16'])*100:+5.1f} {mm['flips']*100:6.1f}% "
                    f"{mm['allflips']*100:6.1f}% {mm['ci']:5d} {mm['ic']:5d} {mm['ww']:5d}")
                pl = "8-bit" if p == "int8" else "4-bit"
                tex.append(
                    f"{SHORT[m]} & {BENCH_LABEL[b]} & {pl} & {mm['acc16']*100:.1f} & "
                    f"{mm['accq']*100:.1f} & {(mm['accq']-mm['acc16'])*100:+.1f} & "
                    f"{mm['flips']*100:.1f} & {mm['allflips']*100:.1f} & "
                    f"{mm['ci']} & {mm['ic']} & {mm['ww']} \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(OUT_DIR, "table1_full_results.txt"), "w") as f:
        f.write("\n".join(lines))
    with open(os.path.join(OUT_DIR, "table1_full_results.tex"), "w") as f:
        f.write("\n".join(tex))

    print("Figures and table regenerated in figures/.\n")
    print("SUGGESTED CAPTIONS (paste into \\caption{...}):\n")
    print("Fig1: Reproduction of the flips finding on current small models. Bars "
          "show the 4-bit Flips rate on MMLU per model; the shaded band marks the "
          "8.1 to 16.6 percent range Dutta et al. report for their 2023 models on "
          "the same metric. All five models fall in or near that range, with "
          "Phi-3.5-mini just below it.\n")
    print("Fig2: Directional imbalance of flips under 4-bit quantization, averaged "
          "over the three benchmarks. Bars show mean counts of correct-to-incorrect "
          "and incorrect-to-correct flips per benchmark. On every model except "
          "Gemma-2, quantization destroys more correct answers than it creates.\n")
    print("Fig3: Mean Flips rate over the three benchmarks at 8-bit and 4-bit. "
          "Heavier quantization produces more flips for every model.\n")
    print("Fig4: Full-precision confidence on the correct answers that 4-bit "
          "quantization changed to incorrect (Qwen2.5-1.5B, MMLU). Most destroyed "
          "correct answers were weakly held, a minority above 0.90 the full model "
          "was confident about.\n")


if __name__ == "__main__":
    main()
