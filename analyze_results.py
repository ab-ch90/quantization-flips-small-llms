"""Compute every metric reported in the paper from the saved result files.

Usage:  python analyze_results.py [results_dir] [--previous OLD_RESULTS_DIR]

Prints, in order:
  1. Table 1: accuracy, Flips, AllFlips, and the directional counts for every
     model, benchmark, and quantized precision, with an exact McNemar test of
     the accuracy change (C1, C2)
  2. Dose-response, fp4 against int8 (C3)
  3. Direction of flips against a benchmark-specific chance baseline, per
     condition and pooled per benchmark, with 95% confidence intervals (C4)
  4. AllFlips relative to Flips on MMLU
  5. Flip rate of initially correct answers by fp16 top-two margin, a
     one-sided Mann-Whitney test that flipped answers had smaller margins than
     kept ones, the share of flips that came from answers with margin >= 0.8,
     and the mean margin of correct and incorrect answers (C5)
  6. Determinism: the archived summary from determinism_check.py, and, with
     --previous, agreement between this run and an earlier one

Definitions, for a benchmark with k options and fp16 accuracy a:
  Flips        (C->I + I->C) / N
  AllFlips     (C->I + I->C + W->W) / N
  recovery     I->C / (I->C + W->W), the share of changed incorrect answers that
               become correct; 1/(k-1) if a changed answer lands on a random
               other option
  chance ratio (k-1) a / (1-a), the C->I : I->C ratio expected if correct and
               incorrect answers changed at the same rate and landed at random
  margin       difference between the two largest option probabilities
"""

import argparse
import json
import os

import numpy as np
from scipy.stats import binomtest, mannwhitneyu

MODELS = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "google/gemma-2-2b-it",
    "Qwen/Qwen2.5-3B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "microsoft/Phi-3.5-mini-instruct",
]
BENCHES = ["arc", "mmlu", "csqa"]
QUANT = ["int8", "fp4", "nf4"]
MARGIN_BINS = [0.0, 0.25, 0.8, 1.0001]
N_BOOT = 2000


def short(m):
    name = m.split("/")[-1]
    for suffix in ("-Instruct", "-instruct", "-it"):
        name = name.removesuffix(suffix)
    return name


def wilson(x, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = x / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def load(results_dir):
    with open(os.path.join(results_dir, "_bench_items.json"), encoding="utf-8") as f:
        items = json.load(f)
    runs = {}
    for fn in os.listdir(results_dir):
        if fn.startswith("_") or not fn.endswith(".json"):
            continue
        with open(os.path.join(results_dir, fn), encoding="utf-8") as f:
            d = json.load(f)
        if "ids" in d:
            ids = [it["id"] for it in items[d["benchmark"]]]
            assert d["ids"] == ids, f"{fn}: item IDs do not match _bench_items.json"
        runs[(d["model"], d["precision"], d["benchmark"])] = d
    return items, runs


def transitions(base, quant, key):
    """Per-item labels: CC, CI, IC, WW (wrong to a different wrong), II (unchanged wrong)."""
    out = []
    for f, q, k in zip(base, quant, key):
        if f == k:
            out.append("CC" if q == k else "CI")
        elif q == k:
            out.append("IC")
        else:
            out.append("WW" if q != f else "II")
    return np.array(out)


def counts(t):
    return {s: int((t == s).sum()) for s in ("CC", "CI", "IC", "WW", "II")}


def boot_ratio(t, rng):
    """95% bootstrap interval for C->I / I->C, resampling items."""
    n = len(t)
    ci = (t == "CI").astype(int)
    ic = (t == "IC").astype(int)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    num, den = ci[idx].sum(1), ic[idx].sum(1)
    r = num[den > 0] / den[den > 0]
    return np.percentile(r, [2.5, 97.5])


def margins(probs):
    p = np.sort(np.asarray(probs), axis=1)
    return p[:, -1] - p[:, -2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", nargs="?", default="results")
    ap.add_argument("--previous", help="earlier results directory to compare predictions with")
    args = ap.parse_args()
    items, runs = load(args.results_dir)
    keys = {b: [it["answer"] for it in items[b]] for b in BENCHES}
    k_opts = {b: len(items[b][0]["options"]) for b in BENCHES}
    quant = [q for q in QUANT if any(p == q for (_, p, _) in runs)]
    rng = np.random.default_rng(0)

    T = {}
    for m in MODELS:
        for b in BENCHES:
            for q in quant:
                T[m, q, b] = transitions(runs[m, "fp16", b]["preds"], runs[m, q, b]["preds"], keys[b])

    # 1 ---------------------------------------------------------------------
    print("=" * 100)
    print("1. TABLE 1  (Flips and AllFlips in percent of N; McNemar exact two-sided p on C->I vs I->C)")
    print("=" * 100)
    print(f"{'model':14s}{'bench':6s}{'prec':6s}{'fp16':>7s}{'quant':>7s}{'dAcc':>7s}"
          f"{'Flips':>7s}{'AllF':>7s}{'C>I':>6s}{'I>C':>6s}{'W>W':>6s}{'McNemar p':>11s}")
    for q in quant:
        for m in MODELS:
            for b in BENCHES:
                c = counts(T[m, q, b])
                n = len(keys[b])
                a0 = (c["CC"] + c["CI"]) / n
                a1 = (c["CC"] + c["IC"]) / n
                p = binomtest(c["CI"], c["CI"] + c["IC"], 0.5).pvalue if c["CI"] + c["IC"] else 1.0
                print(f"{short(m):14s}{b:6s}{q:6s}{100*a0:7.1f}{100*a1:7.1f}{100*(a1-a0):7.1f}"
                      f"{100*(c['CI']+c['IC'])/n:7.1f}{100*(c['CI']+c['IC']+c['WW'])/n:7.1f}"
                      f"{c['CI']:6d}{c['IC']:6d}{c['WW']:6d}{p:11.1e}")
        print()

    # 2 ---------------------------------------------------------------------
    if "int8" in quant and "fp4" in quant:
        print("=" * 100)
        print("2. DOSE-RESPONSE  (Flips %, fp4 vs int8)")
        print("=" * 100)
        held = 0
        for m in MODELS:
            f8 = [100 * (counts(T[m, "int8", b])["CI"] + counts(T[m, "int8", b])["IC"]) / len(keys[b]) for b in BENCHES]
            f4 = [100 * (counts(T[m, "fp4", b])["CI"] + counts(T[m, "fp4", b])["IC"]) / len(keys[b]) for b in BENCHES]
            held += sum(x > y for x, y in zip(f4, f8))
            print(f"  {short(m):14s} int8 mean {np.mean(f8):5.1f}   fp4 mean {np.mean(f4):5.1f}   "
                  + "  ".join(f"{b} {y:.1f}->{x:.1f}" for b, x, y in zip(BENCHES, f4, f8)))
        print(f"  fp4 > int8 in {held}/{len(MODELS) * len(BENCHES)} model-benchmark conditions\n")

    # 3 ---------------------------------------------------------------------
    print("=" * 100)
    print("3. DIRECTION OF FLIPS  (chance recovery = 1/(k-1); chance ratio = (k-1)a/(1-a))")
    print("=" * 100)
    for q in quant:
        print(f"--- {q} ---")
        print(f"{'model':14s}{'bench':6s}{'k':>3s}{'acc':>6s}{'ratio':>7s}{'95% CI':>15s}{'chance':>8s}"
              f"{'recov%':>8s}{'95% CI':>15s}{'chance%':>9s}{'p':>10s}")
        above = sig = total = 0
        for b in BENCHES:
            k = k_opts[b]
            pool = {"CI": 0, "IC": 0, "WW": 0, "CC": 0, "II": 0}
            for m in MODELS:
                t = T[m, q, b]
                c = counts(t)
                for s in pool:
                    pool[s] += c[s]
                a = (c["CC"] + c["CI"]) / len(t)
                ch = c["IC"] + c["WW"]
                rec = c["IC"] / ch if ch else float("nan")
                lo, hi = wilson(c["IC"], ch)
                p = binomtest(c["IC"], ch, 1 / (k - 1), alternative="greater").pvalue if ch else 1.0
                rlo, rhi = boot_ratio(t, rng)
                ratio = c["CI"] / c["IC"] if c["IC"] else float("inf")
                total += 1
                above += rec > 1 / (k - 1)
                sig += p < 0.05
                print(f"{short(m):14s}{b:6s}{k:3d}{100*a:6.1f}{ratio:7.2f}  [{rlo:5.2f},{rhi:5.2f}]"
                      f"{(k-1)*a/(1-a):8.2f}{100*rec:8.1f}  [{100*lo:4.1f},{100*hi:4.1f}]"
                      f"{100/(k-1):9.1f}{p:10.1e}")
            ch = pool["IC"] + pool["WW"]
            lo, hi = wilson(pool["IC"], ch)
            p = binomtest(pool["IC"], ch, 1 / (k - 1), alternative="greater").pvalue
            nc, ni = pool["CC"] + pool["CI"], pool["IC"] + pool["WW"] + pool["II"]
            print(f"  pooled {b}: C>I {pool['CI']}, I>C {pool['IC']}, W>W {pool['WW']}, "
                  f"ratio {pool['CI']/pool['IC']:.2f}; correct changed {100*pool['CI']/nc:.1f}%, "
                  f"incorrect changed {100*ch/ni:.1f}%; recovery {100*pool['IC']/ch:.1f}% "
                  f"[{100*lo:.1f}, {100*hi:.1f}] vs {100/(k-1):.1f}%, p = {p:.1e}")
        print(f"  conditions with recovery above chance: {above}/{total}, significant at 0.05: {sig}/{total}\n")

    # 4 ---------------------------------------------------------------------
    print("=" * 100)
    print("4. AllFlips / Flips on MMLU")
    print("=" * 100)
    for q in quant:
        vals = []
        for m in MODELS:
            c = counts(T[m, q, "mmlu"])
            vals.append(f"{short(m)} {(c['CI']+c['IC']+c['WW'])/(c['CI']+c['IC']):.2f}")
        print(f"  {q}: " + ", ".join(vals))
    print()

    # 5 ---------------------------------------------------------------------
    if all("probs" in runs[m, "fp16", b] for m in MODELS for b in BENCHES):
        print("=" * 100)
        print("5. C5: flip rate of initially correct answers by fp16 top-two margin")
        print("=" * 100)
        labels = [f"[{lo:.2f},{min(hi, 1):.2f})" for lo, hi in zip(MARGIN_BINS[:-1], MARGIN_BINS[1:])]
        for q in quant:
            print(f"--- {q}, pooled over benchmarks ---")
            print(f"{'model':14s}" + "".join(f"{l:>18s}" for l in labels)
                  + "  falling  median margin flipped/kept  one-sided MWU p  flips from top bin")
            for m in MODELS:
                mg = np.concatenate([margins(runs[m, "fp16", b]["probs"]) for b in BENCHES])
                t = np.concatenate([T[m, q, b] for b in BENCHES])
                correct = np.isin(t, ["CC", "CI"])
                flipped = t == "CI"
                rates, cells = [], []
                for lo, hi in zip(MARGIN_BINS[:-1], MARGIN_BINS[1:]):
                    sel = correct & (mg >= lo) & (mg < hi)
                    n, x = int(sel.sum()), int(flipped[sel].sum())
                    rates.append(x / n)
                    cells.append(f"{x}/{n}={100*x/n:.1f}%")
                falling = all(a > b for a, b in zip(rates, rates[1:]))
                p = mannwhitneyu(mg[flipped], mg[correct & ~flipped], alternative="less").pvalue
                top = int((flipped & (mg >= MARGIN_BINS[-2])).sum())
                print(f"{short(m):14s}" + "".join(f"{c:>18s}" for c in cells)
                      + f"  {str(falling):7s}  {np.median(mg[flipped]):.2f}/{np.median(mg[correct & ~flipped]):.2f}"
                      + f"{p:>24.1e}  {top}/{int(flipped.sum())} = {100*top/flipped.sum():.0f}%")
            print()
        print("Mean fp16 top-two margin on MMLU, correct vs incorrect answers")
        for m in MODELS:
            d = runs[m, "fp16", "mmlu"]
            mg = margins(d["probs"])
            ok = np.array([p == k for p, k in zip(d["preds"], keys["mmlu"])])
            print(f"  {short(m):14s} correct {mg[ok].mean():.3f}   incorrect {mg[~ok].mean():.3f}")
        print()

    # 6 ---------------------------------------------------------------------
    print("=" * 100)
    print("6. DETERMINISM")
    print("=" * 100)
    summ = os.path.join(args.results_dir, "determinism", "summary.json")
    if os.path.exists(summ):
        with open(summ, encoding="utf-8") as f:
            for r in json.load(f)["results"]:
                print(f"  {short(r['model']):14s}{r['precision']:6s}{r['condition']:10s}"
                      f"{r['items_changed']}/{r['n_items']} items changed over {r['n_runs']} runs, "
                      f"max prob diff {r['max_abs_prob_diff']:.1e}")
    else:
        print("  no determinism/summary.json; run determinism_check.py")
    if args.previous:
        old_items, old = load(args.previous)
        print(f"\n  Agreement with {args.previous} (same items only)")
        pairs = [("fp16", "fp16"), ("int8", "int8"), ("nf4", "int4")]
        for new_p, old_p in pairs:
            for b in BENCHES:
                if [it["id"] for it in old_items.get(b, [])] != [it["id"] for it in items[b]]:
                    continue
                for m in MODELS:
                    if (m, new_p, b) in runs and (m, old_p, b) in old:
                        a, o = runs[m, new_p, b]["preds"], old[m, old_p, b]["preds"]
                        diff = sum(x != y for x, y in zip(a, o))
                        print(f"  {short(m):14s}{b:6s}{new_p:5s} vs old {old_p:5s} {diff} of {len(a)} differ")


if __name__ == "__main__":
    main()
