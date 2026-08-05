"""Compare results to Dutta et al. (2024) using their Flips and AllFlips definitions."""

import json
import os

RESULTS_DIR = "results"

# Dutta's reported numbers for reference (from their Tables 2, 5; MMLU 5-shot,
# Flips as %, for bitsandbytes BnB W8A8 and W4A4). These are their 2023 models.
# Format: model -> {"acc": fp16_acc, "bnb8_flip": x, "bnb4_flip": y}
DUTTA_MMLU = {
    "Llama2-7b-chat":  {"acc": 47.21, "bnb8_flip": 4.15, "bnb4_flip": 10.65},
    "Llama2-13b-chat": {"acc": 53.54, "bnb8_flip": 3.35, "bnb4_flip": 8.09},
    "Llama2-70b-chat": {"acc": 63.17, "bnb8_flip": 3.32, "bnb4_flip": 5.65},
    "Yi-6b-chat":      {"acc": 62.95, "bnb8_flip": 3.62, "bnb4_flip": 10.90},
    "Yi-34b-chat":     {"acc": 74.89, "bnb8_flip": 2.51, "bnb4_flip": 7.44},
}
# Dutta's key qualitative claims, to check against:
DUTTA_CLAIMS = [
    "C1: accuracy preserved within ~1-2% under quantization",
    "C2: flips are significant (>=5%) even when accuracy is preserved",
    "C3: lower-bit quantization produces more flips (4bit > 8bit)",
    "C4: incorrect answers flip more than correct (low top-margin items flip)",
    "C5: I->C and C->I are roughly balanced, keeping accuracy stable",
]


def load_all():
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


def flip_breakdown(base_preds, quant_preds, keys):
    """Return counts: correct->incorrect, incorrect->correct, wrong->wrong,
    and the total items."""
    ci = ic = ww = 0
    for b, q, k in zip(base_preds, quant_preds, keys):
        if b == q:
            continue
        b_correct = (b == k)
        q_correct = (q == k)
        if b_correct and not q_correct:
            ci += 1
        elif not b_correct and q_correct:
            ic += 1
        else:
            # both wrong, but different answers
            ww += 1
    return ci, ic, ww, len(keys)


def main():
    bench_items, passes = load_all()
    models = sorted({m for (m, _, _) in passes})
    benches = sorted({b for (_, _, b) in passes})

    print("=" * 72)
    print("YOUR CURRENT-GENERATION RESULTS (Dutta metric definitions)")
    print("=" * 72)

    summary = {}
    for model in models:
        for bench in benches:
            base = passes.get((model, "fp16", bench))
            if base is None:
                continue
            keys = [it["answer"] for it in bench_items[bench]]
            n = len(keys)
            acc16 = sum(p == k for p, k in zip(base["preds"], keys)) / n
            print(f"\n{model.split('/')[-1]}  |  {bench}  (N={n})")
            print(f"  fp16 accuracy: {acc16:.1%}")
            for prec in ("int8", "int4"):
                q = passes.get((model, prec, bench))
                if q is None:
                    continue
                accq = sum(p == k for p, k in zip(q["preds"], keys)) / n
                ci, ic, ww, _ = flip_breakdown(base["preds"], q["preds"], keys)
                flips = (ci + ic) / n          # Dutta's Flips
                allflips = (ci + ic + ww) / n  # Dutta's AllFlips
                label = "BnB W8A8" if prec == "int8" else "BnB W4A4"
                print(f"  [{label}] acc {accq:.1%} ({(accq-acc16)*100:+.1f})  "
                      f"Flips {flips:.1%}  AllFlips {allflips:.1%}  "
                      f"(C->I {ci}, I->C {ic}, W->W {ww})")
                summary[(model, bench, prec)] = {
                    "acc16": acc16, "accq": accq, "flips": flips,
                    "allflips": allflips, "ci": ci, "ic": ic, "ww": ww,
                }

    # Reference: Dutta's MMLU numbers
    print("\n" + "=" * 72)
    print("DUTTA REFERENCE (2023 models, MMLU 5-shot, Flips %)")
    print("=" * 72)
    print(f"  {'model':20s}{'fp16 acc':>10s}{'8bit flip':>11s}{'4bit flip':>11s}")
    for m, d in DUTTA_MMLU.items():
        print(f"  {m:20s}{d['acc']:>9.1f}%{d['bnb8_flip']:>10.1f}%{d['bnb4_flip']:>10.1f}%")

    # Claim-by-claim check (MMLU, aggregated)
    print("\n" + "=" * 72)
    print("CLAIM-BY-CLAIM CHECK (your current-gen models, MMLU)")
    print("=" * 72)
    mmlu_cells = [(m, b, p) for (m, b, p) in summary if b == "mmlu"]

    # C1: accuracy preserved within ~2%
    acc_changes = [abs(summary[c]["accq"] - summary[c]["acc16"]) for c in mmlu_cells]
    c1 = max(acc_changes) * 100 if acc_changes else 0
    print(f"\nC1 (acc preserved <~2%): max |acc change| = {c1:.1f} pts "
          f"-> {'HOLDS' if c1 <= 3 else 'DIVERGES'}")

    # C2: flips significant
    flips_vals = [summary[c]["flips"] for c in mmlu_cells if c[2] == "int4"]
    c2 = sum(flips_vals)/len(flips_vals) if flips_vals else 0
    print(f"C2 (flips significant, 4bit): mean Flips = {c2:.1%} "
          f"-> {'HOLDS' if c2 >= 0.05 else 'WEAKER than Dutta'}")

    # C3: 4bit > 8bit
    c3_holds = 0
    c3_total = 0
    for (m, b, p) in mmlu_cells:
        if p == "int4":
            c8 = summary.get((m, b, "int8"))
            if c8:
                c3_total += 1
                if summary[(m, b, p)]["flips"] > c8["flips"]:
                    c3_holds += 1
    print(f"C3 (4bit flips > 8bit): holds in {c3_holds}/{c3_total} models")

    # C5: I->C and C->I roughly balanced
    print("\nC5 (I->C vs C->I balance, 4bit MMLU):")
    for (m, b, p) in mmlu_cells:
        if p == "int4":
            s = summary[(m, b, p)]
            print(f"  {m.split('/')[-1]:24s} C->I {s['ci']:3d}  I->C {s['ic']:3d}  "
                  f"W->W {s['ww']:3d}")

    print("\n" + "=" * 72)
    print("WHAT TO LOOK FOR")
    print("=" * 72)
    print("""
  - Do your current-gen Flips land in a similar range to Dutta's (roughly
    5-11% at 4bit on MMLU)? If yes, the finding REPRODUCES. If your numbers
    are notably different, that DIFFERENCE is your extension's core result.

  - Is W->W a large share of total changes? Dutta downplayed it; if it's big
    on current models, that's a characterization the original lacked.

  - Are C->I and I->C balanced (C5)? If they're lopsided on distilled models,
    that's a generational difference worth reporting.

  This comparison is the reproduction result AND the seed of the extension.
""")


if __name__ == "__main__":
    main()
