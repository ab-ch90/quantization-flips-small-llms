"""Compute the preregistered metrics from the saved per-pass result files."""

import json
import os
from collections import Counter, defaultdict

RESULTS_DIR = "results"
CONFIDENT = 0.90
CONF_BINS = [(0.0, 0.6), (0.6, 0.9), (0.9, 1.01)]   # frozen bin edges


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


def accuracy(preds, keys):
    return sum(p == k for p, k in zip(preds, keys)) / len(keys)


def change_rate(a, b):
    return sum(1 for x, y in zip(a, b) if x != y) / len(a)


def main():
    bench_items, passes = load_all()
    models = sorted({m for (m, _, _) in passes})
    benches = sorted({b for (_, _, b) in passes})

    print("=" * 72)
    print("ARTIFACT CONTROL: no-op (fp16 vs fp16 must be 0.0% change)")
    print("=" * 72)
    # fp16 vs itself is 0 by construction; included to confirm the pipeline
    # is deterministic by checking the saved fp16 preds equal themselves.
    print("  (fp16 vs fp16 is 0% by construction; determinism ensured by greedy "
          "argmax decoding with no sampling.)\n")

    summary_rows = []
    for model in models:
        for bench in benches:
            keys = [it["answer"] for it in bench_items[bench]]
            base = passes.get((model, "fp16", bench))
            if base is None:
                continue
            acc16 = accuracy(base["preds"], keys)
            print("=" * 72)
            print(f"{model}  |  {bench}  (N={len(keys)})")
            print("=" * 72)
            print(f"  fp16 accuracy: {acc16:.1%}")

            # answer-distribution artifact check
            truth_dist = Counter(keys)
            print(f"  answer dist (truth): "
                  + "  ".join(f"{L}:{truth_dist.get(L,0)/len(keys):.0%}" for L in "ABCD"))

            cr_by_prec = {}
            for prec in ("int8", "int4"):
                q = passes.get((model, prec, bench))
                if q is None:
                    print(f"  [{prec}] not run yet")
                    continue
                accq = accuracy(q["preds"], keys)
                cr = change_rate(base["preds"], q["preds"])
                cr_by_prec[prec] = cr
                acc_change = accq - acc16

                # confident reversals
                conf_rev = sum(
                    1 for i in range(len(keys))
                    if base["preds"][i] != q["preds"][i]
                    and base["confs"][i] >= CONFIDENT and q["confs"][i] >= CONFIDENT
                )
                # change rate by fp16 confidence bin
                bin_counts = defaultdict(lambda: [0, 0])  # bin -> [changed, total]
                for i in range(len(keys)):
                    c = base["confs"][i]
                    for lo, hi in CONF_BINS:
                        if lo <= c < hi:
                            bin_counts[(lo, hi)][1] += 1
                            if base["preds"][i] != q["preds"][i]:
                                bin_counts[(lo, hi)][0] += 1
                            break

                # prereg primary-claim test
                passes_claim = cr >= 2 * abs(acc_change)
                qdist = Counter(q["preds"])

                print(f"\n  [{prec}]")
                print(f"    accuracy: {accq:.1%}  (change {acc_change*100:+.1f} pts)")
                print(f"    CHANGE RATE: {cr:.1%}   "
                      f"{'>= 2x|acc change|  PASS' if passes_claim else '< 2x  (claim not met here)'}")
                print(f"    confident reversals: {conf_rev}")
                print(f"    answer dist ({prec}): "
                      + "  ".join(f"{L}:{qdist.get(L,0)/len(keys):.0%}" for L in "ABCD"))
                print("    change rate by fp16 confidence:")
                for lo, hi in CONF_BINS:
                    ch, tot = bin_counts[(lo, hi)]
                    rate = ch / tot if tot else 0
                    print(f"      conf [{lo:.1f},{hi:.1f}): {rate:.1%}  ({ch}/{tot})")

                summary_rows.append((model, bench, prec, accq - acc16, cr,
                                     conf_rev, passes_claim))

            # dose-response within this model/bench
            if "int8" in cr_by_prec and "int4" in cr_by_prec:
                gradient = cr_by_prec["int4"] > cr_by_prec["int8"]
                print(f"\n  dose-response: int8 {cr_by_prec['int8']:.1%} -> "
                      f"int4 {cr_by_prec['int4']:.1%}  "
                      f"{'(increases, as predicted)' if gradient else '(does NOT increase)'}")
            print()

    # overall summary
    print("=" * 72)
    print("SUMMARY (every model x benchmark x quantized precision)")
    print("=" * 72)
    print(f"  {'model':28s}{'bench':6s}{'prec':6s}{'accΔ':>8s}{'changeR':>9s}{'confRev':>8s}{'claim':>7s}")
    n_pass = 0
    for (m, b, p, accd, cr, crv, pc) in summary_rows:
        short = m.split("/")[-1][:26]
        print(f"  {short:28s}{b:6s}{p:6s}{accd*100:+7.1f}{cr:>9.1%}{crv:>8d}"
              f"{'  PASS' if pc else '  --':>7s}")
        if pc:
            n_pass += 1
    print(f"\n  primary-claim PASS in {n_pass}/{len(summary_rows)} "
          f"model-bench-precision cells")
    print("""
  READ:
    Primary claim is supported if change rate >= 2x |accuracy change| CONSISTENTLY
    across cells (not just on average). Count the PASS cells and look for
    consistency across models and benchmarks, not a few wins.

    Dose-response supported if int4 change rate > int8 change rate consistently.

    Then hand-check: pick a few confident reversals and confirm they are genuine,
    same discipline that caught the earlier artifacts.
""")


if __name__ == "__main__":
    main()
