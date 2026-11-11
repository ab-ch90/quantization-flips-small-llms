"""
POSITION-BIAS CHECK

The Phi flips landed disproportionately on option D. This tests whether that is
a real systematic bias or just a pattern in the small flip set.

Reads the saved files from quant_gate.py (Qwen) and replication_check.py (Phi).
No model rerun.

Three tests per model:
  1. Full answer distribution at fp16 vs 4-bit across ALL 400 questions.
     If 4-bit picks D (or later options) more often everywhere, that is a
     systematic position bias, not just a flip-set quirk.
  2. Where the flips LAND (the destination distribution of changed answers).
  3. The key control: what is the distribution of the CORRECT answers? If the
     benchmark itself has more D answers, some D-drift could be coincidence.
     We compare 4-bit's D-rate to the true D-rate.

This tells us whether the paper is about answer CHANGE (solid) or answer change
WITH a directional position bias (sharper).
"""

import json
from collections import Counter

FILES = {
    "Qwen2.5-3B": "quant_gate_outputs.json",
    "Phi-3.5-mini": "replication_phi_outputs.json",
}


def analyze(name, path):
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
    except FileNotFoundError:
        print(f"  ({path} not found, skipping {name})")
        return

    items = d["items"]
    p16 = d["preds_fp16"]
    p4 = d["preds_4bit"]
    keys = [it["answer"] for it in items]
    n = len(items)

    dist16 = Counter(p16)
    dist4 = Counter(p4)
    distkey = Counter(keys)

    print("=" * 60)
    print(name)
    print("=" * 60)
    print(f"  {'option':8s}{'fp16':>8s}{'4bit':>8s}{'truth':>8s}")
    for L in "ABCD":
        print(f"  {L:8s}{dist16.get(L,0)/n:>7.1%}{dist4.get(L,0)/n:>8.1%}{distkey.get(L,0)/n:>8.1%}")

    # where flips land
    flip_dest = Counter(p4[i] for i in range(n) if p16[i] != p4[i])
    n_flip = sum(flip_dest.values())
    print(f"\n  flips: {n_flip}")
    if n_flip:
        print("  where 4-bit flips LAND:")
        for L in "ABCD":
            c = flip_dest.get(L, 0)
            print(f"    {L}: {c:3d}  ({c/n_flip:.0%} of flips)")

    # the test: did 4-bit's D-rate rise above fp16's, and above truth?
    d16 = dist16.get("D", 0) / n
    d4 = dist4.get("D", 0) / n
    dtrue = distkey.get("D", 0) / n
    print(f"\n  D-rate: fp16 {d16:.1%} -> 4bit {d4:.1%}  (truth {dtrue:.1%})")
    if d4 > d16 + 0.02:
        print("  -> 4-bit picks D MORE across all questions: systematic drift.")
    elif flip_dest.get("D", 0) > n_flip * 0.4:
        print("  -> D-drift concentrated in the FLIP set, not the whole dist.")
        print("     Subtler effect: flips prefer D but overall dist is stable.")
    else:
        print("  -> No clear D bias overall. The earlier D-cluster was small-sample.")
    print()


def main():
    print("\nDoes quantization systematically shift answers toward later options?\n")
    for name, path in FILES.items():
        analyze(name, path)
    print("=" * 60)
    print("READ")
    print("=" * 60)
    print("""
  If BOTH models show 4-bit picking later options more across all 400 questions,
  position bias is a real, general, sharp finding -> central to the paper.

  If only the flip-set leans D (overall distribution stable), the effect is real
  but subtler -> a section, not the headline.

  If the D-lean vanishes when you look at the whole distribution, it was
  small-sample noise in the flips -> drop it, keep the confident-reversal finding
  as the core. Same discipline: do not build on a pattern that does not survive
  looking at all the data.
""")


if __name__ == "__main__":
    main()
