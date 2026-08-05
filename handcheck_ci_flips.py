"""Print sample correct-to-incorrect flips for manual inspection."""

import json
import os

RESULTS_DIR = "results"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
BENCH = "mmlu"
SHOW = 12


def main():
    safe = MODEL.replace("/", "__")
    with open(os.path.join(RESULTS_DIR, "_bench_items.json"), encoding="utf-8") as f:
        items = json.load(f)[BENCH]
    with open(os.path.join(RESULTS_DIR, f"{safe}__fp16__{BENCH}.json"), encoding="utf-8") as f:
        fp16 = json.load(f)
    with open(os.path.join(RESULTS_DIR, f"{safe}__int4__{BENCH}.json"), encoding="utf-8") as f:
        int4 = json.load(f)

    shown = 0
    ci_total = 0
    for i, item in enumerate(items):
        key = item["answer"]
        p16, c16 = fp16["preds"][i], fp16["confs"][i]
        p4, c4 = int4["preds"][i], int4["confs"][i]
        # correct -> incorrect
        if p16 == key and p4 != key:
            ci_total += 1
            if shown < SHOW:
                print("=" * 70)
                print(item["question"][:280])
                for L in "ABCD":
                    mark = ""
                    if L == key:
                        mark = "  <- correct"
                    print(f"   {L}) {item['options'][L][:75]}{mark}")
                print(f"   fp16 -> {p16} (conf {c16:.2f}) CORRECT   "
                      f"4bit -> {p4} (conf {c4:.2f}) WRONG")
                shown += 1

    print("=" * 70)
    print(f"\nShowed {shown} of {ci_total} correct->incorrect flips "
          f"({MODEL.split('/')[-1]}, {BENCH}, 4-bit).")
    print("Confirm each is a real question where fp16 got it right and 4-bit")
    print("got it wrong. If genuine, the directionality finding is solid.")

    # also report the confidence profile of these flips
    ci_confs_fp16 = []
    for i, item in enumerate(items):
        if fp16["preds"][i] == item["answer"] and int4["preds"][i] != item["answer"]:
            ci_confs_fp16.append(fp16["confs"][i])
    if ci_confs_fp16:
        hi = sum(1 for c in ci_confs_fp16 if c >= 0.9)
        print(f"\nOf the {len(ci_confs_fp16)} correct->incorrect flips, "
              f"{hi} had fp16 confidence >= 0.90.")
        print("These are cases where the full model was confident AND correct,")
        print("and quantization broke it. The higher this count, the more")
        print("striking the finding.")


if __name__ == "__main__":
    main()
