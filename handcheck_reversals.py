"""Print sample confident reversals (both precisions >=0.90, different answers) for inspection."""

import json
import os

RESULTS_DIR = "results"
MODEL = "Qwen/Qwen2.5-3B-Instruct"
BENCH = "mmlu"          # the cell with 78 confident reversals at int4
CONFIDENT = 0.90
SHOW = 10


def main():
    safe = MODEL.replace("/", "__")
    with open(os.path.join(RESULTS_DIR, "_bench_items.json"), encoding="utf-8") as f:
        items = json.load(f)[BENCH]
    with open(os.path.join(RESULTS_DIR, f"{safe}__fp16__{BENCH}.json"), encoding="utf-8") as f:
        fp16 = json.load(f)
    with open(os.path.join(RESULTS_DIR, f"{safe}__int4__{BENCH}.json"), encoding="utf-8") as f:
        int4 = json.load(f)

    shown = 0
    for i, item in enumerate(items):
        p16, c16 = fp16["preds"][i], fp16["confs"][i]
        p4, c4 = int4["preds"][i], int4["confs"][i]
        if p16 != p4 and c16 >= CONFIDENT and c4 >= CONFIDENT:
            print("=" * 70)
            print(item["question"][:300])
            for L in "ABCD":
                print(f"   {L}) {item['options'][L][:80]}")
            print(f"   fp16 -> {p16} (conf {c16:.2f})    int4 -> {p4} (conf {c4:.2f})"
                  f"    KEY: {item['answer']}")
            shown += 1
            if shown >= SHOW:
                break
    print("=" * 70)
    print(f"\nShowed {shown} confident reversals from {MODEL} on {BENCH}.")
    print("Confirm: each is a real change between two different confident answers")
    print("on a real question, not a parsing glitch.")


if __name__ == "__main__":
    main()
