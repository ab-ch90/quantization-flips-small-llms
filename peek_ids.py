"""Print sample MMLU item ids."""

import json, os

with open(os.path.join("results", "_bench_items.json"), encoding="utf-8") as f:
    bench_items = json.load(f)

mmlu = bench_items["mmlu"]
print("Total MMLU items:", len(mmlu))
print("\nFirst 5 ids:")
for it in mmlu[:5]:
    print("  ", repr(it["id"]))
print("\n5 ids sampled across the set:")
n = len(mmlu)
for i in (0, n // 4, n // 2, 3 * n // 4, n - 1):
    print("  ", repr(mmlu[i]["id"]))
