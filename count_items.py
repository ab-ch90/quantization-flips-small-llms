"""Print the number of items in each benchmark subset."""

import json, os

with open(os.path.join("results", "_bench_items.json"), encoding="utf-8") as f:
    bench_items = json.load(f)

print("Items per benchmark:")
for name in bench_items:
    print(f"  {name:16s} {len(bench_items[name])}")
