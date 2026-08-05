"""Report per-subject full-precision MMLU accuracy, ranked easiest to hardest."""

import json
import os
import re
from collections import defaultdict

RESULTS_DIR = "results"


def subject_of(item_id):
    # 'professional_law_45' -> 'professional_law';  'marketing_124' -> 'marketing'
    return re.sub(r"_\d+$", "", item_id)


with open(os.path.join(RESULTS_DIR, "_bench_items.json"), encoding="utf-8") as f:
    bench_items = json.load(f)

mmlu = bench_items["mmlu"]
keys = [it["answer"] for it in mmlu]
subjects = [subject_of(it["id"]) for it in mmlu]

# Sanity: show the subjects found and their item counts.
counts = defaultdict(int)
for s in subjects:
    counts[s] += 1
print("Subjects found (item counts):")
for s in sorted(counts):
    print(f"  {counts[s]:4d}  {s}")
print()

per_subject = defaultdict(list)   # subject -> [per-model accuracy]
models = []

for fn in os.listdir(RESULTS_DIR):
    if fn.startswith("_") or not fn.endswith(".json"):
        continue
    with open(os.path.join(RESULTS_DIR, fn), encoding="utf-8") as f:
        d = json.load(f)
    if d.get("precision") != "fp16" or d.get("benchmark") != "mmlu":
        continue
    models.append(d["model"])
    preds = d["preds"]
    correct = defaultdict(int)
    total = defaultdict(int)
    for p, k, s in zip(preds, keys, subjects):
        total[s] += 1
        if p == k:
            correct[s] += 1
    for s in total:
        per_subject[s].append(correct[s] / total[s])

print(f"Per-subject fp16 accuracy, averaged over {len(models)} models (MMLU):\n")
ranking = sorted(per_subject.items(),
                 key=lambda kv: sum(kv[1]) / len(kv[1]), reverse=True)
for rank, (s, accs) in enumerate(ranking, 1):
    print(f"  {rank}.  {sum(accs)/len(accs)*100:5.1f}%   {s}")

print("\n(top = easiest, bottom = hardest)")
print("\nCLAIM CHECK:")
print("  'marketing' should be near the TOP,")
print("  'professional_medicine' and 'professional_law' near the BOTTOM.")
