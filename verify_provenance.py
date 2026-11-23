"""Check the provenance metadata of every saved result file, without a GPU or network access.

Usage:  python verify_provenance.py [results_dir]        (default: results)

For every result file, checks that the model revision matches revisions.json,
that the item IDs and item hash match _bench_items.json, that the prediction
and probability lists have one entry per item, that a prompt hash and chat
date are recorded, and that the recorded script hash belongs to one of the
generator versions archived in provenance/. Prints which generator wrote which
files and exits with an error if any check fails.
"""

import hashlib
import json
import os
import sys
from collections import Counter

RESULTS_DIR = sys.argv[1] if len(sys.argv) > 1 else "results"


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def main():
    generators = {}
    for fn in sorted(os.listdir("provenance")):
        if fn.endswith(".py"):
            with open(os.path.join("provenance", fn), "rb") as f:
                generators[sha256_bytes(f.read())] = fn
    with open("revisions.json", encoding="utf-8") as f:
        revs = json.load(f)
    with open(os.path.join(RESULTS_DIR, "_bench_items.json"), encoding="utf-8") as f:
        items = json.load(f)
    item_hash = {b: sha256_bytes(json.dumps(v, sort_keys=True, ensure_ascii=False).encode("utf-8"))
                 for b, v in items.items()}

    problems, by_generator, added = [], Counter(), 0
    files = [fn for fn in sorted(os.listdir(RESULTS_DIR))
             if fn.endswith(".json") and not fn.startswith("_")]
    for fn in files:
        with open(os.path.join(RESULTS_DIR, fn), encoding="utf-8") as f:
            d = json.load(f)
        b, n = d["benchmark"], len(items[d["benchmark"]])
        checks = {
            "model revision": d.get("model_revision") == revs["models"][d["model"]],
            "dataset revision": d.get("dataset_revision") == revs["datasets"][d["dataset"]],
            "item hash": d.get("items_sha256") == item_hash[b],
            "item IDs": d.get("ids") == [it["id"] for it in items[b]],
            "entry counts": all(len(d.get(k, [])) == n for k in ("preds", "confs", "probs", "logits")),
            "prompt hash": bool(d.get("prompts_sha256")),
            "chat date": bool(d.get("chat_date")),
            "archived generator": d["environment"]["script_sha256"] in generators,
        }
        for name, ok in checks.items():
            if not ok:
                problems.append(f"{fn}: {name}")
        by_generator[generators.get(d["environment"]["script_sha256"], "not archived")] += 1
        added += "prompt_hash_added" in d

    print(f"{len(files)} result files checked")
    for gen, count in sorted(by_generator.items()):
        print(f"  {count} written by {gen}")
    print(f"  {added} with the prompt hash added afterwards by add_prompt_hashes.py")
    if problems:
        print("\nProblems:")
        for p in problems:
            print("  " + p)
        sys.exit(1)
    print("All checks passed")


if __name__ == "__main__":
    main()
