"""Measure and archive the run-to-run determinism floor of the evaluation pipeline.

Usage:  python determinism_check.py [results_dir]        (default: results)

Run after run_experiment.py, which pins the revisions and writes the item sets.
Uses the first 300 ARC-Challenge items and the pinned model revisions.

  identical  For every model at fp16 and at fp4, four identical single-item
             passes with the same code path as run_experiment.py.
  batched    For Qwen2.5-3B at fp16, four passes with batch sizes 1, 8, 16, 4
             (left padding), which gives batch-dependent numerics a chance to
             change predictions.

Every pass is saved with its predictions and full option probabilities in
<results_dir>/determinism/, together with summary.json, so the reported floor
can be checked without rerunning anything. Model and precision combinations
whose output file already exists are skipped, and summary.json is rebuilt from
every file in the folder.
"""

import json
import os

import torch
from transformers import AutoTokenizer

from run_experiment import (ITEMS_FILE, MODELS, REVISIONS_FILE, RESULTS_DIR,
                            effective_config, environment, items_sha256,
                            letter_token_ids, load_model, render, run_pass)

N_ITEMS = 300
N_RUNS = 4
PRECISIONS = ["fp16", "fp4"]
BATCH_MODEL = "Qwen/Qwen2.5-3B-Instruct"
BATCH_SIZES = [1, 8, 16, 4]
OUT_DIR = os.path.join(RESULTS_DIR, "determinism")


def batched_pass(model, tok, items, letter_ids, batch_size):
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    out = {"ids": [], "preds": [], "probs": []}
    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        texts = [render(tok, it) for it in chunk]
        inputs = tok(texts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            last = model(**inputs).logits[:, -1, :].float()
        for r, it in enumerate(chunk):
            letters = list(it["options"])
            probs = torch.softmax(torch.stack([last[r, letter_ids[L]] for L in letters]).cpu(), 0)
            out["ids"].append(it["id"])
            out["preds"].append(letters[int(torch.argmax(probs))])
            out["probs"].append([float(p) for p in probs])
    return out


def compare(runs):
    """Items whose prediction differs across runs, and the largest probability difference."""
    n = len(runs[0]["preds"])
    changed = [runs[0]["ids"][i] for i in range(n)
               if len({r["preds"][i] for r in runs}) > 1]
    max_diff = max(abs(a - b)
                   for i in range(n)
                   for r in runs[1:]
                   for a, b in zip(runs[0]["probs"][i], r["probs"][i]))
    return {"n_items": n, "n_runs": len(runs), "items_changed": len(changed),
            "changed_ids": changed, "max_abs_prob_diff": max_diff}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(REVISIONS_FILE, encoding="utf-8") as f:
        revs = json.load(f)
    with open(ITEMS_FILE, encoding="utf-8") as f:
        items = json.load(f)["arc"][:N_ITEMS]
    for name in MODELS:
        rev = revs["models"][name]
        tok = AutoTokenizer.from_pretrained(name, revision=rev)
        letter_ids = letter_token_ids(tok)
        for precision in PRECISIONS:
            stem = f"{name.replace('/', '__')}__{precision}__identical.json"
            if os.path.exists(os.path.join(OUT_DIR, stem)):
                print(f"{name} [{precision}] done, skipping")
                continue
            print(f"{name} [{precision}]")
            model = load_model(name, rev, precision)
            model.eval()
            conditions = {"identical": [run_pass(model, tok, items, letter_ids)
                                        for _ in range(N_RUNS)]}
            if name == BATCH_MODEL and precision == "fp16":
                conditions["batched"] = [batched_pass(model, tok, items, letter_ids, bs)
                                         for bs in BATCH_SIZES]
            for cond, runs in conditions.items():
                stats = compare(runs)
                row = {"model": name, "model_revision": rev, "precision": precision,
                       "condition": cond, "items_sha256": items_sha256(items),
                       "environment": environment(), **stats}
                if cond == "batched":
                    row["batch_sizes"] = BATCH_SIZES
                fn = f"{name.replace('/', '__')}__{precision}__{cond}.json"
                with open(os.path.join(OUT_DIR, fn), "w", encoding="utf-8") as f:
                    json.dump({**row, **effective_config(model), "runs": runs}, f)
                print(f"  {cond}: {stats['items_changed']}/{stats['n_items']} items changed, "
                      f"max prob diff {stats['max_abs_prob_diff']:.2e}")
            del model
            torch.cuda.empty_cache()

    rows = []
    for fn in sorted(os.listdir(OUT_DIR)):
        if fn.endswith(".json") and fn != "summary.json":
            with open(os.path.join(OUT_DIR, fn), encoding="utf-8") as f:
                d = json.load(f)
            rows.append({k: v for k, v in d.items()
                         if k not in ("runs", "quantization_config", "parameter_dtypes")})
    with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"results": rows}, f, indent=2)
    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
