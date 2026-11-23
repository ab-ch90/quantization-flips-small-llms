"""Add the rendered-prompt hash to result files written before run_experiment.py recorded it.

Usage:  python add_prompt_hashes.py [results_dir]        (default: results)

48 result files (every model except Llama-3.2) were written by the generator
archived as provenance/run_experiment_44f279bf.py, which did not yet record
prompts_sha256 or chat_date. For each such file, this script loads the
tokenizer at the pinned revision, renders every prompt exactly as that
generator did (no date argument) and as the current run_experiment.py does
(date fixed to CHAT_DATE), and checks that the two are identical item for
item. Only then does it add prompts_sha256, chat_date, and a prompt_hash_added
record to the file. Predictions, probabilities, and all other fields are left
unchanged. It stops on any mismatch. It needs only the tokenizers, not the
model weights or a GPU.
"""

import hashlib
import importlib.util
import json
import os
import sys

print("Loading libraries...", flush=True)
import run_experiment as current  # imports torch and bitsandbytes first, as in the main run
from transformers import AutoTokenizer

RESULTS_DIR = sys.argv[1] if len(sys.argv) > 1 else "results"
OLD_GENERATOR = os.path.join("provenance", "run_experiment_44f279bf.py")
OLD_SHA256 = "44f279bf6b940337d8fa02830869bf68f450689d20c54da3e07992a816dc86cc"


def file_sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_old_generator():
    if file_sha256(OLD_GENERATOR) != OLD_SHA256:
        sys.exit(f"STOP: {OLD_GENERATOR} does not match the recorded generator hash.")
    spec = importlib.util.spec_from_file_location("generator_44f279bf", OLD_GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    print("Loading the archived generator...", flush=True)
    old = load_old_generator()
    with open(current.REVISIONS_FILE, encoding="utf-8") as f:
        revs = json.load(f)
    with open(os.path.join(RESULTS_DIR, "_bench_items.json"), encoding="utf-8") as f:
        items = json.load(f)

    todo = []
    for fn in sorted(os.listdir(RESULTS_DIR)):
        if fn.startswith("_") or not fn.endswith(".json"):
            continue
        path = os.path.join(RESULTS_DIR, fn)
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        if "prompts_sha256" in d:
            continue
        if d["environment"]["script_sha256"] != OLD_SHA256:
            sys.exit(f"STOP: {fn} has no prompt hash and was not written by the archived generator.")
        if d["model_revision"] != revs["models"][d["model"]]:
            sys.exit(f"STOP: {fn} does not match the pinned model revision.")
        todo.append((path, d))
    print(f"{len(todo)} result files need a prompt hash", flush=True)

    hashes = {}
    for path, d in todo:
        key = (d["model"], d["benchmark"])
        if key not in hashes:
            tok = AutoTokenizer.from_pretrained(d["model"], revision=d["model_revision"])
            bench = items[d["benchmark"]]
            assert [it["id"] for it in bench] == d["ids"], f"{path}: item IDs do not match"
            for it in bench:
                old_text = tok.apply_chat_template(
                    [{"role": "user", "content": old.build_prompt(it)}],
                    tokenize=False, add_generation_prompt=True)
                if old_text != current.render(tok, it):
                    sys.exit(f"STOP: {d['model']} {d['benchmark']} item {it['id']} renders "
                             "differently under the archived and current generators.")
            hashes[key] = current.prompts_sha256(tok, bench)
            print(f"  {d['model']} {d['benchmark']}: {len(bench)} prompts identical, "
                  f"sha256 {hashes[key][:12]}", flush=True)
        d["chat_date"] = current.CHAT_DATE
        d["prompts_sha256"] = hashes[key]
        d["prompt_hash_added"] = {
            "by": "add_prompt_hashes.py",
            "generator_sha256": OLD_SHA256,
            "note": "Hash computed after the run from the pinned tokenizer; every prompt "
                    "rendered by the archived generator was checked to be identical to the "
                    "current render with chat_date fixed.",
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f)
    print(f"Added prompt hashes to {len(todo)} files")


if __name__ == "__main__":
    main()
