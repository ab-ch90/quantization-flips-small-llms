"""Run every (model, precision, benchmark) pass and save per-item predictions with full provenance.

Usage:  python run_experiment.py [results_dir]        (default: results)

Precisions
  fp16  16-bit reference
  int8  bitsandbytes LLM.int8() with library defaults
  fp4   bitsandbytes 4-bit with library defaults (FP4, float32 compute), the
        configuration used by Dutta et al. ("default parameters supported with
        HuggingFace"); this is the replication condition
  nf4   bitsandbytes 4-bit NF4 with float16 compute, reported as an extension

Benchmarks
  arc   ARC-Challenge test, first 1,000 items whose options are labelled A-D
  mmlu  MMLU test, eight fixed subjects, first 125 items of each (975 in total,
        because college_computer_science has only 100)
  csqa  CommonsenseQA validation, first 1,000 items, all five options A-E

Model and dataset revisions are resolved once and written to revisions.json.
Every later run loads exactly those revisions, so commit that file with the
results. Each result file records the revisions, a hash of the item set, the
item IDs beside the predictions, the full probability distribution over the
answer options, the effective quantization configuration, and the software
and hardware environment. Existing result files are skipped only if their
item hash, revisions, and prompt hash match the current ones; otherwise the
script stops. The chat date is fixed (see CHAT_DATE) so prompts do not depend on
the day of the run.
"""

import datetime
import hashlib
import json
import os
import platform
import sys
import time

import bitsandbytes
import datasets
import torch
import transformers
from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODELS = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "google/gemma-2-2b-it",
    "Qwen/Qwen2.5-3B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "microsoft/Phi-3.5-mini-instruct",
]
PRECISIONS = ["fp16", "int8", "fp4", "nf4"]
BENCHMARKS = ["arc", "mmlu", "csqa"]
DATASETS = {"arc": "allenai/ai2_arc", "mmlu": "cais/mmlu", "csqa": "tau/commonsense_qa"}
N_PER_BENCHMARK = 1000
MMLU_SUBJECTS = [
    "college_computer_science", "high_school_physics",
    "philosophy", "professional_law",
    "high_school_psychology", "high_school_macroeconomics",
    "professional_medicine", "marketing",
]

RESULTS_DIR = sys.argv[1] if len(sys.argv) > 1 else "results"
REVISIONS_FILE = "revisions.json"
ITEMS_FILE = os.path.join(RESULTS_DIR, "_bench_items.json")
# The Llama-3.2 chat template writes "Today Date: <current date>" into every
# prompt unless a date is supplied, which makes its predictions depend on the
# day of the run. Fixing the date to the template's own fallback value keeps
# prompts identical across runs. Templates without a date field ignore it.
CHAT_DATE = "26 Jul 2024"


# ---------------------------------------------------------------- provenance
def file_sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def items_sha256(items):
    blob = json.dumps(items, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def resolve_revisions():
    """Load pinned revisions, or resolve the current commit hashes and pin them."""
    if os.path.exists(REVISIONS_FILE):
        with open(REVISIONS_FILE, encoding="utf-8") as f:
            revs = json.load(f)
    else:
        revs = {"models": {}, "datasets": {}}
    api = HfApi()
    changed = False
    for m in MODELS:
        if m not in revs["models"]:
            revs["models"][m] = api.model_info(m).sha
            changed = True
    for b in BENCHMARKS:
        d = DATASETS[b]
        if d not in revs["datasets"]:
            revs["datasets"][d] = api.dataset_info(d).sha
            changed = True
    if changed:
        revs["resolved_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with open(REVISIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(revs, f, indent=2)
        print(f"Pinned revisions written to {REVISIONS_FILE}")
    return revs


def environment():
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0),
        "transformers": transformers.__version__,
        "bitsandbytes": bitsandbytes.__version__,
        "datasets": datasets.__version__,
        "script_sha256": file_sha256(os.path.abspath(__file__)),
    }


# ---------------------------------------------------------------- benchmarks
def _item(qid, question, options, answer):
    return {"id": qid, "question": question, "options": options, "answer": answer}


def load_arc(rev):
    ds = load_dataset(DATASETS["arc"], "ARC-Challenge", split="test", revision=rev)
    items = []
    for ex in ds:
        labels, texts = ex["choices"]["label"], ex["choices"]["text"]
        if labels != ["A", "B", "C", "D"]:
            continue
        items.append(_item(ex["id"], ex["question"], dict(zip(labels, texts)), ex["answerKey"]))
        if len(items) >= N_PER_BENCHMARK:
            break
    return items


def load_mmlu(rev):
    per = N_PER_BENCHMARK // len(MMLU_SUBJECTS)
    items = []
    for subj in MMLU_SUBJECTS:
        ds = load_dataset(DATASETS["mmlu"], subj, split="test", revision=rev)
        for i, ex in enumerate(ds):
            if i >= per:
                break
            opts = dict(zip("ABCD", ex["choices"]))
            items.append(_item(f"{subj}_{i}", ex["question"], opts, "ABCD"[ex["answer"]]))
    return items


def load_csqa(rev):
    ds = load_dataset(DATASETS["csqa"], split="validation", revision=rev)
    items = []
    for ex in ds:
        labels, texts = ex["choices"]["label"], ex["choices"]["text"]
        assert labels == ["A", "B", "C", "D", "E"], ex["id"]
        items.append(_item(ex["id"], ex["question"], dict(zip(labels, texts)), ex["answerKey"]))
        if len(items) >= N_PER_BENCHMARK:
            break
    return items


LOADERS = {"arc": load_arc, "mmlu": load_mmlu, "csqa": load_csqa}


def load_items(revs):
    """Build the item sets and check them against any saved copy."""
    fresh = {b: LOADERS[b](revs["datasets"][DATASETS[b]]) for b in BENCHMARKS}
    if os.path.exists(ITEMS_FILE):
        with open(ITEMS_FILE, encoding="utf-8") as f:
            saved = json.load(f)
        for b in BENCHMARKS:
            if b in saved and items_sha256(saved[b]) != items_sha256(fresh[b]):
                sys.exit(f"STOP: the {b} items differ from {ITEMS_FILE}. "
                         "Use a new results directory or restore the pinned revisions.")
    with open(ITEMS_FILE, "w", encoding="utf-8") as f:
        json.dump(fresh, f, ensure_ascii=False)
    for b in BENCHMARKS:
        print(f"  {b}: {len(fresh[b])} items, sha256 {items_sha256(fresh[b])[:12]}")
    return fresh


# ---------------------------------------------------------------- models
def quant_config(precision):
    if precision == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    if precision == "fp4":
        return BitsAndBytesConfig(load_in_4bit=True)
    if precision == "nf4":
        return BitsAndBytesConfig(load_in_4bit=True,
                                  bnb_4bit_quant_type="nf4",
                                  bnb_4bit_compute_dtype=torch.float16)
    return None


def load_model(name, rev, precision):
    if precision == "fp16":
        return AutoModelForCausalLM.from_pretrained(
            name, revision=rev, dtype=torch.float16, device_map="cuda")
    return AutoModelForCausalLM.from_pretrained(
        name, revision=rev, quantization_config=quant_config(precision), device_map="cuda")


def effective_config(model):
    qc = getattr(model.config, "quantization_config", None)
    if qc is not None and not isinstance(qc, dict):
        qc = qc.to_dict()
    dtypes = {}
    for p in model.parameters():
        dtypes[str(p.dtype)] = dtypes.get(str(p.dtype), 0) + p.numel()
    return {"quantization_config": json.loads(json.dumps(qc, default=str)),
            "parameter_dtypes": dtypes}


# ---------------------------------------------------------------- inference
def build_prompt(item):
    opts = "\n".join(f"{k}) {v}" for k, v in item["options"].items())
    return ("Answer the following multiple choice question with just the letter "
            "of the correct option.\n\n"
            f"{item['question']}\n{opts}\n\nAnswer:")


def render(tok, item):
    msgs = [{"role": "user", "content": build_prompt(item)}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   date_string=CHAT_DATE)


def prompts_sha256(tok, items):
    h = hashlib.sha256()
    for item in items:
        h.update(render(tok, item).encode("utf-8"))
    return h.hexdigest()


def letter_token_ids(tok):
    ids = {}
    for L in "ABCDE":
        enc = tok.encode(L, add_special_tokens=False)
        assert len(enc) == 1, f"letter {L} is not a single token: {enc}"
        ids[L] = enc[0]
    return ids


def run_pass(model, tok, items, letter_ids):
    out = {"ids": [], "preds": [], "confs": [], "probs": [], "logits": []}
    for item in items:
        letters = list(item["options"])
        inputs = tok(render(tok, item), return_tensors="pt").to("cuda")
        with torch.no_grad():
            last = model(**inputs).logits[0, -1, :].float()
        ll = torch.stack([last[letter_ids[L]] for L in letters]).cpu()
        probs = torch.softmax(ll, dim=0)
        idx = int(torch.argmax(probs))
        out["ids"].append(item["id"])
        out["preds"].append(letters[idx])
        out["confs"].append(float(probs[idx]))
        out["probs"].append([float(p) for p in probs])
        out["logits"].append([float(x) for x in ll])
    return out


def result_path(model, precision, bench):
    return os.path.join(RESULTS_DIR, f"{model.replace('/', '__')}__{precision}__{bench}.json")


def is_current(path, model_rev, item_hash, prompt_hash):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return (d.get("model_revision") == model_rev and d.get("items_sha256") == item_hash
            and d.get("prompts_sha256", prompt_hash) == prompt_hash)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    revs = resolve_revisions()
    print("Loading benchmarks...")
    items = load_items(revs)
    hashes = {b: items_sha256(items[b]) for b in BENCHMARKS}
    env = environment()

    for name in MODELS:
        rev = revs["models"][name]
        print(f"\n{'=' * 60}\n{name} @ {rev[:12]}\n{'=' * 60}")
        tok = AutoTokenizer.from_pretrained(name, revision=rev)
        letter_ids = letter_token_ids(tok)
        phash = {b: prompts_sha256(tok, items[b]) for b in BENCHMARKS}
        for precision in PRECISIONS:
            todo = []
            for b in BENCHMARKS:
                p = result_path(name, precision, b)
                if not os.path.exists(p):
                    todo.append(b)
                elif not is_current(p, rev, hashes[b], phash[b]):
                    sys.exit(f"STOP: {p} was produced with a different revision, item set, or prompt.")
            if not todo:
                print(f"  [{precision}] done, skipping")
                continue
            print(f"  [{precision}] loading model...")
            model = load_model(name, rev, precision)
            model.eval()
            cfg = effective_config(model)
            for b in todo:
                t0 = time.time()
                res = run_pass(model, tok, items[b], letter_ids)
                record = {
                    "model": name, "model_revision": rev, "tokenizer_revision": rev,
                    "precision": precision, "benchmark": b,
                    "dataset": DATASETS[b], "dataset_revision": revs["datasets"][DATASETS[b]],
                    "items_sha256": hashes[b], "n_items": len(items[b]),
                    "chat_date": CHAT_DATE, "prompts_sha256": phash[b],
                    "option_letters": "ABCDE" if b == "csqa" else "ABCD",
                    **cfg, "environment": env,
                    "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "seconds": round(time.time() - t0, 1),
                    **res,
                }
                with open(result_path(name, precision, b), "w", encoding="utf-8") as f:
                    json.dump(record, f)
                print(f"    {b}: {len(items[b])} items in {record['seconds']:.0f}s, saved")
            del model
            torch.cuda.empty_cache()

    print("\nAll runs complete. Next: python determinism_check.py, then python analyze_results.py")


if __name__ == "__main__":
    main()
