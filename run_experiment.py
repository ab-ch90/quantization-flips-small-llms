"""Run each (model, precision, benchmark) pass and save per-pass result files."""

import json
import os
import time
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ---------------- frozen config (matches preregistration) ----------------
MODELS = [
    "Qwen/Qwen2.5-3B-Instruct",
    "microsoft/Phi-3.5-mini-instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "google/gemma-2-2b-it",
    "meta-llama/Llama-3.2-3B-Instruct",
]
PRECISIONS = ["fp16", "int8", "int4"]
BENCHMARKS = ["arc", "mmlu", "csqa"]
N_PER_BENCHMARK = 1000          # prereg N; uses all available if fewer
MMLU_SUBJECTS = [
    "college_computer_science", "high_school_physics",
    "philosophy", "professional_law",
    "high_school_psychology", "high_school_macroeconomics",
    "professional_medicine", "marketing",
]
SEED = 20260622
RESULTS_DIR = "results"
# -------------------------------------------------------------------------

os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------- benchmark loaders: each returns list of standardized items ----------
def _std_item(qid, question, options_dict, answer_letter):
    return {"id": qid, "question": question,
            "options": options_dict, "answer": answer_letter}


def load_arc(n):
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    items = []
    for ex in ds:
        labels, texts = ex["choices"]["label"], ex["choices"]["text"]
        if labels != ["A", "B", "C", "D"]:
            continue
        items.append(_std_item(ex["id"], ex["question"],
                               dict(zip(labels, texts)), ex["answerKey"]))
        if len(items) >= n:
            break
    return items


def load_mmlu(n):
    """Pool the chosen subjects, standard 4-option, map index answer to letter."""
    items = []
    per = max(1, n // len(MMLU_SUBJECTS))
    for subj in MMLU_SUBJECTS:
        ds = load_dataset("cais/mmlu", subj, split="test")
        cnt = 0
        for ex in ds:
            choices = ex["choices"]
            if len(choices) != 4:
                continue
            letter = "ABCD"[ex["answer"]]
            opts = {"A": choices[0], "B": choices[1],
                    "C": choices[2], "D": choices[3]}
            items.append(_std_item(f"{subj}_{cnt}", ex["question"], opts, letter))
            cnt += 1
            if cnt >= per:
                break
    return items


def load_csqa(n):
    """CommonsenseQA has 5 options A-E. Keep only the first 4 to match the
    A-D letter-probability method, dropping items whose answer is E."""
    ds = load_dataset("tau/commonsense_qa", split="validation")
    items = []
    for ex in ds:
        labels = ex["choices"]["label"]
        texts = ex["choices"]["text"]
        ans = ex["answerKey"]
        if ans not in ("A", "B", "C", "D"):
            continue
        # take first four options only
        opts = {labels[i]: texts[i] for i in range(4)}
        if list(opts.keys()) != ["A", "B", "C", "D"]:
            continue
        items.append(_std_item(ex["id"], ex["question"], opts, ans))
        if len(items) >= n:
            break
    return items


BENCH_LOADERS = {"arc": load_arc, "mmlu": load_mmlu, "csqa": load_csqa}


def build_prompt(item):
    opts = "\n".join(f"{k}) {v}" for k, v in item["options"].items())
    return ("Answer the following multiple choice question with just the letter "
            "of the correct option.\n\n"
            f"{item['question']}\n{opts}\n\nAnswer:")


def load_model(model_name, precision):
    common = dict(device_map="cuda")
    if precision == "fp16":
        return AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float16, **common)
    if precision == "int8":
        cfg = BitsAndBytesConfig(load_in_8bit=True)
        return AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=cfg, **common)
    if precision == "int4":
        cfg = BitsAndBytesConfig(load_in_4bit=True,
                                 bnb_4bit_compute_dtype=torch.float16,
                                 bnb_4bit_quant_type="nf4")
        return AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=cfg, **common)
    raise ValueError(precision)


def option_token_ids(tok):
    return {L: tok.encode(L, add_special_tokens=False)[0] for L in "ABCD"}


def run_pass(model, tok, items, letter_ids):
    preds, confs = [], []
    for item in items:
        msgs = [{"role": "user", "content": build_prompt(item)}]
        text = tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model(**inputs)
        nl = out.logits[0, -1, :]
        ll = torch.tensor([nl[letter_ids[L]].item() for L in "ABCD"])
        probs = torch.softmax(ll, dim=0)
        idx = int(torch.argmax(probs))
        preds.append("ABCD"[idx])
        confs.append(float(probs[idx]))
    return preds, confs


def result_path(model_name, precision, bench):
    safe = model_name.replace("/", "__")
    return os.path.join(RESULTS_DIR, f"{safe}__{precision}__{bench}.json")


def main():
    # load and cache the benchmark item sets once (identical across all runs)
    print("Loading benchmarks...")
    bench_items = {}
    for b in BENCHMARKS:
        items = BENCH_LOADERS[b](N_PER_BENCHMARK)
        bench_items[b] = items
        print(f"  {b}: {len(items)} items")
    # save the item sets so the analysis has the questions and keys
    with open(os.path.join(RESULTS_DIR, "_bench_items.json"), "w",
              encoding="utf-8") as f:
        json.dump(bench_items, f)

    for model_name in MODELS:
        print(f"\n{'='*60}\nMODEL: {model_name}\n{'='*60}")
        tok = AutoTokenizer.from_pretrained(model_name)
        letter_ids = option_token_ids(tok)
        for precision in PRECISIONS:
            # skip completed passes (incremental safety)
            todo = [b for b in BENCHMARKS
                    if not os.path.exists(result_path(model_name, precision, b))]
            if not todo:
                print(f"  [{precision}] all benchmarks done, skipping")
                continue
            print(f"  [{precision}] loading model...")
            try:
                model = load_model(model_name, precision)
            except Exception as e:
                print(f"  ERROR loading {model_name} @ {precision}: {e}")
                print("  (if gated/access error, accept license + hf auth login)")
                continue
            for b in todo:
                items = bench_items[b]
                t0 = time.time()
                preds, confs = run_pass(model, tok, items, letter_ids)
                with open(result_path(model_name, precision, b), "w",
                          encoding="utf-8") as f:
                    json.dump({"model": model_name, "precision": precision,
                               "benchmark": b, "preds": preds, "confs": confs},
                              f)
                print(f"    {b}: {len(items)} items in {time.time()-t0:.0f}s "
                      f"-> saved")
            del model
            torch.cuda.empty_cache()

    print("\nAll requested runs complete. Run analyze_results.py next.")


if __name__ == "__main__":
    main()
