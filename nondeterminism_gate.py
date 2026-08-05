"""Measure the temperature-zero run-to-run nondeterminism floor, single-item and batched."""

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-3B-Instruct"
N_QUESTIONS = 300
N_RUNS = 4
BATCH_SIZES_FOR_VARIED = [1, 8, 16, 4]  # one per run, deliberately different


def load_questions(n):
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    items = []
    for ex in ds:
        labels, texts = ex["choices"]["label"], ex["choices"]["text"]
        if labels != ["A", "B", "C", "D"]:
            continue
        items.append({
            "question": ex["question"],
            "options": dict(zip(labels, texts)),
            "answer": ex["answerKey"],
        })
        if len(items) >= n:
            break
    return items


def build_prompt(item):
    opts = "\n".join(f"{k}) {v}" for k, v in item["options"].items())
    return ("Answer the following multiple choice question with just the letter "
            "of the correct option.\n\n"
            f"{item['question']}\n{opts}\n\nAnswer:")


def option_token_ids(tok):
    return {L: tok.encode(L, add_special_tokens=False)[0] for L in "ABCD"}


def predict_single(model, tok, item, letter_ids):
    """One question, single-item forward pass."""
    msgs = [{"role": "user", "content": build_prompt(item)}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(**inputs)
    nl = out.logits[0, -1, :]
    ll = torch.tensor([nl[letter_ids[L]].item() for L in "ABCD"])
    return "ABCD"[int(torch.argmax(ll))]


def predict_batch(model, tok, items, letter_ids, batch_size):
    """Predict all items using a given batch size (left-padded)."""
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    preds = []
    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        texts = []
        for item in chunk:
            msgs = [{"role": "user", "content": build_prompt(item)}]
            texts.append(tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True))
        inputs = tok(texts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = model(**inputs)
        # last non-pad position per row is just the last position (left padded)
        last_logits = out.logits[:, -1, :]
        for r in range(last_logits.shape[0]):
            ll = torch.tensor([last_logits[r, letter_ids[L]].item() for L in "ABCD"])
            preds.append("ABCD"[int(torch.argmax(ll))])
    return preds


def floor_rate(runs):
    """Given a list of prediction-lists (one per run), fraction of items whose
    prediction is NOT identical across all runs."""
    n = len(runs[0])
    changed = 0
    for i in range(n):
        vals = {run[i] for run in runs}
        if len(vals) > 1:
            changed += 1
    return changed / n


def main():
    items = load_questions(N_QUESTIONS)
    tok = AutoTokenizer.from_pretrained(MODEL)
    letter_ids = option_token_ids(tok)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, device_map="cuda")

    # Condition A: naive identical single-item passes
    print("Condition A (naive single-item), running", N_RUNS, "identical passes...")
    naive_runs = []
    for r in range(N_RUNS):
        preds = [predict_single(model, tok, it, letter_ids) for it in items]
        naive_runs.append(preds)
        print(f"  run {r+1} done")
    naive_floor = floor_rate(naive_runs)

    # Condition B: varied batch size per run
    print("\nCondition B (varied batch sizes), running", N_RUNS, "passes...")
    varied_runs = []
    for r in range(N_RUNS):
        bs = BATCH_SIZES_FOR_VARIED[r % len(BATCH_SIZES_FOR_VARIED)]
        preds = predict_batch(model, tok, items, letter_ids, bs)
        varied_runs.append(preds)
        print(f"  run {r+1} (batch size {bs}) done")
    varied_floor = floor_rate(varied_runs)

    # Also: does single-item disagree with batched? (cross-condition)
    cross_floor = floor_rate([naive_runs[0], varied_runs[0]])

    print("\n" + "=" * 60)
    print("NONDETERMINISM FLOOR RESULTS")
    print("=" * 60)
    print(f"  questions: {len(items)}, runs per condition: {N_RUNS}")
    print()
    print(f"  A. naive single-item floor:   {naive_floor:.2%}")
    print(f"  B. varied-batch floor:        {varied_floor:.2%}")
    print(f"  cross (single vs batched):    {cross_floor:.2%}")
    print()
    print("=" * 60)
    if naive_floor > 0.005:
        print("FLOOR IS NONZERO in the naive setup. The correction angle is")
        print("alive: baseline reruns already disagree, so some quantization")
        print("churn is really nondeterminism. Proceed.")
    elif varied_floor > 0.005 or cross_floor > 0.005:
        print("Floor is ~zero naively but NONZERO under varied batching.")
        print("Nondeterminism is latent and surfaces under realistic serving.")
        print("That is itself a finding worth discussing before deciding.")
    else:
        print("FLOOR IS ESSENTIALLY ZERO in all conditions. Your local setup")
        print("is deterministic. The subtract-the-floor framing is dead.")
        print("Move to pure reproducibility, as agreed.")
    print("""
Then sanity-check: a nonzero floor should look like a SMALL fraction of items
flipping, concentrated on low-confidence items. If it is huge or random, suspect
a bug (e.g. padding handling in the batched path) before believing it.
""")


if __name__ == "__main__":
    main()
