"""Replicate the confident-reversal comparison on Phi-3.5-mini."""

import json
import time
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL = "microsoft/Phi-3.5-mini-instruct"
N_QUESTIONS = 400
CONFIDENT_THRESHOLD = 0.90      # fixed before seeing results


def load_questions(n):
    print("Loading ARC-Challenge...")
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    items = []
    for ex in ds:
        labels = ex["choices"]["label"]
        texts = ex["choices"]["text"]
        if labels != ["A", "B", "C", "D"]:
            continue
        items.append({
            "id": ex["id"], "question": ex["question"],
            "options": dict(zip(labels, texts)), "answer": ex["answerKey"],
        })
        if len(items) >= n:
            break
    print(f"Using {len(items)} questions.")
    return items


def build_prompt(item):
    opts = "\n".join(f"{k}) {v}" for k, v in item["options"].items())
    return ("Answer the following multiple choice question with just the letter "
            "of the correct option.\n\n"
            f"{item['question']}\n{opts}\n\nAnswer:")


def option_token_ids(tok):
    return {L: tok.encode(L, add_special_tokens=False)[0] for L in "ABCD"}


def run_pass(model, tok, items, letter_ids):
    preds, confs = [], []
    for item in items:
        msgs = [{"role": "user", "content": build_prompt(item)}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model(**inputs)
        next_logits = out.logits[0, -1, :]
        letter_logits = torch.tensor([next_logits[letter_ids[L]].item() for L in "ABCD"])
        probs = torch.softmax(letter_logits, dim=0)
        idx = int(torch.argmax(probs))
        preds.append("ABCD"[idx])
        confs.append(float(probs[idx]))
    return preds, confs


def main():
    items = load_questions(N_QUESTIONS)
    answers = [it["answer"] for it in items]
    tok = AutoTokenizer.from_pretrained(MODEL)
    letter_ids = option_token_ids(tok)

    print("\n--- float16 pass ---")
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map="cuda")
    t0 = time.time()
    preds16, confs16 = run_pass(m, tok, items, letter_ids)
    print(f"  done in {time.time()-t0:.0f}s")
    del m; torch.cuda.empty_cache()

    print("\n--- 4-bit pass ---")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_quant_type="nf4")
    m = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb, device_map="cuda")
    t0 = time.time()
    preds4, confs4 = run_pass(m, tok, items, letter_ids)
    print(f"  done in {time.time()-t0:.0f}s")
    del m; torch.cuda.empty_cache()

    with open("replication_phi_outputs.json", "w", encoding="utf-8") as f:
        json.dump({"items": items, "preds_fp16": preds16, "confs_fp16": confs16,
                   "preds_4bit": preds4, "confs_4bit": confs4}, f, indent=2)

    n = len(items)
    acc16 = sum(p == a for p, a in zip(preds16, answers)) / n
    acc4 = sum(p == a for p, a in zip(preds4, answers)) / n
    any_change = sum(1 for a, b in zip(preds16, preds4) if a != b)

    # split flips into confident reversals vs borderline churn
    confident_flips = []
    borderline_flips = []
    for i in range(n):
        if preds16[i] != preds4[i]:
            if confs16[i] >= CONFIDENT_THRESHOLD and confs4[i] >= CONFIDENT_THRESHOLD:
                confident_flips.append(i)
            else:
                borderline_flips.append(i)

    print("\n" + "=" * 64)
    print(f"REPLICATION CHECK: {MODEL}")
    print("=" * 64)
    print(f"  float16 accuracy: {acc16:.1%}")
    print(f"  4-bit  accuracy:  {acc4:.1%}")
    print(f"  average accuracy change: {(acc4-acc16)*100:+.1f} points")
    print()
    print(f"  any answer change: {any_change}  ({any_change/n:.1%} of items)")
    print(f"  CONFIDENT reversals (both >= {CONFIDENT_THRESHOLD}): {len(confident_flips)}")
    print(f"  borderline flips:                       {len(borderline_flips)}")
    print()
    print("  sample confident reversals (the strong-effect cases):")
    for i in confident_flips[:12]:
        print(f"    item {i}: fp16={preds16[i]}({confs16[i]:.2f}) "
              f"4bit={preds4[i]}({confs4[i]:.2f}) key={answers[i]}")
    print()
    print("=" * 64)
    print("VERDICT")
    print("=" * 64)
    qwen_had = "Qwen showed ~49 flips with several confident reversals."
    print(f"  {qwen_had}")
    if len(confident_flips) >= 5:
        print(f"  Phi shows {len(confident_flips)} confident reversals too.")
        print("  -> The effect REPLICATES across model families. Preregister.")
    elif any_change >= 0.05 * n:
        print(f"  Phi flips {any_change} items but few are confident reversals.")
        print("  -> Partial replication: churn yes, confident reversals weaker.")
        print("     The honest framing may be about churn generally, not")
        print("     confident reversals specifically. Discuss before prereg.")
    else:
        print(f"  Phi barely changed ({any_change} flips).")
        print("  -> Does NOT replicate. The Qwen effect may be model-specific.")
        print("     This is important. Rethink before any prereg.")
    print("""
  Then hand-check the confident reversals above, same as with Qwen, to confirm
  they are genuine and not measurement glitches.
""")


if __name__ == "__main__":
    main()
