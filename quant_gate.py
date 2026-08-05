"""Measure quantization behavior change beyond average accuracy: flip set and confidence."""

import json
import re
import time
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL = "Qwen/Qwen2.5-3B-Instruct"
N_QUESTIONS = 400


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
            "id": ex["id"],
            "question": ex["question"],
            "options": dict(zip(labels, texts)),
            "answer": ex["answerKey"],
        })
        if len(items) >= n:
            break
    print(f"Using {len(items)} questions.")
    return items


def build_prompt(item):
    opts = "\n".join(f"{k}) {v}" for k, v in item["options"].items())
    return (
        "Answer the following multiple choice question with just the letter "
        "of the correct option.\n\n"
        f"{item['question']}\n{opts}\n\nAnswer:"
    )


# Letter token ids, resolved per tokenizer.
def option_token_ids(tok):
    ids = {}
    for letter in "ABCD":
        # leading space variant is how these usually tokenize mid-text
        toks = tok.encode(letter, add_special_tokens=False)
        ids[letter] = toks[0]
    return ids


def run_pass(model, tok, items, letter_ids):
    """For each item return (predicted_letter, confidence_on_chosen).

    confidence = softmax probability the model puts on its chosen letter at the
    first generated position, among the four option letters. This is the
    calibration signal.
    """
    preds = []
    confs = []
    for item in items:
        prompt = build_prompt(item)
        msgs = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model(**inputs)
        # logits for the next token after the prompt
        next_logits = out.logits[0, -1, :]
        # restrict to the four option-letter tokens, softmax over just those
        letter_logits = torch.tensor(
            [next_logits[letter_ids[L]].item() for L in "ABCD"]
        )
        probs = torch.softmax(letter_logits, dim=0)
        idx = int(torch.argmax(probs))
        pred = "ABCD"[idx]
        preds.append(pred)
        confs.append(float(probs[idx]))
    return preds, confs


def load_fp16():
    return AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map="cuda")


def load_4bit():
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    return AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb, device_map="cuda")


def main():
    items = load_questions(N_QUESTIONS)
    answers = [it["answer"] for it in items]
    tok = AutoTokenizer.from_pretrained(MODEL)
    letter_ids = option_token_ids(tok)

    print("\n--- float16 pass ---")
    m = load_fp16()
    t0 = time.time()
    preds16, confs16 = run_pass(m, tok, items, letter_ids)
    print(f"  done in {time.time()-t0:.0f}s")
    del m
    torch.cuda.empty_cache()

    print("\n--- 4-bit pass ---")
    m = load_4bit()
    t0 = time.time()
    preds4, confs4 = run_pass(m, tok, items, letter_ids)
    print(f"  done in {time.time()-t0:.0f}s")
    del m
    torch.cuda.empty_cache()

    # save for re-analysis without rerunning
    with open("quant_gate_outputs.json", "w", encoding="utf-8") as f:
        json.dump({
            "items": items,
            "preds_fp16": preds16, "confs_fp16": confs16,
            "preds_4bit": preds4, "confs_4bit": confs4,
        }, f, indent=2)

    n = len(items)
    acc16 = sum(p == a for p, a in zip(preds16, answers)) / n
    acc4 = sum(p == a for p, a in zip(preds4, answers)) / n

    # flip set
    c2w = sum(1 for p16, p4, a in zip(preds16, preds4, answers) if p16 == a and p4 != a)
    w2c = sum(1 for p16, p4, a in zip(preds16, preds4, answers) if p16 != a and p4 == a)
    any_change = sum(1 for p16, p4 in zip(preds16, preds4) if p16 != p4)

    # calibration
    mean_conf16 = sum(confs16) / n
    mean_conf4 = sum(confs4) / n

    print("\n" + "=" * 64)
    print("QUANTIZATION BEHAVIOR-CHANGE GATE RESULTS")
    print("=" * 64)
    print(f"  questions: {n}")
    print()
    print(f"  float16 accuracy: {acc16:.1%}")
    print(f"  4-bit  accuracy:  {acc4:.1%}")
    print(f"  average accuracy change: {(acc4-acc16)*100:+.1f} points")
    print()
    print("  FLIP SET (the hidden effect):")
    print(f"    correct -> wrong:  {c2w}")
    print(f"    wrong -> correct:  {w2c}")
    print(f"    any answer change: {any_change}  ({any_change/n:.1%} of items)")
    print()
    print("  CALIBRATION:")
    print(f"    mean confidence float16: {mean_conf16:.3f}")
    print(f"    mean confidence 4-bit:   {mean_conf4:.3f}")
    print(f"    confidence change: {(mean_conf4-mean_conf16):+.3f}")
    print()
    print("=" * 64)
    print("HOW TO READ THIS")
    print("=" * 64)
    print(f"""
  The key comparison: average accuracy change ({(acc4-acc16)*100:+.1f} pts)
  versus how many items actually CHANGED ({any_change}, {any_change/n:.1%}).

  If average accuracy barely moved but {any_change} items flipped answers, the
  average is HIDING real behavioral change. That gap is the finding: quantization
  doesn't just shave accuracy, it reshuffles which items the model gets right.
  That is a defensible, non-obvious, TMLR-shaped result. -> build it.

  If almost nothing flipped (any_change near 0), quantization is benign on this
  model/benchmark and the angle is weak. Reconsider.

  Calibration: a clear systematic confidence shift is a SECOND real effect worth
  reporting even on its own.

  Then hand-check: open quant_gate_outputs.json, read a few flipped items, and
  confirm they are genuine answer changes, not a measurement glitch. Same
  discipline as the CRI 'Just' bug and the strict-parser strawman.
""")


if __name__ == "__main__":
    main()
