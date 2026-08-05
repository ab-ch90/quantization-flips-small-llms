"""Verify that 4-bit quantization loads, uses less GPU memory than fp16, and answers sensibly."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL = "Qwen/Qwen2.5-3B-Instruct"

question = """Answer with just the letter of the correct option.

Which planet is closest to the Sun?
A) Earth
B) Mercury
C) Jupiter
D) Venus

Answer:"""


def ask(model, tok):
    msgs = [{"role": "user", "content": question}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(gen, skip_special_tokens=True)


tok = AutoTokenizer.from_pretrained(MODEL)

# ---- 1. float16 baseline ----
print("Loading float16 baseline...")
m16 = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map="cuda")
mem16 = torch.cuda.memory_allocated() / 1e9
ans16 = ask(m16, tok)
print(f"  float16 GPU memory: {mem16:.2f} GB")
print(f"  float16 answer: {ans16!r}")
del m16
torch.cuda.empty_cache()

# ---- 2. 4-bit quantized load ----
print("\nLoading 4-bit quantized...")
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",        # standard 4-bit type used in practice
)
m4 = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb, device_map="cuda")
mem4 = torch.cuda.memory_allocated() / 1e9
ans4 = ask(m4, tok)
print(f"  4-bit GPU memory: {mem4:.2f} GB")
print(f"  4-bit answer: {ans4!r}")

# ---- 3. verdict ----
print("\n" + "=" * 50)
print("VERIFICATION")
print("=" * 50)
print(f"  float16 memory: {mem16:.2f} GB")
print(f"  4-bit memory:   {mem4:.2f} GB")
saved = (1 - mem4 / mem16) * 100
print(f"  memory reduction: {saved:.0f}%")
print()
if mem4 < mem16 * 0.75 and ans4.strip().upper().startswith("B"):
    print("PASS: 4-bit uses clearly less memory AND still answers correctly.")
    print("Quantization is working. We can build the behavior-change gate.")
elif mem4 >= mem16 * 0.75:
    print("PROBLEM: 4-bit did NOT save much memory. Quantization may have")
    print("silently fallen back to full precision. Do not trust quant results")
    print("until this is fixed. Paste this output back.")
else:
    print("PARTIAL: memory dropped but the answer looks off. Worth a closer look.")
    print(f"  expected B, got: {ans4!r}")
