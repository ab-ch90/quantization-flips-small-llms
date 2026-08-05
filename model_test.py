"""Model-loading smoke test."""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model (first run downloads ~6GB, please wait)...")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,   # half precision to fit comfortably in 8GB
    device_map="cuda",
)
print(f"Model loaded in {time.time() - t0:.1f}s")
print(f"GPU memory used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# A multiple-choice question for the smoke test.
question = """Answer with just the letter of the correct option.

Which planet is closest to the Sun?
A) Earth
B) Mercury
C) Jupiter
D) Venus

Answer:"""

messages = [{"role": "user", "content": question}]
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
inputs = tokenizer(text, return_tensors="pt").to("cuda")

print("\nRunning inference...")
t0 = time.time()
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False,        # deterministic, so the test is reproducible
    )
gen = outputs[0][inputs["input_ids"].shape[1]:]   # only the new tokens
answer = tokenizer.decode(gen, skip_special_tokens=True)
print(f"Inference done in {time.time() - t0:.1f}s")

print("\n" + "=" * 50)
print("MODEL'S RAW ANSWER:")
print(repr(answer))
print("=" * 50)
print("\nExpected: the model should indicate B (Mercury).")
print("If it did, the full stack works and we can build the real gate.")
