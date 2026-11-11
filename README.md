# Accuracy is Not All You Need on Current Small Instruction-Tuned Models

Code and data for the ReScience C replication report
**"[Re] Accuracy is Not All You Need on Current Small Instruction-Tuned Models"**
by Avyay Bommaraju.

This repository replicates the bitsandbytes quantization-flips analysis of
Dutta et al., *Accuracy is Not All You Need* (NeurIPS 2024,
[arXiv:2407.09141](https://arxiv.org/abs/2407.09141)), on five current small
instruction-tuned models.

## What this measures

For each model, the code compares full-precision (fp16) predictions with
quantized predictions on three multiple-choice benchmarks and reports

- **Flips**, answers that move between correct and incorrect in either direction,
- **AllFlips**, Flips plus wrong-to-wrong transitions,
- the **directional breakdown** (C->I, I->C, W->W) against a benchmark-specific
  chance baseline, and
- the flip rate of initially correct answers as a function of the fp16
  top-two probability margin.

## Conditions

| Label | Configuration |
|---|---|
| `fp16` | 16-bit reference |
| `int8` | bitsandbytes LLM.int8(), library defaults |
| `fp4`  | bitsandbytes 4-bit, library defaults (FP4, float32 compute). This matches the "default parameters" condition of Dutta et al. and is the replication condition. |
| `nf4`  | bitsandbytes 4-bit NF4 with float16 compute, reported as an extension |

Models: Qwen2.5-1.5B-Instruct, Qwen2.5-3B-Instruct, Llama-3.2-3B-Instruct,
Gemma-2-2B-it, Phi-3.5-mini-instruct.

Benchmarks:
- ARC-Challenge test, the first 1,000 items whose options are labelled A to D
- MMLU test, the first 125 items of each of eight fixed subjects (975 items)
- CommonsenseQA validation, the first 1,000 items with all five options

The exact items and answer keys are in `results/_bench_items.json`.

## Provenance

`revisions.json` pins the commit hash of every model and dataset. Every
result file records the model and dataset revisions, a SHA-256 hash of the item
set, the item IDs beside the predictions, the probability of every answer
option, the effective quantization configuration and parameter dtypes, the
package versions, the GPU, and a hash of the script that produced it.

## Environment

The results were produced on Windows with an NVIDIA RTX 3070 (8 GB), Python
3.12, and the versions pinned in `requirements.txt`.

```bash
pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Llama-3.2 and Gemma-2 are gated on Hugging Face, so accept their licenses and
run `hf auth login` first.

## Reproducing the results

```bash
python run_experiment.py          # all passes, written to results/
python determinism_check.py       # repeated runs, written to results/determinism/
python analyze_results.py results --previous results_2026-06   # every number in the paper
python make_figures.py results figures
```

`analyze_results.py` and `make_figures.py` read only the saved files and need
no GPU, so the paper's numbers can be checked from the released predictions
alone. The `--previous` option compares the predictions with the earlier run in
`results_2026-06/`, which used NF4 as its 4-bit condition and a four-option
version of CommonsenseQA and did not fix the date in the Llama-3.2 prompt. It
is kept only for that comparison.

The Llama-3.2 chat template writes the current date into the prompt unless a
date is supplied. `run_experiment.py` fixes it to the template's default of
26 July 2024 (`CHAT_DATE`), so that predictions do not depend on the day of the
run.

## Repository layout

- `run_experiment.py` runs the passes and writes `results/*.json`.
- `determinism_check.py` runs and archives the repeated-run determinism check.
- `analyze_results.py` computes every metric reported in the paper.
- `make_figures.py` produces the figures and the LaTeX tables.
- `revisions.json` pins the model and dataset revisions.
- `results/` contains the per-item predictions, `_bench_items.json`, and
  `determinism/`.
- `results_2026-06/` contains the earlier run used for the cross-run comparison.
- `pilot/` contains the exploratory checks run while designing the study. They
  are not needed to reproduce the paper.

## License

Code is released under the MIT License. See `LICENSE`.

## Citation

If you use this code, please cite the ReScience C article and the original work:

> Abhinav Dutta, Sanjeev Krishnan, Nipun Kwatra, and Ramachandran Ramjee.
> Accuracy is Not All You Need. Advances in Neural Information Processing Systems
> (NeurIPS), 2024.
