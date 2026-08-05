# Accuracy is Not All You Need on Current Small Instruction-Tuned Models

Code and data for the ReScience C replication report
**"[Re] Accuracy is Not All You Need on Current Small Instruction-Tuned Models"**
by Avyay Bommaraju.

This repository replicates the bitsandbytes quantization-flips analysis of
Dutta et al., *Accuracy is Not All You Need* (NeurIPS 2024,
[arXiv:2407.09141](https://arxiv.org/abs/2407.09141)), on five current small
instruction-tuned models, and reports where the results diverge on this model
generation.

## What this measures

For each model, the code compares the full-precision (fp16) predictions against
the 8-bit and 4-bit quantized predictions on three multiple-choice benchmarks,
and counts:

- **Flips**: answers that move between correct and incorrect (either direction).
- **AllFlips**: Flips plus wrong-to-wrong transitions.
- **Directional breakdown**: correct-to-incorrect (C->I), incorrect-to-correct
  (I->C), and wrong-to-wrong (W->W).

## Models and benchmarks

Models: Qwen2.5-1.5B-Instruct, Qwen2.5-3B-Instruct, Llama-3.2-3B-Instruct,
Gemma-2-2B-it, Phi-3.5-mini-instruct.

Benchmarks (fixed subsets): ARC-Challenge (1,000 items), MMLU (975 items, eight
subjects), CommonsenseQA (986 items). The exact items and answer keys are in
`results/_bench_items.json`.

## Environment

- Python 3.10 or newer
- A CUDA GPU (experiments were run on a single 8 GB consumer GPU)
- Install dependencies:

```bash
pip install -r requirements.txt
```

Key packages: torch, transformers, bitsandbytes, datasets, numpy, scipy,
matplotlib.

## Reproducing the results

```bash
# 1. Run every (model, precision, benchmark) pass and save per-run predictions
python run_experiment.py

# 2. Compute the metrics and print the full results table
python analyze_results.py

# 3. Regenerate the figures and the LaTeX results table
python make_figures.py

# 4. Optional supporting checks
python count_items.py            # item count per benchmark
python nondeterminism_gate.py    # determinism floor (should be zero)
python check_mmlu_difficulty.py  # per-subject MMLU accuracy (appendix)
```

Model weights are downloaded from their public repositories at run time and are
not stored here. Benchmark subsets are fixed and included, so the numbers in the
paper are reproducible.

## Repository layout

- `run_experiment.py` runs the passes and writes `results/*.json`.
- `analyze_results.py` computes Flips, AllFlips, and the directional breakdown.
- `make_figures.py` produces the figures and the results table.
- `dutta_comparison.py` compares against the original claim definitions.
- `nondeterminism_gate.py` measures the run-to-run determinism floor.
- `results/` per-run predictions and `_bench_items.json` (the fixed subsets).
- Supporting scripts: `handcheck_ci_flips.py`, `handcheck_reversals.py`,
  `position_bias_check.py`, `quant_gate.py`, `replication_check.py`,
  `model_test.py`, `quant_test.py`, `peek_ids.py`.

## License

Code is released under the MIT License. See `LICENSE`.

## Citation

If you use this code, please cite the ReScience C article (details will be added
once the DOI is assigned) and the original work:

> Abhinav Dutta, Sanjeev Krishnan, Nipun Kwatra, and Ramachandran Ramjee.
> Accuracy is Not All You Need. Advances in Neural Information Processing Systems
> (NeurIPS), 2024.
