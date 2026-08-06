# Reproducibility notes

## Environment used to produce the results

- OS: Windows
- GPU: NVIDIA GeForce RTX 3070 (8 GB)
- Python: 3.12
- CUDA: PyTorch built against CUDA 12.4 (torch 2.6.0+cu124)

Exact package versions are pinned in `requirements.txt`.

## Install

Install PyTorch with the matching CUDA build first, then the rest:

```
pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Reproduce

```
python run_experiment.py      # loads each model at fp16/8-bit/4-bit, saves results/
python analyze_results.py     # metrics from results/
python make_figures.py        # figures and the results table
```

Model weights are downloaded from their public Hugging Face repositories at run
time. Gemma-2-2B and Llama-3.2-3B are gated; a reviewer needs a Hugging Face
account with access granted to those model pages.

## Note on exact numbers

The aggregate results and every claim in the paper are robust. Exact per-item
counts in Table 1 can shift by a small number on different hardware, CUDA
versions, or library versions, because a few low-margin items can tip under tiny
numeric differences. On the environment above the run-to-run floor is zero
(see the determinism control in the paper). Reviewers on different GPUs may see
a handful of items differ while all reported claims and the pooled statistics
hold.
