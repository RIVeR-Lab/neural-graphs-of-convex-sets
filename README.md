# Neural Graphs of Convex Sets

This repository contains the code associated to our paper [Neural Graphs of Convex Sets: Towards Real-Time Long-Horizon Discrete and Continuous Motion Planning](https://neural-gcs.github.io/).

**TL;DR:** We accelerate motion planning in Graphs of Convex Sets by replacing the convex relaxation stage with a Graph Neural Network that predicts edge flows, from which we sample candidate paths. A RankNet-style ranker identifies the top candidate paths, and we round only those paths. We demonstrate the proposed approach on obstacle-free quadrotor motion planning and planning through contact.

## Acknowledgments

We are grateful to the authors of [Motion Planning around Obstacles with Convex Optimization](https://arxiv.org/abs/2205.04422) and [Towards Tight Convex Relaxations for Contact-Rich Manipulation](https://arxiv.org/abs/2402.10312) for making their codebases public. This project builds on their work.

---

## Setup

Tested with Python 3.12 and CUDA 12.1. If your machine uses a different CUDA version, replace `cu121` in the URLs below.

```console
python3.12 -m venv venv
source venv/bin/activate
python3 -m pip install -U pip
pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu121
python3 -m pip install -r requirements.txt
python3 -m pip install -e . --no-deps
pip install torch-scatter -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__.split('+')[0])")+cu121.html
```

**Drake / MOSEK:** The quadrotor and GCS examples use Drake's GCS bindings. Install and activate a MOSEK license to reproduce those solver paths exactly.

---

## Quadrotor Motion Planning

Randomly generated 3×3 indoor environments (≈ 9 m × 9 m, 3 m ceiling). Start and goal poses are sampled uniformly inside convex regions with δ = 0.3 m margin and minimum graph-hop distance of 3. Dataset: 700 instances (500 train / 100 val / 100 test), one query per building.

```console
bash scripts/quadrotor_collect.sh     # collect dataset
bash scripts/quadrotor_train.sh       # train Flow GNN then RankNet (W&B by default)
bash scripts/quadrotor_benchmark.sh   # benchmark on test set
```

Checkpoints saved to `quadrotor/checkpoints/`. Results saved to `quadrotor/results/benchmark_results.json`.

---

## Planning Through Contact

Planar pushing with `sugar_box` and `tee` bodies. Benchmarks Neural GCS against vanilla GCS and a contact-implicit baseline.

### Option A: Download pretrained artifacts

```console
pip install gdown
bash scripts/download_data.sh
```

### Option B: Generate data and train from scratch

```console
bash scripts/generate_and_train.sh --box    # sugar_box
bash scripts/generate_and_train.sh --tee    # tee
bash scripts/generate_and_train.sh --box --tee
```

### Run the full benchmark

```console
bash scripts/run_full_benchmark.sh --box
bash scripts/run_full_benchmark.sh --tee
bash scripts/run_full_benchmark.sh --box --skip_trajopt   # skip SNOPT baseline
```

Results written to `benchmark_results/<body>_N/` (auto-incrementing), including `results_table.pdf`.

### Visualization

```console
bash scripts/visualize.sh --box
bash scripts/render_figures.sh --box
```
