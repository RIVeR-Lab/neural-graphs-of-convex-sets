# Neural Graphs of Convex Sets
This repository contains the code associated to our paper [Neural Graphs of Convex Sets: Towards Real-Time Long-Horizon Discrete and Continuous Motion Planning](https://neural-gcs.github.io/).

**TL;DR:** We accelerate motion planning in Graphs of Convex Sets by replacing the convex relaxation stage with a Graph Neural Network that predicts edge flows, from which we sample candidate paths. A RankNet-style ranker identifies the top candidate paths, and we round only those paths. We demonstrate the proposed approach on obstacle-free quadrotor motion planning and planning through contact.

## Acknowledgments

We are grateful to the authors of [Motion Planning around Obstacles with Convex Optimization](https://arxiv.org/abs/2205.04422) and [Towards Tight Convex Relaxations for Contact-Rich Manipulation](https://arxiv.org/abs/2402.10312) for making their codebases public. This project builds on their work.

## Setup

Tested with Python 3.12 and CUDA 12.1 : If your machine uses a different CUDA version, replace `cu121` in the PyTorch and PyG URLs below.

```console
python3.12 -m venv venv
source venv/bin/activate
python3 -m pip install -U pip
```

Install PyTorch (CUDA 12.1):
```console
pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu121
```

Install remaining dependencies:
```console
python3 -m pip install -r requirements.txt
python3 -m pip install -e . --no-deps
```

Install `torch-scatter` (matched to your torch + CUDA version):
```console
pip install torch-scatter -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__.split('+')[0])")+cu121.html
```

Drake / MOSEK: The quadrotor and GCS examples use Drake's GCS bindings. Some GCS solves go through MOSEK — install and activate a MOSEK license if you want to reproduce those solver paths exactly.

## Planning Through Contact

The planning through contact experiments benchmark Neural GCS against vanilla GCS and a contact-implicit trajectory optimization baseline on planar pushing tasks (sugar_box and tee bodies).

### Reproducing Benchmark Results

To run the benchmark you need data and trained checkpoints in place. Choose one of the two paths below — both produce the same artifact layout.

### Option A: Download pretrained artifacts

Data and pretrained checkpoints are hosted on [Google Drive](https://drive.google.com/drive/folders/1WGXLxlpctM_KHSMEiCVzDsGjrM8w0DvB).
Run the download script:

```console
pip install gdown
bash scripts/download_data.sh
```

This places files at:
```
planning_through_contact/dataset/data/sugar_box/   ← gcs_solutions.h5, global_features.csv, node_features.csv
planning_through_contact/dataset/data/tee/         ← same
checkpoints/gcs_gnn/sugar_box_flow.ckpt
checkpoints/gcs_gnn/tee_flow.ckpt
checkpoints/ranknet/sugar_box_ranker.ckpt
checkpoints/ranknet/tee_ranker.ckpt
```

### Option B: Generate data and train from scratch

This runs dataset generation (plan index → node features → GCS solutions) followed by GNN and RankNet training, writing artifacts to the same paths as Option A.

```console
bash scripts/generate_and_train.sh --box    # sugar_box only
bash scripts/generate_and_train.sh --tee    # tee only
bash scripts/generate_and_train.sh --box --tee
```

Dataset size defaults to 500 train / 100 val / 100 test instances. Override with environment variables e.g:

```console
NUM_TRAIN=1000 NUM_TEST=200 bash scripts/generate_and_train.sh --box
```

### Run the full benchmark

```console
bash scripts/run_full_benchmark.sh --box          # sugar_box only
bash scripts/run_full_benchmark.sh --tee          # tee only
bash scripts/run_full_benchmark.sh --box --tee    # both, box first
```

This runs all methods on the held-out test split and writes results to `benchmark_results/<body>_1/`, `benchmark_results/<body>_2/`, etc. (auto-incrementing).

To skip the Contact-Implicit baseline (requires SNOPT):

```console
bash scripts/run_full_benchmark.sh --box --skip_trajopt
```

To verify your SNOPT installation:

```console
python3 -c "from pydrake.solvers import SnoptSolver; s = SnoptSolver(); print('available:', s.available(), '| licensed:', s.enabled())"
```

Both should print `True` if SNOPT is correctly installed and licensed.

### Results table

The script automatically generates `results_table.pdf` in the benchmark output directory.


The table compares the following methods on success rate, computation time (relaxation / rounding / total, mean ± std), and optimality gap:

| Method | Notes |
|---|---|
| GCS (100 steps) | Vanilla GCS, full rounding budget |
| GCS (10 steps) | Vanilla GCS, reduced rounding budget |
| Neural GCS w/ RankNet (100 paths) | GNN + RankNet, full budget |
| Neural GCS w/ RankNet (10 paths) | GNN + RankNet, reduced budget |
| Neural GCS w/o RankNet (100 paths) | GNN only, full budget |
| Neural GCS w/o RankNet (10 paths) | GNN only, reduced budget |
| Contact-Implicit | Direct trajopt via SNOPT (requires license) |

### Visualization

Generate contact-annotated videos for test-set instances (green = contact, red = non-contact):

```console
bash scripts/visualize.sh --box                         # 1 instance, 100 paths, GNN + RankNet
bash scripts/visualize.sh --box --num 4 --max_paths 10  # 4 instances, reduced rounding budget
bash scripts/visualize.sh --box --without_ranknet       # GNN only, no reranking
bash scripts/visualize.sh --tee                         # tee body
```

Videos are saved to `trajectories/<body>/video_1/`, `video_2/`, etc. (auto-incrementing), under `plan_<id>/trajectory/`:
- `prediction.mp4` — Neural GCS trajectory (green = contact, red = non-contact)
- `ground_truth.mp4` — Vanilla GCS trajectory for comparison (requires H5 data)

### Trajectory figures

Generate static PDF figures — ghosted slider positions with fading opacity, color-coded pusher trail (green = contact, red = non-contact), dashed goal outline:

```console
bash scripts/render_figures.sh --box                         # 100 instances, 100 paths, with RankNet (defaults)
bash scripts/render_figures.sh --box --max_paths 10          # 100 instances, 10 paths (faster)
bash scripts/render_figures.sh --box --num 4                 # 4 instances
bash scripts/render_figures.sh --box --without_ranknet       # GNN only, no reranking
bash scripts/render_figures.sh --tee
```

Figures are saved to `figures/<body>/figure_1/`, `figure_2/`, etc. (auto-incrementing). Each instance produces:
- `plan_<id>_trajectory.pdf` — all segments in one figure with legend at top
- `plan_<id>_panel_0.pdf`, `_panel_1.pdf`, … — one PDF per segment group (for paper layout)
- `plan_<id>_legend.pdf` — standalone legend (contact / non-contact / goal)
