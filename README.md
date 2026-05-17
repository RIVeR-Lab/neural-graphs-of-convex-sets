# Neural Graphs of Convex Sets
This repository contains the code associated to our paper [Neural Graphs of Convex Sets: Towards Real-Time Long-Horizon Discrete and Continuous Motion Planning](https://neural-gcs.github.io/).

**TL;DR:** We accelerate motion planning in Graphs of Convex Sets by replacing the convex relaxation stage with a Graph Neural Network that predicts edge flows, from which we sample candidate paths. A RankNet-style ranker identifies the top candidate paths, and we round only those paths. We demonstrate the proposed approach on obstacle-free quadrotor motion planning and planning through contact, pushing both toward real-time rates exceeding ~50Hz.


## Acknowledgments

We are grateful to the authors of [Motion Planning around Obstacles with Convex Optimization](https://arxiv.org/abs/2205.04422) and [Towards Tight Convex Relaxations for Contact-Rich Manipulation](https://arxiv.org/abs/2402.10312) for making their codebases public. This project builds on their work.


## Setup
```console
python3.12 -m venv venv
source venv/bin/activate
python3 -m pip install -U pip
pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu121
python3 -m pip install -r requirements.txt
python3 -m pip install -e . --no-deps
pip install torch-scatter -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__.split('+')[0])")+cu121.html
```

Note: These commands were tested with CUDA 12.1 (`cu121`). If your machine uses a different CUDA version, adjust the PyTorch index URL and PyG wheel URL accordingly.

The quadrotor/GCS examples use Drake's GCS bindings. Some vanilla GCS solves may use MOSEK through Drake, so install and license MOSEK if you want to reproduce those solver paths exactly.

## Run Planning Through Contact Via Nominal GCS
Validate the setup by ensuring a plan through contact can be successfully generated

Example:

```console
python -m planning_through_contact.scripts.planar_pushing.create_plans --body sugar_box --seed 0 --num 1 --print_profile
```

## Dataset creation

Generate a deterministic plan-index CSV (randomized initial slider poses + train/test split) used for dataset collection:

```console
python3 planning_through_contact/dataset/create_plan_index.py
```

Generate static graph node features which are instance-independent for planar pushing problem:

```console
python3 planning_through_contact/dataset/create_static_node_features.py
```

Collect solved instances into a consolidated HDF5 file:

```console
python3 planning_through_contact/dataset/collect_solutions.py
```

## GNN: training and validation

Download the box-pushing dataset and trained checkpoint before running GNN inference:

```console
./planning_through_contact/dataset/download_box_pushing_assets.sh
```

**Train** the graph neural network (from repo root):

```console
./planning_through_contact/scripts/planar_pushing/train.sh
```

**Test** (GNN inference vs vanilla GCS on the held-out test split; uses `checkpoints/gcs_gnn/reduced_data.ckpt` if present, otherwise `box_pushing.ckpt`):

```console
./planning_through_contact/scripts/planar_pushing/test.sh
```

Inference runs on CPU by default; use `--device cuda` for GPU inference.

After a test run, check graph visualizations in the output directory: **`prediction.svg`** (GNN + rounding pipeline) and **`ground_truth.svg`** (SDP ground truth from H5, when using test plans).

Motion videos are saved by default under **`trajectory/`**: **`prediction.mp4`** (GNN + rounding) and **`ground_truth.mp4`** (SDP + rounding, when using test plans and H5). Both use the same camera view so you can compare motion directly.

Recorded videos now include a top-left legend for pusher contact state (red = non-contact, green = contact) and default to **1920x1080** output.

To run exactly 4 held-out test instances with this visualization and save videos:

```console
./planning_through_contact/scripts/planar_pushing/test_4_instances_record_videos.sh
```