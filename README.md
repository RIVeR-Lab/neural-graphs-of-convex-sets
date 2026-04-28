# neural-graphs-of-convex-sets

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

## Run Planning Through Contact Via Nominal GCS
Validate the setup by ensuring a plan through contact can be successfully generated

Example:

```console
python -m planning_through_contact.scripts.planar_pushing.create_plans --body sugar_box --seed 0 --num 1
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

**Train** the graph neural network (from repo root):

```console
./planning_through_contact/scripts/planar_pushing/train.sh
```

**Test** (inference / planning with the trained model):

```console
./planning_through_contact/scripts/planar_pushing/test.sh
```

After a test run, check graph visualizations in the output directory: **`prediction.svg`** (GNN + rounding pipeline) and **`ground_truth.svg`** (SDP ground truth from H5, when using test plans).

Motion videos are saved by default under **`trajectory/`**: **`prediction.mp4`** (GNN + rounding) and **`ground_truth.mp4`** (SDP + rounding, when using test plans and H5). Both use the same camera view so you can compare motion directly.

Recorded videos now include a top-left legend for pusher contact state (red = non-contact, green = contact) and default to **1920x1080** output.

To run exactly 4 held-out test instances with this visualization and save videos:

```console
./planning_through_contact/scripts/planar_pushing/test_4_instances_record_videos.sh
```