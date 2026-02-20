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