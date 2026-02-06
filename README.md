# neural-graphs-of-convex-sets

## Setup

From the repo root:

```console
python3.12 -m venv venv
source venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -r requirements.txt
python3 -m pip install -e . --no-deps
```

## Run Planning Through Contact Via Nominal GCS

Example (generate 1 plan):

```console
python -m planning_through_contact.scripts.planar_pushing.create_plans --body sugar_box --seed 0 --num 1
```
