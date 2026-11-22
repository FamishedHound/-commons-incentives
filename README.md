# Incentive-driven MARL for commons dilemmas in land use

> Code for Chapter 3 of the PhD thesis *Multi-Level Agent-Based Land Use Decision-Making Using Multi-Agent Reinforcement Learning* (Lukasz Pelcner, Lancaster University, 2026). The Pollinators' Game, P-MADDPG and ORAA were introduced in our [PRIMA 2024 paper](https://doi.org/10.1007/978-3-031-77367-9_21).

The **Pollinators' Game** is a small commons dilemma in land use. Four landowners share a 4 x 4 grid. At every step a landowner chooses, for each of its cells, the share given over to pollinator habitat and crops the rest; cropped land earns income only if the cell is pollinated. A regulator pays or penalises each landowner to move its habitat towards a target.

The repository has the game, the landowners' learner (**P-MADDPG**), the regulator (**ORAA**, the Online Regulatory Agent Algorithm, planning with learned models of the landowners or with their own policies), a reward-shaping baseline, **LToS** as a second baseline, and the code that runs the experiments and makes the tables and figures of the chapter. The record of every run reported in the chapter is included.

This repository was prepared after the thesis was examined, as part of its corrections, using an AI coding assistant. The game, P-MADDPG, ORAA and the reward-shaping baseline are a cleaned-up and clarified port of the original research code, which is not part of this repository. The LToS baseline, the experiment campaign and the analysis were written for the corrected thesis, and every experiment in Chapter 3 was run again with this code.

If you use this code, please cite the paper (see [Citation](#citation)).

## What is where

| | |
|---|---|
| `chapter3_pollinators/env` | the Pollinators' Game |
| `chapter3_pollinators/learning` | P-MADDPG |
| `chapter3_pollinators/regulators` | ORAA, reward shaping, the training curriculum |
| `chapter3_pollinators/ltos` | LToS baseline |
| `chapter3_pollinators/experiment`, `campaign` | a single run, and the whole campaign |
| `chapter3_pollinators/analysis` | tables and figures |
| `chapter3_pollinators/results/campaign_main` | records of the 620 runs |
| `chapter3_pollinators/results/thesis` | the tables, figures and numbers of Chapter 3 |
| `chapter3_pollinators/tests` | tests |

## Setup

Python 3.10 or newer (tested with 3.13). Running experiments needs a CUDA build of PyTorch; rebuilding the tables from the records does not need a GPU.

```bash
git clone https://github.com/FamishedHound/-commons-incentives.git commons-incentives
cd commons-incentives
pip install torch --index-url https://download.pytorch.org/whl/cu129
pip install -r requirements.txt
```

Run everything from the repository root. ORAA uses Optuna's default TPE sampler, whose defaults change between versions, so `requirements.txt` pins `optuna==4.9.0`.

## Reproducing the results of Chapter 3

**Tables**, from the run records in the repository (a few seconds on the CPU):

```bash
python -m chapter3_pollinators.analysis --root chapter3_pollinators/results/campaign_main \
    --out chapter3_pollinators/results/rebuilt --no-figures
```

**Figures.** The cumulative-budget and time-series figures are drawn from the per-step arrays of the runs. These are attached to the [release](https://github.com/FamishedHound/-commons-incentives/releases) as `chapter3_runs.zip` (about 122 MB). Unpack the archive and run the analysis without `--no-figures`:

```bash
python -m zipfile -e chapter3_runs.zip chapter3_pollinators/results/campaign_main
python -m chapter3_pollinators.analysis --root chapter3_pollinators/results/campaign_main \
    --out chapter3_pollinators/results/rebuilt
```

| In the thesis | File in `results/thesis` (and in `results/rebuilt`) |
|---|---|
| Table 3.3 | `pooled` entries of `summary_<group>.json`, tests in `paired_<group>.tex` |
| Figures 3.5 to 3.8 | `bars_HT_HTC`, `bars_HT_HMC`, `bars_HM_HMC`, `bars_HM_HTC` |
| Figure 3.9 | `budget_cumulative_HT_HTC`, `budget_cumulative_HT_HMC` |
| Figure 3.10 | `ltos_comparison_HT_HTC` |
| Figures 3.11 and 3.14 | `oraa_response_HT_HTC`, `oraa_response_HT_HTC_bounds` |
| Figures 3.12 and 3.13 | `timeseries_HT_HTC`, `timeseries_HT_HMC` |
| Table 3.4 | `table_HT_HTC_bounds.tex` |

`<group>` is `HT_HTC`, `HT_HMC`, `HM_HMC`, `HM_HTC` or `HT_HTC_bounds`. Every figure has a `.csv` with the numbers it draws. `timeseries_HT_HTC` also draws the two conditions with the incentive bound 3 (Section 3.6.1); Figure 3.12 as printed shows the five conditions of Figure 3.5.

Figures 3.11 and 3.14 are drawn from the landowners' networks saved at the end of each warm-up. Those checkpoints are about 41 GB and are not distributed; the numbers behind the two figures are in their `.csv` files, and running the warm-ups again (below) with `--response-curves` in the analysis redraws them.

**Running the experiments again.** A branch or a run without a regulator starts from the warm-up of the same community and seed, so the warm-up comes first (about 1.25 hours on one process).

```bash
# single runs
python -m chapter3_pollinators.experiment.run warmup --community HT1 --seed 0 \
    --root chapter3_pollinators/results/campaign_new
python -m chapter3_pollinators.experiment.run branch --community HT1 --government HTC \
    --condition oraa_mb --seed 0 --root chapter3_pollinators/results/campaign_new
python -m chapter3_pollinators.experiment.run no_regulator --community HT1 --seed 0 \
    --root chapter3_pollinators/results/campaign_new

# the whole campaign (620 runs), then its tables and figures
python -m chapter3_pollinators.campaign launch --root chapter3_pollinators/results/campaign_new \
    --seeds 0 1 2 3 4 5 6 7 8 9 --group3-seeds 0 1 2 3 4 --busy-workers 6 --free-workers 22
python -m chapter3_pollinators.campaign status --root chapter3_pollinators/results/campaign_new
python -m chapter3_pollinators.analysis --root chapter3_pollinators/results/campaign_new \
    --out chapter3_pollinators/results/thesis_new --response-curves
```

Results on another GPU can differ slightly from the published ones.

Tests: `python -m pytest chapter3_pollinators/tests`

## Citation

```bibtex
@inproceedings{pelcner2024incentive,
  title     = {Incentive-driven Multi-Agent Reinforcement Learning Approach for Commons Dilemmas in Land-Use},
  author    = {Pelcner, Lukasz and do Carmo Alves, Matheus Aparecido and Marcolino, Leandro Soriano and Harrison, Paula A. and Atkinson, Peter M.},
  booktitle = {PRIMA 2024: Principles and Practice of Multi-Agent Systems},
  pages     = {284--289},
  publisher = {Springer},
  year      = {2024},
  doi       = {10.1007/978-3-031-77367-9_21}
}

@phdthesis{pelcner2026thesis,
  title  = {Multi-Level Agent-Based Land Use Decision-Making Using Multi-Agent Reinforcement Learning},
  author = {Pelcner, Lukasz},
  school = {Lancaster University},
  year   = {2026}
}
```

## License

MIT, see [LICENSE](LICENSE).
