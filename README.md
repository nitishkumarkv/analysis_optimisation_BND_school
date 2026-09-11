# Analysis optimisation

This repository contains the starter material for the **Analysis Optimisation** ([Indico link](https://indico.global/event/17239/contributions/164027/)) project of the BND Graduate School 2026.

# Installing `gato-hep` in Editable Mode

`gato-hep` requires **Python 3.10 or newer**. See `pyproject.toml` for the authoritative dependency requirements.

## Option 1 — Python virtual environment

```bash
git clone https://github.com/FloMau/gato-hep.git
cd gato-hep

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e ".[dev]"
cd ..
```

## Option 2 — Micromamba environment

```bash
micromamba create -n gato python=3.10 -y
micromamba activate gato

git clone https://github.com/FloMau/gato-hep.git
cd gato-hep

python -m pip install --upgrade pip
pip install -e ".[dev]"
cd ..
```

The `-e` option installs `gato-hep` in **editable mode**, so changes made to the local source code are immediately available without reinstalling the package.

You can verify the installation with:

```bash
python -c "import gatohep; print('gato-hep successfully installed')"
```



# How to get the Data
```
wget "https://cernbox.cern.ch/s/ckRpXdr1iTsgJJH/download" -O data_for_BND_school.tar.gz
tar -xzf data_for_BND_school.tar.gz
```

# Details of the Data

Two dataset sizes are provided for the project:

- **~1M events**
- **~5M events**

Both datasets are derived from the full **FAIR Universe $H\rightarrow \tau\tau$ dataset**.

The event weights are rescaled such that the total normalization corresponds to the full dataset. In addition, a **post-selection is applied**, unlike the setup used in the *Learning to bin* ([Eur. Phys. J. C 86, 1014 (2026)](https://link.springer.com/article/10.1140/epjc/s10052-026-16255-1)) paper.

For final optimizations and significance calculations, it is recommended to use the **5M-event samples**, since limited statistics can affect the stability of the significance estimate. The **1M-event samples** are useful for quick tests, code development, and debugging.

Each dataset folder contains `.pkl` files. Each `.pkl` file stores a Python dictionary of pandas DataFrames, with one DataFrame for each process.

The available dictionary keys are:

- `ztautau`
- `diboson`
- `ttbar`
- `htautau`

Each DataFrame contains:

- the input features used for classifier training,
- an event-weight column (`weight`),
- an `NN_output` column containing the output of a pretrained four-class neural network.

The `NN_output` column is four-dimensional, with the classifier scores ordered as

`[ztautau, diboson, ttbar, htautau]`

That is:

- `NN_output[:, 0]` → `ztautau` score
- `NN_output[:, 1]` → `diboson` score
- `NN_output[:, 2]` → `ttbar` score
- `NN_output[:, 3]` → `htautau` score

## Project 1 — JES-aware optimisation

For the JES systematic-variation project, use:

- `gato_like_scores_nominal.pkl`
- `gato_like_scores_jes_up.pkl`
- `gato_like_scores_jes_down.pkl`

Use `gato_like_scores_nominal.pkl` as the nominal dataset and the corresponding JES-up and JES-down files as the systematic variations.

The nominal, JES-up, and JES-down samples should be used together when constructing the systematics-aware optimisation objective.

## Project 2 — BSM-oriented optimisation

For the BSM-oriented project, use:

`gato_like_scores_with_BSM_weights.pkl`

The signal DataFrame already contains three event-weight columns:

- `weight` — nominal signal hypothesis
- `weight_hard` — harder Higgs-$p_T$-like signal hypothesis
- `weight_soft` — softer Higgs-$p_T$-like signal hypothesis

The kinematic features and classifier scores are unchanged. The different signal hypotheses are constructed by reweighting the signal events.

The background weights should remain unchanged.

## Project 3 — $t\bar{t}$ control-region optimisation

For the $t\bar{t}$ control-region optimisation project, use:

`gato_like_scores_nominal.pkl`

The nominal multiclass classifier outputs can be used to construct:

- Higgs-enriched signal categories,
- and a $t\bar{t}$-enriched control region.

The `ttbar` classifier score is available as the third component of `NN_output`.

# Examples

For additional examples of how to use GATO, see the [`gato-hep` GitHub repository](https://github.com/FloMau/gato-hep).

This repository also provides a simple example script demonstrating a GATO optimisation using the nominal dataset.

Run the example with:

```bash
python3 example_script.py