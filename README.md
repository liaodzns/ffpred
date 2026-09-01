# ffpred

A machine learning model that projects weekly fantasy football points for NFL skill
position players, built on public nflverse data.

The model looks at what a player and his team have done in previous weeks, combines that
with context about the upcoming game, and produces a projected point total for each player
at each position. The intended use is making start, sit, and waiver wire decisions in a
weekly redraft league.

## What this project is not

This is not a daily fantasy sports optimizer. It does not handle salary caps, lineup
constraints, or ownership projections. It also does not model kickers or team defenses,
because those positions are close to random from week to week and are not worth the effort.

## League scoring

The model is trained on this specific league's scoring rules rather than a generic format.
The league is full point per reception at its base, with four differences from a standard
PPR setup:

- Passing yards are worth one point per 20 yards, rather than the more common 25.
- Interceptions cost one point, rather than two.
- A 100 yard rushing game earns a three point bonus.
- A 100 yard receiving game earns a one point bonus.

Because of the bonuses and the adjusted passing values, the fantasy point columns that come
prepackaged in the nflverse data are not usable as the prediction target. The target is
computed from raw box score statistics instead, using the rules in `config/config.yaml`.
If your league differs, editing the `scoring` block of that file is the only change needed.

## Requirements

- Python 3.11 or newer
- Roughly 2 GB of free disk space for cached data

## Setup

```bash
git clone <your-repo-url>
cd fantasy-football-ml

# Using uv, which is recommended
uv sync

# Or using pip in a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Usage

The pipeline runs in four stages, in order. Each stage writes its output to disk, so a
later stage can be rerun without repeating the work of an earlier one.

```bash
# 1. Download raw data from the nflverse repositories into data/raw/
python scripts/pull_data.py

# 2. Clean, join, and build the feature matrix into data/processed/
python scripts/build_features.py

# 3. Train one model per position and save them to models/
python scripts/train_model.py

# 4. Project the coming week into data/predictions/
python scripts/predict_week.py
```

`predict_week.py` defaults to the current NFL week. To project a different one:

```bash
python scripts/predict_week.py --season 2025 --week 7
```

During the season, you only need to rerun the first two steps each week to pick up new
results. Retraining is worth doing every few weeks rather than daily.

Run the tests with:

```bash
pytest
```

## Configuration

Nearly every setting lives in `config/config.yaml`, including which seasons to pull, the
league scoring rules, rolling window sizes, model hyperparameters, and the validation
split. Change settings there rather than editing source files, so that experiments stay
reproducible.

## Data sources

All data comes from the [nflverse](https://github.com/nflverse) project through the
[nflreadpy](https://github.com/nflverse/nflreadpy) package. The main tables used are weekly
player statistics, game schedules with betting lines, snap counts, weekly rosters, expected
fantasy points from the ffopportunity project, depth charts, and injury reports.

nflreadpy returns Polars DataFrames. This project standardizes on pandas, so every frame is
converted immediately on arrival inside `src/ffml/data/ingest.py`. That file is the only
place in the codebase that touches nflreadpy or Polars.

Data is pulled fresh from public repositories, so nothing in `data/` is committed to
version control. If you lose it, pull it again.

## Project layout

```
config/       settings, including league scoring rules
data/         raw pulls, processed tables, and predictions (all gitignored)
models/       saved models and feature lists (gitignored)
notebooks/    exploratory analysis
scripts/      thin command line entry points
src/ffml/     the package itself
  data/       downloading, cleaning, and joining
  features/   feature engineering
  models/     baseline, training, evaluation, prediction
  utils/      shared input and output helpers
tests/        test suite, including the leakage check
```

## How the model works

**One model per position.** Quarterbacks, running backs, receivers, and tight ends produce
fantasy points through different processes, so each gets its own model. A single shared
model would spend most of its effort learning that quarterbacks score more than everyone
else, which is already known.

**Gradient boosted trees.** The model is LightGBM. The data is tabular with plenty of
non-linear interactions, which is the setting where boosted trees still beat neural
networks, and LightGBM handles the missing values that injury and snap data inevitably
contain.

**A baseline to beat.** Every model is scored against a naive predictor that simply projects
each player's average over his last three games. Sports models frequently fail to beat
simple averages, and reporting the two side by side is the only way to know whether the
machine learning is earning its keep.

**Time based validation.** Training uses earlier seasons and validation uses later ones. A
random train and test split would leak future information backward and produce scores that
look excellent and mean nothing.

**Ranking matters more than raw error.** Alongside mean absolute error and root mean squared
error, the model reports rank correlation within each position and week. The real decision
a manager makes is which of two players to start, not what either one will score exactly.

## The most important rule in this codebase

Every feature must contain only information that existed before kickoff of the game being
predicted. Rolling averages must be shifted so they never include the week being projected.
This is the mistake that quietly ruins most sports models, because it makes validation
scores look wonderful while real predictions stay useless. `tests/test_no_leakage.py` guards
against it and should always pass.


## Roadmap

Planned but not yet built:

- Quantile models that project a floor and a ceiling in addition to an expected value,
  which is more useful for start and sit decisions than a single number.
- Predicting rushing and receiving yards separately, then applying the scoring rules, so
  that the 100 yard bonuses are handled exactly rather than smoothed over.
- Play by play features such as red zone touches and air yards by field position.

## Data license

nflverse data is released under CC-BY 4.0, with the FTN charting data under CC-BY-SA 4.0.
This project is for personal use.