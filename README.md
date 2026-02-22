# TDT4215 Recommender systems group project

# Setup

## Download the datasets
If you haven't already, download the datasets here:
```bash
wget https://huggingface.co/datasets/yjw1029/MIND/resolve/main/MINDsmall_train.zip
wget https://huggingface.co/datasets/yjw1029/MIND/resolve/main/MINDsmall_dev.zip
```
...these are the small datasets for development.


When you want the big ones, they are here:
```bash
wget https://huggingface.co/datasets/yjw1029/MIND/resolve/main/MINDlarge_train.zip
wget https://huggingface.co/datasets/yjw1029/MIND/resolve/main/MINDlarge_dev.zip
wget https://huggingface.co/datasets/yjw1029/MIND/resolve/main/MINDlarge_test.zip
```

Unzip them in a subfolder, we assume this structure:
```bash
MIND
├── MINDsmall_dev
│   ├── behaviors.tsv
│   ├── entity_embedding.vec
│   ├── news.tsv
│   └── relation_embedding.vec
└── MINDsmall_train
    ├── behaviors.tsv
    ├── entity_embedding.vec
    ├── news.tsv
    └── relation_embedding.vec
```

## 2. Install Julia

1. Download Julia from https://julialang.org/downloads/ and install it.
2. Add Julia to your PATH (or use the full path to the `julia` executable).
3. Verify the installation:

	```bash
	julia --version
	```

## 3. Set up the Julia environment

From the repository root:

```bash
julia --project=.
```

Then, while in the Julia REPL, activate and instantiate the project:

```julia
] activate .
] instantiate
```

If there is no `Project.toml` yet, create one by activating the environment
and adding packages as needed:

```julia
] add CSV DataFrames Flux TextAnalysis ZipFile Dates Downloads Printf Random
```

Exit the REPL with `Ctrl-Z`.

Running Julia programs from the project root:
```bash
julia src/main.jl
```

## 4. How it works

This project builds and compares multiple news recommendation strategies on the
MIND dataset, then shows ranked candidate articles for a sample user session.

### End-to-end pipeline

1. **Load data** from `MIND/MINDsmall_train` and `MIND/MINDsmall_dev`.
2. **Preprocess fields** in `news.tsv` and `behaviors.tsv`.
3. **Build models**:
     - Popularity baseline
     - User-based collaborative filtering
     - Content-based TF-IDF model
     - Hybrid weighted model
4. **Score candidate impressions** for one dev behavior row.
5. **Print top-5 ranked candidates** per method for quick comparison.

The main script is:

- `src/main.jl`

### Dataset format used by this code

- `news.tsv` provides article metadata (`NewsID`, `Category`, `SubCategory`,
    `Title`, `Abstract`, ...).
- `behaviors.tsv` provides user sessions (`UserID`, `History`, `Impressions`).
    - `History`: space-separated previously clicked/read `NewsID`s.
    - `Impressions`: space-separated `NewsID-label` pairs (e.g. `N12345-1`).

### Module overview

- `src/preprocessing.jl`
    - Loads/sanitizes tab-separated files.
    - Parses history/impressions.
    - Builds user-item and item-user interaction structures.

- `src/baseline.jl`
    - `compute_popularity`: counts clicks/read interactions globally.
    - `score_popularity`: assigns candidate scores from those global counts.
    - `score_random`: random baseline for sanity checks.

- `src/collaborative_filtering.jl`
    - Defines `UserCFRecommender` with sparse interaction maps.
    - Uses user overlap with cosine-style normalization to find neighbors.
    - Scores candidates by weighted neighbor support.

- `src/content_based.jl`
    - Tokenizes article text (`Title`, `Abstract`, `Category`, `SubCategory`).
    - Builds normalized TF-IDF vectors per article.
    - Builds a user content profile from history.
    - Scores candidates by profile/article similarity (dot product).

- `src/hybrid.jl`
    - Min-max normalizes each component score set.
    - Combines models with weighted fusion:
        - default weights: `w_pop = 0.2`, `w_cf = 0.3`, `w_cbf = 0.5`
    - Final score per candidate:
        - `hybrid = w_pop * pop_norm + w_cf * cf_norm + w_cbf * cbf_norm`

### What happens when you run `julia src/main.jl`

- Loads train/dev splits.
- Trains/builds all model components.
- Selects the first dev behavior row.
- Extracts candidate `NewsID`s from impressions.
- Computes scores from all methods.
- Ranks candidates and prints top-5 per method.

### Notes and assumptions

- This script is a **demonstration pipeline** (single-session printed output),
    not yet a full offline evaluation benchmark.
- Randomness is seeded (`Random.seed!(42)`) for reproducible random baseline
    behavior.
- The code currently points to the **MINDsmall** split in `src/main.jl`.
