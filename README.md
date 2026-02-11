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
```julia
julia src/download.jl
```

