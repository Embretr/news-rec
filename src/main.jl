
using CSV
using DataFrames
using Statistics
using Printf
using Random

# Entry-point script for loading MIND data, building recommenders, and running a demo session.
const SRC = @__DIR__
include(joinpath(SRC, "preprocessing.jl"))
include(joinpath(SRC, "baseline.jl"))
include(joinpath(SRC, "collaborative_filtering.jl"))
include(joinpath(SRC, "content_based.jl"))
include(joinpath(SRC, "hybrid.jl"))
include(joinpath(SRC, "metrics.jl"))
include(joinpath(SRC, "analysis.jl"))

# Deterministic behavior for reproducible random baseline outputs.
Random.seed!(42)

# Project data locations (train/dev of MINDsmall split).
const ROOT      = joinpath(@__DIR__, "..")
const TRAIN_DIR = joinpath(ROOT, "MIND", "MINDsmall_train")
const DEV_DIR   = joinpath(ROOT, "MIND", "MINDsmall_dev")

# ── Load data ─────────────────────────────────────────────────────────────────
println("=== Loading and preprocessing data ===")
train = preprocess_data(TRAIN_DIR)
dev   = preprocess_data(DEV_DIR)
println("  train: $(nrow(train.news)) articles, $(nrow(train.behaviors)) sessions")
println("  dev  : $(nrow(dev.news))   articles, $(nrow(dev.behaviors))   sessions")

# ── Dataset analysis (Issue 1) ────────────────────────────────────────────────
analyze_dataset("MINDsmall_train", train.news, train.behaviors)
analyze_dataset("MINDsmall_dev",   dev.news,   dev.behaviors)
print_dataset_weaknesses()

# ── Build models ──────────────────────────────────────────────────────────────
println("=== Building models ===")

# 1) Global popularity model from training interactions.
print("  [1/4] Computing popularity scores…  ")
popularity = compute_popularity(train.behaviors)
println("done  ($(length(popularity)) articles seen)")

# 2) User-based collaborative filtering model.
print("  [2/4] Building user-based CF model… ")
cf_rec = UserCFRecommender(train.behaviors; k=20)
println("done  ($(length(cf_rec.user_items)) users)")

# 3) Content-based TF-IDF index over available news metadata.
print("  [3/4] Building TF-IDF index…        ")
all_news   = vcat(train.news, dev.news)
news_tfidf = build_tfidf(unique(all_news, :NewsID))
println("done  ($(length(news_tfidf)) articles indexed)")

println("  [4/4] Hybrid model uses components above — no extra build step.")

# ── Evaluate all recommenders on dev set (Issues 6 & 7) ─────────────────────
println("\n=== Evaluating on dev set (k=5) ===")

n_catalog = length(dev.all_news_ids)

scorers = [
    ("Random",     (uid, cands, hist) -> score_random(uid, cands, hist)),
    ("Popularity", (uid, cands, hist) -> score_popularity(uid, cands, hist, popularity)),
    ("User-CF",    (uid, cands, hist) -> score_candidates_ucf(cf_rec, uid, cands, hist)),
    ("CBF",        (uid, cands, hist) -> score_candidates_cbf(uid, cands, hist, news_tfidf)),
    ("Hybrid",     (uid, cands, hist) -> score_hybrid(uid, cands, hist, popularity, cf_rec, news_tfidf)),
]

results = EvalResult[]
for (name, scorer) in scorers
    print("  Evaluating $name…")
    r = evaluate_recommender(name, scorer, dev.behaviors, popularity, news_tfidf, n_catalog; k=5)
    push!(results, r)
    println(" done  ($(r.n_sessions) sessions)")
end

println()
print_eval_table(results)
println("\nDone.")
