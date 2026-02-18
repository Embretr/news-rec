
using CSV
using DataFrames
using Statistics
using Printf
using Random

const SRC = @__DIR__
include(joinpath(SRC, "preprocessing.jl"))
include(joinpath(SRC, "analysis.jl"))
include(joinpath(SRC, "baseline.jl"))
include(joinpath(SRC, "collaborative_filtering.jl"))
include(joinpath(SRC, "content_based.jl"))
include(joinpath(SRC, "hybrid.jl"))

Random.seed!(42)

const ROOT      = joinpath(@__DIR__, "..")
const TRAIN_DIR = joinpath(ROOT, "MIND", "MINDsmall_train")
const DEV_DIR   = joinpath(ROOT, "MIND", "MINDsmall_dev")

println("=== Loading training data ===")
train_news, train_behaviors = load_mind(TRAIN_DIR)
println("  news rows      : ", nrow(train_news))
println("  behavior rows  : ", nrow(train_behaviors))

println("\n=== Loading dev/validation data ===")
dev_news, dev_behaviors = load_mind(DEV_DIR)
println("  news rows      : ", nrow(dev_news))
println("  behavior rows  : ", nrow(dev_behaviors))

println()
analyze_dataset(train_news, train_behaviors)

println("\n=== Building models ===")

print("  [1/4] Computing popularity scores…  ")
popularity = compute_popularity(train_behaviors)
println("done  ($(length(popularity)) articles seen)")

print("  [2/4] Building user-based CF model… ")
cf_rec = UserCFRecommender(train_behaviors; k=20)
println("done  ($(length(cf_rec.user_items)) users)")

print("  [3/4] Building TF-IDF index…        ")
all_news   = vcat(train_news, dev_news)
news_tfidf = build_tfidf(unique(all_news, :NewsID))
println("done  ($(length(news_tfidf)) articles indexed)")

println("  [4/4] Hybrid model uses components above — no extra build step.")

println("\n=== Demo: scoring the first dev session ===")

row        = first(dev_behaviors, 1)[1, :]
candidates = [nid for (nid, _) in parse_impressions(row.Impressions)]
history    = parse_history(row.History)
user_id    = row.UserID

println("User: $user_id  |  history length: $(length(history))  |  candidates: $(length(candidates))")

pop_scores    = score_popularity(user_id, candidates, history, popularity)
cf_scores     = score_candidates_ucf(cf_rec, user_id, candidates, history)
cbf_scores    = score_candidates_cbf(user_id, candidates, history, news_tfidf)
hybrid_scores = score_hybrid(user_id, candidates, history, popularity, cf_rec, news_tfidf)

println("\nTop-5 candidates by each method:")
for (name, scores) in [("Popularity", pop_scores), ("User-CF", cf_scores),
                        ("CBF", cbf_scores),        ("Hybrid",  hybrid_scores)]
    ranked = sort(candidates, by=nid -> get(scores, nid, 0.0), rev=true)
    @printf("  %-12s %s\n", name, join(ranked[1:min(5, end)], "  "))
end

println("\nDone.")
