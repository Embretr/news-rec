
using Statistics


# ─── Accuracy metrics ─────────────────────────────────────────────────────────

function auc_score(labels::Vector{Int}, scores::Vector{Float64})::Float64
    n_pos = sum(labels)
    n_neg = length(labels) - n_pos
    (n_pos == 0 || n_neg == 0) && return 0.5

    order = sortperm(scores, rev=true)
    tp = 0
    concordant = 0
    for idx in order
        if labels[idx] == 1
            tp += 1
        else
            concordant += tp
        end
    end
    return concordant / (n_pos * n_neg)
end


function _dcg_at_k(sorted_labels::Vector{Int}, k::Int)::Float64
    dcg = 0.0
    for i in 1:min(k, length(sorted_labels))
        dcg += sorted_labels[i] / log2(i + 1)
    end
    return dcg
end

function ndcg_at_k(labels::Vector{Int}, scores::Vector{Float64}, k::Int)::Float64
    order         = sortperm(scores, rev=true)
    actual_dcg    = _dcg_at_k(labels[order], k)
    ideal_dcg     = _dcg_at_k(sort(labels, rev=true), k)
    ideal_dcg == 0.0 && return 0.0
    return actual_dcg / ideal_dcg
end


function mrr_score(labels::Vector{Int}, scores::Vector{Float64})::Float64
    order = sortperm(scores, rev=true)
    for (rank, idx) in enumerate(order)
        labels[idx] == 1 && return 1.0 / rank
    end
    return 0.0
end


# ─── Beyond-accuracy metrics ──────────────────────────────────────────────────

function _cosine_sim(a::Dict{String,Float64}, b::Dict{String,Float64})::Float64
    sim = 0.0
    for (t, v) in a
        haskey(b, t) && (sim += v * b[t])
    end
    return sim
end

# Average pairwise cosine dissimilarity of the top-k ranked items.
function intra_list_diversity(ranked_ids::Vector{String},
                               news_tfidf::Dict{String,Dict{String,Float64}},
                               k::Int)::Float64
    top_k = [nid for nid in ranked_ids[1:min(k, end)] if haskey(news_tfidf, nid)]
    length(top_k) < 2 && return 0.0

    total = 0.0
    pairs = 0
    for i in 1:length(top_k), j in (i+1):length(top_k)
        total += 1.0 - _cosine_sim(news_tfidf[top_k[i]], news_tfidf[top_k[j]])
        pairs  += 1
    end
    return total / pairs
end

# Mean self-information of recommended items: -log2(p(item)).
# Higher = recommending less popular / more surprising items.
function novelty_score(ranked_ids::Vector{String},
                        popularity::Dict{String,Int},
                        n_total_clicks::Int,
                        k::Int)::Float64
    top_k = ranked_ids[1:min(k, end)]
    isempty(top_k) && return 0.0

    total = 0.0
    for nid in top_k
        p = get(popularity, nid, 0) / max(n_total_clicks, 1)
        total += p > 0 ? -log2(p) : 0.0
    end
    return total / length(top_k)
end

# Fraction of the catalog that appears in at least one top-k recommendation list.
function catalog_coverage(all_ranked::Vector{Vector{String}},
                           n_catalog_items::Int,
                           k::Int)::Float64
    n_catalog_items == 0 && return 0.0
    seen = Set{String}()
    for ranked in all_ranked
        for nid in ranked[1:min(k, end)]
            push!(seen, nid)
        end
    end
    return length(seen) / n_catalog_items
end


# ─── Evaluation harness ───────────────────────────────────────────────────────

struct EvalResult
    name::String
    n_sessions::Int
    auc::Float64
    ndcg::Float64
    mrr::Float64
    diversity::Float64
    novelty::Float64
    coverage::Float64
end

function evaluate_recommender(
    name::String,
    scorer::Function,           # (user_id, candidates, history) -> Dict{String,Float64}
    behaviors::DataFrame,
    popularity::Dict{String,Int},
    news_tfidf::Dict{String,Dict{String,Float64}},
    n_catalog::Int;
    k::Int=5,
)::EvalResult
    aucs        = Float64[]
    ndcgs       = Float64[]
    mrrs        = Float64[]
    diversities = Float64[]
    novelties   = Float64[]
    all_ranked  = Vector{String}[]

    n_total_clicks = sum(values(popularity))

    for row in eachrow(behaviors)
        impr = parse_impressions(row.Impressions)
        isempty(impr) && continue

        candidates = [nid for (nid, _) in impr]
        labels     = [clicked for (_, clicked) in impr]
        history    = parse_history(row.History)

        sum(labels) == 0 && continue   # skip no-positive sessions

        scores_dict = scorer(row.UserID, candidates, history)
        scores      = [get(scores_dict, nid, 0.0) for nid in candidates]

        push!(aucs,   auc_score(labels, scores))
        push!(ndcgs,  ndcg_at_k(labels, scores, k))
        push!(mrrs,   mrr_score(labels, scores))

        ranked = sort(candidates, by=nid -> get(scores_dict, nid, 0.0), rev=true)
        push!(diversities, intra_list_diversity(ranked, news_tfidf, k))
        push!(novelties,   novelty_score(ranked, popularity, n_total_clicks, k))
        push!(all_ranked,  ranked)
    end

    cov = catalog_coverage(all_ranked, n_catalog, k)

    return EvalResult(
        name,
        length(aucs),
        isempty(aucs) ? 0.0 : mean(aucs),
        isempty(ndcgs) ? 0.0 : mean(ndcgs),
        isempty(mrrs) ? 0.0 : mean(mrrs),
        isempty(diversities) ? 0.0 : mean(diversities),
        isempty(novelties) ? 0.0 : mean(novelties),
        cov,
    )
end


function print_eval_table(results::Vector{EvalResult})
    header = ("Recommender", "Sessions", "AUC", "NDCG@5", "MRR", "Diversity", "Novelty", "Coverage")
    @printf("%-18s  %8s  %6s  %6s  %6s  %9s  %7s  %8s\n", header...)
    println(repeat('-', 84))
    for r in results
        @printf("%-18s  %8d  %6.4f  %6.4f  %6.4f  %9.4f  %7.4f  %8.4f\n",
            r.name, r.n_sessions, r.auc, r.ndcg, r.mrr, r.diversity, r.novelty, r.coverage)
    end
end
