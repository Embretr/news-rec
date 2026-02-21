using Statistics

"""
Compute AUC for a single impression.
scores: predicted scores for candidate items
labels: true labels (0/1)
"""
function auc(scores::Vector{Float64}, labels::Vector{Int})::Float64
    pos = scores[labels .== 1]
    neg = scores[labels .== 0]

    total = length(pos) * length(neg)
    total == 0 && return 0.0

    count = 0.0
    for p in pos
        for n in neg
            if p > n
                count += 1
            elseif p == n
                count += 0.5
            end
        end
    end

    return count / total
end

"""
Compute MRR for a single impression.
Assumes at least one positive item.
"""
function mrr(scores::Vector{Float64}, labels::Vector{Int})::Float64
    # sort indices by score (descending)
    order = sortperm(scores, rev=true)

    for (rank, idx) in enumerate(order)
        if labels[idx] == 1
            return 1.0 / rank
        end
    end

    return 0.0
end

"""
Compute nDCG@k for a single impression (binary relevance).
"""
function ndcg(scores::Vector{Float64}, labels::Vector{Int}; k::Int=10)::Float64
    # sort indices by predicted score (descending)
    order = sortperm(scores, rev=true)

    dcg = 0.0
    for i in 1:min(k, length(order))
        idx = order[i]
        if labels[idx] == 1
            dcg += 1 / log2(i + 1)
        end
    end

    # Ideal DCG (one positive item)
    idcg = 1 / log2(1 + 1)

    return dcg / idcg
end

function evaluate_accuracy(behaviors::DataFrame, score_fn; max_rows=1000)

    auc_list = Float64[]
    mrr_list = Float64[]
    ndcg_list = Float64[]

    count = 0

    for row in eachrow(first(behaviors, max_rows))

        impressions = parse_impressions(row.Impressions)
        isempty(impressions) && continue

        candidates = [nid for (nid, _) in impressions]
        labels     = [clicked for (_, clicked) in impressions]

        history = parse_history(row.History)
        user_id = row.UserID

        scores_dict = score_fn(user_id, candidates, history)
        scores = [get(scores_dict, nid, 0.0) for nid in candidates]

        push!(auc_list, auc(scores, labels))
        push!(mrr_list, mrr(scores, labels))
        push!(ndcg_list, ndcg(scores, labels))

        count += 1
        count % 200 == 0 && println("Processed ", count)
    end

    return mean(auc_list), mean(mrr_list), mean(ndcg_list)
end