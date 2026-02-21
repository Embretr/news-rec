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
Evaluate AUC over all impressions for a given scoring function.
score_fn must return Dict{NewsID => score}
"""
function evaluate_auc(behaviors::DataFrame, score_fn)
    auc_list = Float64[]

    for row in eachrow(first(behaviors, 1000))

        impressions = parse_impressions(row.Impressions)
        isempty(impressions) && continue

        candidates = [nid for (nid, _) in impressions]
        labels     = [clicked for (_, clicked) in impressions]

        history = parse_history(row.History)
        user_id = row.UserID

        scores_dict = score_fn(user_id, candidates, history)
        scores = [get(scores_dict, nid, 0.0) for nid in candidates]

        push!(auc_list, auc(scores, labels))
    end

    return mean(auc_list)
end