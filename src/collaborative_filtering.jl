

struct UserCFRecommender
    user_items::Dict{String, Set{String}}
    item_users::Dict{String, Vector{String}}
    k_neighbors::Int
end

function UserCFRecommender(behaviors::DataFrame; k::Int=20)::UserCFRecommender
    user_items, item_users = build_user_item_data(behaviors)
    return UserCFRecommender(user_items, item_users, k)
end


function score_candidates_ucf(rec::UserCFRecommender, user_id::String,
                               candidates::Vector{String},
                               history::Vector{String})::Dict{String,Float64}
    scores = Dict(nid => 0.0 for nid in candidates)

    isempty(history) && return scores

    history_set = Set(history)
    n_history   = length(history_set)

    overlap_count = Dict{String,Int}()
    for nid in history
        if haskey(rec.item_users, nid)
            for uid in rec.item_users[nid]
                uid == user_id && continue
                overlap_count[uid] = get(overlap_count, uid, 0) + 1
            end
        end
    end

    isempty(overlap_count) && return scores

    neighbor_sims = Tuple{String,Float64}[]
    for (uid, overlap) in overlap_count
        n_neighbor = length(rec.user_items[uid])
        sim = overlap / sqrt(n_history * n_neighbor)
        push!(neighbor_sims, (uid, sim))
    end
    sort!(neighbor_sims, by=x -> x[2], rev=true)
    top_k = neighbor_sims[1:min(rec.k_neighbors, end)]

    total_sim = sum(s for (_, s) in top_k)
    total_sim == 0.0 && return scores

    for (uid, sim) in top_k
        neighbor_items = rec.user_items[uid]
        weight = sim / total_sim
        for nid in candidates
            if nid in neighbor_items
                scores[nid] += weight
            end
        end
    end

    return scores
end
