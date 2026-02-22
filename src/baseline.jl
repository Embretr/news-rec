
using Random

# Baseline rankers: popularity and random scoring.

"""Aggregate click-like interactions into global news popularity counts."""
function compute_popularity(behaviors::DataFrame)::Dict{String,Int}
    click_counts = Dict{String,Int}()

    for row in eachrow(behaviors)
        for nid in parse_history(row.History)
            click_counts[nid] = get(click_counts, nid, 0) + 1
        end
        for (nid, clicked) in parse_impressions(row.Impressions)
            if clicked == 1
                click_counts[nid] = get(click_counts, nid, 0) + 1
            end
        end
    end

    return click_counts
end

"""Score candidates by global popularity (same score for all users)."""
function score_popularity(candidates::Vector{String},
                          popularity::Dict{String,Int})::Dict{String,Float64}
    return Dict(nid => Float64(get(popularity, nid, 0)) for nid in candidates)
end


"""Assign random scores; useful as a quick sanity baseline."""
function score_random(candidates::Vector{String})::Dict{String,Float64}
    return Dict(nid => rand() for nid in candidates)
end
