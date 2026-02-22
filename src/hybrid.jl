
# Hybrid ranker combines popularity, user-CF, and content-based scores.

"""Min-max normalize score dictionaries to [0, 1] for stable weighted fusion."""
function _minmax_normalise(scores::Dict{String,Float64})::Dict{String,Float64}
    isempty(scores) && return scores
    lo = minimum(values(scores))
    hi = maximum(values(scores))
    span = hi - lo
    span == 0.0 && return Dict(k => 0.0 for k in keys(scores))
    return Dict(k => (v - lo) / span for (k, v) in scores)
end


"""
Compute final hybrid scores by:
1) generating component scores,
2) normalizing each component,
3) combining with user-configurable weights.
"""
function score_hybrid(user_id::String, candidates::Vector{String},
                      history::Vector{String},
                      popularity::Dict{String,Int},
                      cf_rec::UserCFRecommender,
                      news_tfidf::Dict{String, Dict{String,Float64}};
                      w_pop::Float64=0.2,
                      w_cf::Float64=0.3,
                      w_cbf::Float64=0.5)::Dict{String,Float64}
    @assert isapprox(w_pop + w_cf + w_cbf, 1.0; atol=1e-6) "Weights must sum to 1"

    pop_raw  = score_popularity(user_id, candidates, history, popularity)
    cf_raw   = score_candidates_ucf(cf_rec, user_id, candidates, history)
    cbf_raw  = score_candidates_cbf(user_id, candidates, history, news_tfidf)

    pop_norm = _minmax_normalise(pop_raw)
    cf_norm  = _minmax_normalise(cf_raw)
    cbf_norm = _minmax_normalise(cbf_raw)

    hybrid = Dict{String,Float64}()
    for nid in candidates
        hybrid[nid] = (w_pop * get(pop_norm,  nid, 0.0) +
                       w_cf  * get(cf_norm,   nid, 0.0) +
                       w_cbf * get(cbf_norm,  nid, 0.0))
    end

    return hybrid
end
