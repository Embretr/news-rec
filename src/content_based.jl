
# Content-based recommendation with lightweight TF-IDF features.

"""Lowercase, strip punctuation, and split free text into tokens."""
function tokenize(text::Union{AbstractString,Missing})::Vector{String}
    ismissing(text) && return String[]
    cleaned = replace(lowercase(text), r"[^a-z0-9\s]" => " ")
    return filter(!isempty, String.(split(cleaned)))
end


"""
Build normalized TF-IDF vectors per news article from title, abstract, and category fields.
Returns `news_id => term-weight map`.
"""
function build_tfidf(news::DataFrame;
                     max_features::Int=15_000)::Dict{String, Dict{String,Float64}}
    doc_tf = Dict{String, Dict{String,Int}}()

    for row in eachrow(news)
        nid   = row.NewsID
        terms = vcat(tokenize(row.Title),
                     tokenize(row.Abstract),
                     tokenize(row.Category),
                     tokenize(row.SubCategory))

        tf = Dict{String,Int}()
        for t in terms
            tf[t] = get(tf, t, 0) + 1
        end
        doc_tf[nid] = tf
    end

    df = Dict{String,Int}()
    for (_, tf) in doc_tf
        for term in keys(tf)
            df[term] = get(df, term, 0) + 1
        end
    end

    sorted_terms = sort(collect(df), by=x -> x[2], rev=true)
    vocab = Set(t for (t, _) in sorted_terms[1:min(max_features, end)])

    n_docs = length(doc_tf)

    news_tfidf = Dict{String, Dict{String,Float64}}()

    for (nid, tf) in doc_tf
        doc_len = sum(values(tf))
        tfidf   = Dict{String,Float64}()

        for (term, cnt) in tf
            term in vocab || continue
            tf_score  = cnt / doc_len
            idf_score = log((n_docs + 1) / (df[term] + 1)) + 1.0  # smoothed
            tfidf[term] = tf_score * idf_score
        end

        norm = sqrt(sum(v^2 for v in values(tfidf)))
        if norm > 0.0
            for t in keys(tfidf)
                tfidf[t] /= norm
            end
        end

        news_tfidf[nid] = tfidf
    end

    return news_tfidf
end


"""Build a user profile by averaging TF-IDF vectors of clicked/read history items."""
function build_user_content_profile(history::Vector{String},
                                    news_tfidf::Dict{String, Dict{String,Float64}})::Dict{String,Float64}
    profile = Dict{String,Float64}()
    n = 0

    for nid in history
        haskey(news_tfidf, nid) || continue
        for (term, score) in news_tfidf[nid]
            profile[term] = get(profile, term, 0.0) + score
        end
        n += 1
    end

    if n > 1
        for t in keys(profile)
            profile[t] /= n
        end
    end

    return profile
end


"""Score candidate items by cosine-equivalent dot product with the user content profile."""
function score_candidates_cbf(user_id::String, candidates::Vector{String},
                               history::Vector{String},
                               news_tfidf::Dict{String, Dict{String,Float64}})::Dict{String,Float64}
    scores  = Dict(nid => 0.0 for nid in candidates)
    profile = build_user_content_profile(history, news_tfidf)
    isempty(profile) && return scores

    for nid in candidates
        haskey(news_tfidf, nid) || continue
        sim = 0.0
        for (term, user_score) in profile
            if haskey(news_tfidf[nid], term)
                sim += user_score * news_tfidf[nid][term]
            end
        end
        scores[nid] = sim
    end

    return scores
end
