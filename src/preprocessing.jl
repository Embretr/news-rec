
using CSV
using DataFrames

# Data loading utilities for MIND news recommendation splits.

const STOPWORDS = Set([
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "by","from","up","about","into","through","during","is","are","was","were",
    "be","been","being","have","has","had","do","does","did","will","would",
    "could","should","may","might","shall","can","need","dare","ought","used",
    "it","its","this","that","these","those","i","me","my","we","our","you",
    "your","he","him","his","she","her","they","them","their","what","which",
    "who","whom","not","no","nor","so","yet","both","either","each","few",
    "more","most","other","some","such","than","too","very","just","as","also",
])

function tokenize(text::Union{AbstractString,Missing})::Vector{String}
    ismissing(text) && return String[]
    cleaned = replace(lowercase(text), r"[^a-z0-9\s]" => " ")
    return filter(t -> !isempty(t) && t ∉ STOPWORDS, String.(split(cleaned)))
end


"""Load and sanitize `news.tsv` into a typed DataFrame with named columns."""
function load_news(path::String)::DataFrame
    news = CSV.read(path, DataFrame;
        delim='\t', header=false, missingstring="", strict=false)
    rename!(news, [:NewsID, :Category, :SubCategory, :Title,
                   :Abstract, :URL, :TitleEntities, :AbstractEntities])
    sanitize!(news)
    return news
end

"""Load and sanitize `behaviors.tsv` into a typed DataFrame with named columns."""
function load_behaviors(path::String)::DataFrame
    behaviors = CSV.read(path, DataFrame;
        delim='\t', header=false, missingstring="", strict=false)
    rename!(behaviors, [:ImpressionID, :UserID, :Time, :History, :Impressions])
    sanitize!(behaviors)
    return behaviors
end


"""Normalize string-like columns to `Union{String,Missing}` for downstream parsing."""
function sanitize!(df::DataFrame)::DataFrame
    for col in propertynames(df)
        v = df[!, col]
        eltype(v) <: Union{AbstractString,Missing} || continue
        new_v = Vector{Union{String,Missing}}(missing, length(v))
        for i in eachindex(v)
            if isassigned(v, i)
                val = v[i]
                ismissing(val) || (new_v[i] = String(val))
            end
        end
        df[!, col] = new_v
    end
    return df
end


"""Parse browsing history field into a vector of news IDs."""
function parse_history(s::Union{AbstractString, Missing})::Vector{String}
    ismissing(s) && return String[]
    return String.(split(s))
end

"""Parse impression field into `(news_id, clicked)` tuples."""
function parse_impressions(s::Union{AbstractString, Missing})::Vector{Tuple{String,Int}}
    ismissing(s) && return Tuple{String,Int}[]
    result = Tuple{String,Int}[]
    for item in split(s)
        parts = split(item, '-')
        length(parts) == 2 || continue
        push!(result, (String(parts[1]), parse(Int, parts[2])))
    end
    return result
end


"""
Build interaction structures used by collaborative filtering:
- `user_items`: items consumed per user
- `item_users`: users associated with each item
"""
function build_user_item_data(behaviors::DataFrame)
    user_items = Dict{String, Set{String}}()
    item_users = Dict{String, Vector{String}}()

    for row in eachrow(behaviors)
        uid = row.UserID
        if !haskey(user_items, uid)
            user_items[uid] = Set{String}()
        end

        for nid in parse_history(row.History)
            push!(user_items[uid], nid)
        end

        for (nid, clicked) in parse_impressions(row.Impressions)
            if clicked == 1
                push!(user_items[uid], nid)
            end
        end
    end

    for (uid, items) in user_items
        for nid in items
            if !haskey(item_users, nid)
                item_users[nid] = String[]
            end
            push!(item_users[nid], uid)
        end
    end

    return user_items, item_users
end

"""Load both `news.tsv` and `behaviors.tsv` for a given dataset split directory."""
function load_mind(split_dir::String)
    news      = load_news(joinpath(split_dir, "news.tsv"))
    behaviors = load_behaviors(joinpath(split_dir, "behaviors.tsv"))
    return news, behaviors
end


struct MINDData
    news::DataFrame
    behaviors::DataFrame
    user_items::Dict{String,Set{String}}
    item_users::Dict{String,Vector{String}}
    all_news_ids::Set{String}
end

# Full preprocessing pipeline: load → clean → index user/item interactions.
function preprocess_data(split_dir::String)::MINDData
    news      = load_news(joinpath(split_dir, "news.tsv"))
    behaviors = load_behaviors(joinpath(split_dir, "behaviors.tsv"))

    # deduplicate news (same article can appear in multiple splits)
    unique!(news, :NewsID)

    user_items, item_users = build_user_item_data(behaviors)
    all_news_ids = Set(news.NewsID)

    return MINDData(news, behaviors, user_items, item_users, all_news_ids)
end
