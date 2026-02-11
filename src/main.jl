using CSV
using DataFrames

root = joinpath(@__DIR__, "..")
mind_dir = joinpath(root, "MIND", "MINDsmall_dev")

print("Loading news.tsv...")
news = CSV.read(joinpath(mind_dir, "news.tsv"), DataFrame;
    delim='\t', header=false, missingstring="", strict=false)
println("Done!")

print("Loading behaviors.tsv...")
    behaviors = CSV.read(joinpath(mind_dir, "behaviors.tsv"), DataFrame;
    delim='\t', header=false, missingstring="", strict=false)
println("Done!")

function load_vec(path)
    emb = Dict{String, Vector{Float64}}()
    for line in eachline(path)
        parts = split(line)
        id = parts[1]
        vals = Float64.(parse.(Float64, parts[2:end]))
        emb[id] = vals
    end
    
    return emb
end

print("Loading entity vectors...")
entity_emb = load_vec(joinpath(mind_dir, "entity_embedding.vec"))
println("Done!")

print("Loading relation vectors...")
relation_emb = load_vec(joinpath(mind_dir, "relation_embedding.vec"))
println("Done!")
