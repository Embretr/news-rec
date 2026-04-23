
using DataFrames
using Statistics
using Printf


function analyze_dataset(split_name::String, news::DataFrame, behaviors::DataFrame)
    println("\n=== Dataset analysis: $split_name ===")

    # ── News stats ──────────────────────────────────────────────────────────
    n_articles = nrow(news)
    n_categories    = length(unique(skipmissing(news.Category)))
    n_subcategories = length(unique(skipmissing(news.SubCategory)))
    println("\n  Articles")
    @printf("    Total unique articles : %d\n", n_articles)
    @printf("    Categories            : %d\n", n_categories)
    @printf("    Sub-categories        : %d\n", n_subcategories)

    cat_counts = sort(
        combine(groupby(dropmissing(news, :Category), :Category), nrow => :count),
        :count, rev=true
    )
    println("\n  Top-10 categories by article count:")
    for row in eachrow(first(cat_counts, 10))
        @printf("    %-25s  %d\n", row.Category, row.count)
    end

    # ── User / behavior stats ────────────────────────────────────────────────
    n_users       = length(unique(skipmissing(behaviors.UserID)))
    n_impressions = nrow(behaviors)
    println("\n  Users & interactions")
    @printf("    Unique users          : %d\n", n_users)
    @printf("    Impression sessions   : %d\n", n_impressions)

    history_lengths = [length(parse_history(r.History)) for r in eachrow(behaviors)]
    @printf("    History length  mean  : %.1f\n", mean(history_lengths))
    @printf("    History length  median: %.1f\n", median(history_lengths))
    @printf("    History length  max   : %d\n",   maximum(history_lengths))
    @printf("    Users with no history : %d (%.1f%%)\n",
        count(==(0), history_lengths),
        100 * count(==(0), history_lengths) / n_impressions)

    impression_sizes = Int[]
    total_clicks = 0
    total_shown  = 0
    for row in eachrow(behaviors)
        impr = parse_impressions(row.Impressions)
        push!(impression_sizes, length(impr))
        for (_, c) in impr
            total_shown  += 1
            total_clicks += c
        end
    end
    ctr = total_clicks / max(total_shown, 1)
    @printf("    Impression list mean  : %.1f articles per session\n", mean(impression_sizes))
    @printf("    Total articles shown  : %d\n", total_shown)
    @printf("    Total clicks          : %d\n", total_clicks)
    @printf("    Click-through rate    : %.4f (%.2f%%)\n", ctr, 100 * ctr)

    # ── Article popularity distribution ────────────────────────────────────
    pop = Dict{String,Int}()
    for row in eachrow(behaviors)
        for nid in parse_history(row.History)
            pop[nid] = get(pop, nid, 0) + 1
        end
        for (nid, c) in parse_impressions(row.Impressions)
            c == 1 && (pop[nid] = get(pop, nid, 0) + 1)
        end
    end
    counts = collect(values(pop))
    n_long_tail = count(<=(2), counts)
    @printf("\n    Clicked articles              : %d\n", length(counts))
    @printf("    Long-tail (≤2 clicks)         : %d (%.1f%%)\n",
        n_long_tail, 100 * n_long_tail / max(length(counts), 1))
    @printf("    Top-1%% articles share of clicks: %.1f%%\n",
        begin
            sorted = sort(counts, rev=true)
            top1pct = max(1, div(length(sorted), 100))
            100 * sum(sorted[1:top1pct]) / sum(sorted)
        end)
end


function print_dataset_weaknesses()
    println("""

=== MIND dataset: known weaknesses ===

  1. Selection bias
     Impressions are not a random sample of all news — they were pre-filtered
     by MSN's existing recommendation algorithm. Models trained here learn a
     biased signal, not true organic user interest.

  2. Popularity bias
     A small fraction of articles accounts for the majority of clicks (power-law
     distribution). Naïve models over-recommend popular content; long-tail
     articles are chronically under-served.

  3. Temporal decay
     News articles have very short relevance windows (hours to days). The train
     split and dev split are from adjacent weeks, so items in the dev set can
     already be stale relative to training. This exacerbates cold-start for new
     articles.

  4. Cold-start (users)
     Many users appear only once or have very short histories, giving collaborative
     filtering almost no signal. The dataset does not include demographic or
     contextual features that could compensate.

  5. Implicit feedback only
     Clicks are a noisy proxy for genuine interest — users click
     accidentally, out of curiosity, or because a headline is misleading.
     Non-clicks are not necessarily negative (the article may not have been
     seen). No dwell time, rating, or explicit feedback is available.

  6. English-only, US-centric
     All content comes from MSN US. Models trained here do not generalise to
     multilingual or culturally different audiences.

  7. Entity-embedding sparsity
     The provided entity and relation embeddings cover only a subset of named
     entities in the corpus. Many articles have empty entity fields, limiting
     knowledge-graph–based approaches.
""")
end
