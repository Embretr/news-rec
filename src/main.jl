using CSV, DataFrames

news_file = joinpath(pwd(), "MIND", "MINDsmall_dev", "news.tsv")
news_df = CSV.read(news_file, DataFrame; delim='\t', header=["NewsID", "Category", "SubCategory", "Title", "Abstract", "URL", "Entities", "TitleEntities"])
