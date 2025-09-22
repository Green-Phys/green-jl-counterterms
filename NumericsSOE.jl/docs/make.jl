using NumericsESPRIT
using Documenter

DocMeta.setdocmeta!(NumericsESPRIT, :DocTestSetup, :(using NumericsESPRIT); recursive=true)

makedocs(;
    modules=[NumericsESPRIT],
    authors="Yang Yu <yangyu.code@gmail.com> and contributors",
    sitename="NumericsESPRIT.jl",
    format=Documenter.HTML(;
        canonical="https://yangyu-code.github.io/NumericsESPRIT.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/yangyu-code/NumericsESPRIT.jl",
    devbranch="main",
)
