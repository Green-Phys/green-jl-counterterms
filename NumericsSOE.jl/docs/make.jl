using NumericsSOE
using Documenter

DocMeta.setdocmeta!(NumericsSOE, :DocTestSetup, :(using NumericsSOE); recursive=true)

makedocs(;
    modules=[NumericsSOE],
    authors="Yang Yu <yangyu.code@gmail.com> and contributors",
    sitename="NumericsSOE.jl",
    format=Documenter.HTML(;
        canonical="https://yangyu-code.github.io/NumericsSOE.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/QMBToolkit/NumericsSOE.jl",
    devbranch="main",
)