using QMBToolkitSOEXCA
using Documenter

DocMeta.setdocmeta!(QMBToolkitSOEXCA, :DocTestSetup, :(using QMBToolkitSOEXCA); recursive=true)

makedocs(;
    modules=[QMBToolkitSOEXCA],
    authors="Yang Yu <yangyu.code@gmail.com> and contributors",
    sitename="QMBToolkitSOEXCA.jl",
    format=Documenter.HTML(;
        canonical="https://QMBToolkit.github.io/QMBToolkitSOEXCA.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/QMBToolkit/QMBToolkitSOEXCA.jl",
    devbranch="main",
)
