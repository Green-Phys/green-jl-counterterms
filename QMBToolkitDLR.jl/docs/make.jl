using QMBToolkitDLR
using Documenter

DocMeta.setdocmeta!(QMBToolkitDLR, :DocTestSetup, :(using QMBToolkitDLR); recursive=true)

makedocs(;
    modules=[QMBToolkitDLR],
    authors="Yang Yu <yangyu.code@gmail.com> and contributors",
    sitename="QMBToolkitDLR.jl",
    format=Documenter.HTML(;
        canonical="https://QMBToolkit.github.io/QMBToolkitDLR.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/QMBToolkit/QMBToolkitDLR.jl",
    devbranch="main",
)
