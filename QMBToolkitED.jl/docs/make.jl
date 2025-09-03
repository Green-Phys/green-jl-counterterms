using QMBToolkitED
using Documenter

DocMeta.setdocmeta!(QMBToolkitED, :DocTestSetup, :(using QMBToolkitED); recursive=true)

makedocs(;
    modules=[QMBToolkitED],
    authors="Yang Yu <yangyu.code@gmail.com> and contributors",
    sitename="QMBToolkitED.jl",
    format=Documenter.HTML(;
        canonical="https://QMBToolkit.github.io/QMBToolkitED.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/QMBToolkit/QMBToolkitED.jl",
    devbranch="main",
)
