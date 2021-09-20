using Documenter, StochasticIntegrals

makedocs(
    format = Documenter.HTML(),
    sitename = "StochasticIntegrals",
    modules = [StochasticIntegrals],
    pages = Any["Index" => "index.md",
                "API" => "api.md"]
)

deploydocs(
    repo   = "github.com/s-baumann/StochasticIntegrals.jl.git",
    target = "build",
    deps   = nothing,
    make   = nothing
)
