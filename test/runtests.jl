using StochasticIntegrals
using Test

# Run tests

println("Test Stochastic Integral Generation")
@time @test include("new_tests.jl")
@time @test include("conditioning.jl")
@time @test include("test_ito_processes.jl")
