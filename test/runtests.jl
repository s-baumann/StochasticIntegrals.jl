using StochasticIntegrals
using Test

# Run tests

println("Test Stochastic Integral Generation")
include("new_tests.jl")
include("conditioning.jl")
include("test_ito_processes.jl")
include("regression_test.jl")
