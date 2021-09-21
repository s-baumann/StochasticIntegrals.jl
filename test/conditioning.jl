using UnivariateFunctions
using StochasticIntegrals
using DataFrames
using Dates
using LinearAlgebra
using Distributions: Normal, MersenneTwister, quantile, Exponential
using Statistics: var, mean, cov
using Sobol
using Random
const tol = 10*eps()
twister = MersenneTwister(22)

USD_mean_reversion  = 0.15
USD_vol = 0.01
GBP_mean_reversion = 0.1
GBP_vol = 0.001
GBP_FX_Vol = 0.2

today    = Date(2016,12,1)
tommorow = Date(2016,12,2)

Barclays_vol = PE_Function(0.1, 0.0, 0.0, 0)
CS_vol       = PE_Function(0.2, 0.0, 0.0, 0)

brownian_corr_matrix = Hermitian([1.0 0.75;
                                  0.75 1.0])
assets = [ :CS, :BARC]
BARC_ito    = ItoIntegral(:BARC, Barclays_vol)
CS_ito    = ItoIntegral(:CS, CS_vol)
ito_integrals = Dict([:CS, :BARC] .=> [CS_ito, BARC_ito])

ito_set_ = ItoSet(brownian_corr_matrix, assets, ito_integrals)
covar = ForwardCovariance(ito_set_, years_from_global_base(today), years_from_global_base(tommorow))

T = Float64
conditioning_draws = Dict{Symbol,T}(:CS => 0.02)
conditional_mean, conditional_covar = generate_conditioned_distribution(covar, conditioning_draws)
conditional_mean[1] > 100*eps()

# Testing ito processes
stock_processes = Dict(assets .=> map(a -> ItoProcess(0.0, 0.0, PE_Function(0.00, 0.0, 0.0, 0), ito_integrals[a]), assets))
update_rates = Dict(assets .=> Exponential.(    rand(twister, length(assets))))

ts_syncronous  = make_ito_process_syncronous_time_series(stock_processes, covar,
mean(a-> update_rates[a].θ, assets),1000; ito_twister = twister)
nrow(ts_syncronous) == 2000
ts_asyncronous = make_ito_process_non_syncronous_time_series(stock_processes, covar, update_rates, 1000; timing_twister = twister, ito_twister = twister)
nrow(ts_asyncronous) == 1000
