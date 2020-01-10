using UnivariateFunctions
using StochasticIntegrals
using DataFrames
using Dates
using LinearAlgebra
using Distributions: Exponential, MersenneTwister, quantile
using DataStructures: OrderedDict
using Statistics: std, var, mean, cov
using Sobol
using Random
const tol = 10*eps()

brownian_corr_matrix = Symmetric([1.0 0.75 0.5 0.0;
                                  0.0 1.0 0.5 0.25;
                                  0.0 0.0 1.0 0.25;
                                  0.0 0.0 0.0 1.0])
brownian_ids = [:BARC, :HSBC, :VODL, :RYAL]

BARC  = ItoIntegral(:BARC, PE_Function(0.2))
HSBC  = ItoIntegral(:HSBC, PE_Function(0.3))
VODL  = ItoIntegral(:VODL, PE_Function(0.4))
RYAL  = ItoIntegral(:RYAL, PE_Function(0.5))
ito_integrals = Dict([:BARC, :HSBC, :VODL, :RYAL] .=> [BARC, HSBC, VODL, RYAL])
ito_set_ = ItoSet(brownian_corr_matrix, brownian_ids, ito_integrals)

update_rates = OrderedDict([:BARC, :HSBC, :VODL, :RYAL] .=> [Exponential(2.0), Exponential(3.0), Exponential(5.0), Exponential(2.5)])
covar = ForwardCovariance(ito_set_, 0.0, 1.0)
stock_processes = Dict([:BARC, :HSBC, :VODL, :RYAL] .=>
                           [ItoProcess(0.0, 180.0, PE_Function(0.00), ito_integrals[:BARC]),
                           ItoProcess(0.0, 360.0, PE_Function(0.00), ito_integrals[:HSBC]),
                           ItoProcess(0.0, 720.0, PE_Function(-0.00), ito_integrals[:VODL]),
                           ItoProcess(0.0, 500.0, PE_Function(0.0), ito_integrals[:RYAL])])
ts = make_ito_process_non_syncronous_time_series(stock_processes, covar, update_rates, 100000;
                                                 timing_twister = MersenneTwister(2), ito_twister = MersenneTwister(5))



from_time = 0.0

next_tick(ts, 0.0)
next_tick(ts, 69965.41)

refresh_frequency = secs_between_refreshes(ts)

at_times = get_all_refresh_times(ts)
dd_compiled = latest_value(ts, at_times)

dd = get_returns(dd_compiled, 1.0; returns = :simple)

R = Float64
mat = naive_covariance(dd)
obs = nrow(dd)
regularised = eigenvalue_clean(mat; obs = obs)
blocks = Array{R,1}([1,2])
# block estimation
# In: ts, blocks
refresh_frequency = secs_between_refreshes(ts)
at_times = get_all_refresh_times(ts)
dd_compiled = latest_value(ts, at_times)
dd = get_returns(dd_compiled, 1.0; returns = :simple)





N = nrow(refresh_frequency)
for i in 1:N
