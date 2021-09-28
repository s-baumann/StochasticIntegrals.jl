using UnivariateFunctions
using StochasticIntegrals
using DataFrames, Dates
using LinearAlgebra
using Distributions: Normal, MersenneTwister, quantile
using Statistics: var, mean, cov
using Sobol
using Distributions
using Random
const tol = 10*eps()

USD_vol = 0.0001
GBP_vol = 0.00001
GBP_FX_Vol = 0.002

today = Date(2016,12,1)
tommorow = Date(2016,12,2)
date_2020 = Date(2020,12,1)
later_date = Date(2035,12,1)
later_later_date = Date(2080,12,1)

USD_hw_a_curve = PE_Function(USD_vol)
GBP_hw_a_curve = PE_Function(GBP_vol)

brownian_corr_matrix = Hermitian([1.0 0.75 0.5 0.0;
                                  0.0 1.0 0.5 0.25;
                                  0.0 0.0 1.0 0.25;
                                  0.0 0.0 0.0 1.0])
brownian_ids = [:USD_IR, :GBP_IR, :GBP_FX, :BARC]
USD_IR_a_ito  = ItoIntegral(:USD_IR, USD_hw_a_curve)
GBP_IR_a_ito  = ItoIntegral(:GBP_IR, GBP_hw_a_curve)
GBP_FX_ito    = ItoIntegral(:GBP_FX, GBP_FX_Vol)
ito_integrals = Dict([:USD_IR_a, :GBP_IR_a, :GBP_FX] .=> [USD_IR_a_ito, GBP_IR_a_ito, GBP_FX_ito])

ito_set_ = ItoSet(brownian_corr_matrix, brownian_ids, ito_integrals)

for_covar = ForwardCovariance(ito_set_, years_from_global_base(today),
                              years_from_global_base(date_2020))
simp_covar = SimpleCovariance(ito_set_, years_from_global_base(today),
                              years_from_global_base(date_2020))

# Making ItoProcesses


ito_processes = Dict{Symbol,ItoProcess{Float64}}()
update_rates = Dict{Symbol,Exponential}()
for k in keys(ito_integrals)
    ito_processes[k] = ItoProcess( 0.0, 100.0, PE_Function(0.0), ito_integrals[k] )
    update_rates[k] = Exponential(5.0)
end

# Making the series.

for_series_sync = make_ito_process_syncronous_time_series(deepcopy(ito_processes), deepcopy(for_covar), 5.0, 1000)
abs(nrow(for_series_sync) - 3000) < 0.5
simp_series_sync = make_ito_process_syncronous_time_series(deepcopy(ito_processes), deepcopy(simp_covar), 5.0, 1000)
abs(nrow(simp_series_sync) - 3000) < 0.5

for_series_nonsync  = make_ito_process_non_syncronous_time_series(deepcopy(ito_processes), deepcopy(for_covar), update_rates , 1000)
abs(nrow(for_series_nonsync) - 1000) < 0.5
simp_series_nonsync = make_ito_process_non_syncronous_time_series(deepcopy(ito_processes), deepcopy(simp_covar), update_rates , 1000)
abs(nrow(simp_series_nonsync) - 1000) < 0.5
