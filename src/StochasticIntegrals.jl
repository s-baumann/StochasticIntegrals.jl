module StochasticIntegrals

using DataFrames
using Dates
using Distributions: Normal, quantile
using LinearAlgebra: Symmetric, cholesky, inv, det, LowerTriangular
using MultivariateFunctions
using Random
using Sobol: SobolSeq, next!

include("1_main_functions.jl")
export ItoIntegral, get_volatility, get_variance, get_covariance, get_correlation, ItoSet, CovarianceAtDate
export get_normal_draws, get_sobol_normal_draws, get_zero_draws, pdf, make_covariance, log_likelihood
include("2_data_conversions.jl")
export draws_to_array, draws_to_dataframe, array_to_draws, array_to_dataframe, dataframe_to_draws, dataframe_to_array

end
