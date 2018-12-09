module StochasticIntegrals

using LinearAlgebra: Symmetric, cholesky, inv, det, LowerTriangular
using Dates
using MultivariateFunctions#: UnivariateFunction, PE_Function, evaluate_integral, years_between, years_from_global_base, evaluate
using Distributions: Normal, quantile
using Sobol: SobolSeq, next!
using Random

include("Brown.jl")

export ito_integral, flat_ito, get_volatility, get_variance, get_covariance, get_correlation, ito_set, covariance_at_date
export get_normal_draws, get_sobol_normal_draws, get_zero_draws, pdf, make_covariance, log_likelihood
end
