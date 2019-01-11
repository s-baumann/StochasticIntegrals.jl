module StochasticIntegrals

using LinearAlgebra: Symmetric, cholesky, inv, det, LowerTriangular
using Dates
using MultivariateFunctions
using Distributions: Normal, quantile
using Sobol: SobolSeq, next!
using Random

include("Brown.jl")

export ItoIntegral, get_volatility, get_variance, get_covariance, get_correlation, ItoSet, CovarianceAtDate
export get_normal_draws, get_sobol_normal_draws, get_zero_draws, pdf, make_covariance, log_likelihood
end
