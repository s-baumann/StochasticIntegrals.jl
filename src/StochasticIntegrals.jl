module StochasticIntegrals

using DataFrames
using DataStructures: OrderedDict
using Dates
using Distributions
using LinearAlgebra: Hermitian, cholesky, inv, det, LowerTriangular, diag
using UnivariateFunctions
using Random
using Sobol: SobolSeq, next!
using FixedPointAcceleration

include("4_number_generators.jl")
export NumberGenerator, Mersenne, SobolGen, next!
include("1_main_functions.jl")
export ItoIntegral, volatility, variance, covariance, correlation, ItoSet, ForwardCovariance
export get_draws, get_zero_draws, pdf, make_covariance, log_likelihood, brownians_in_use
include("2_data_conversions.jl")
export to_draws, to_dataframe, to_array
include("3_getConfidenceHypercube.jl")
export get_confidence_hypercube
include("5_conditional_distributions.jl")
export generate_conditioned_distribution
include("6_ItoProcesses.jl")
export ItoProcess, evolve, evolve_covar_and_ito_processes, make_ito_process_syncronous_time_series, make_ito_process_non_syncronous_time_series

end
