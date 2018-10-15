using UnivariateFunctions: evaluate, years_between, years_from_global_base, PE_Function, Sum_Of_Functions, derivative, indefinite_integral, evaluate_integral
using StochasticIntegrals: ito_integral, flat_ito, get_volatility, get_variance, get_covariance, get_correlation, ito_set, covariance_at_date, get_normal_draws, get_sobol_normal_draws, get_zero_draws, pdf, log_likelihood
using Dates
using LinearAlgebra
using Distributions: Normal, MersenneTwister
using Statistics: var, mean, cov
using Sobol
using Random
const tol = 10*eps()

USD_mean_reversion  = 0.15
USD_vol = 0.01
GBP_mean_reversion = 0.1
GBP_vol = 0.001
GBP_FX_Vol = 0.2

today = Date(2016,12,1)
date_2020 = Date(2020,12,1)
later_date = Date(2035,12,1)
later_later_date = Date(2080,12,1)

USD_hw_a_curve = PE_Function(USD_vol, USD_mean_reversion, today,0)
USD_hw_aB_curve = USD_hw_a_curve * (1/USD_mean_reversion) * (1 - PE_Function(1.0, -USD_mean_reversion, today,0))
GBP_hw_a_curve = PE_Function(GBP_vol, GBP_mean_reversion, today,0)
GBP_hw_aB_curve = GBP_hw_a_curve * (1/GBP_mean_reversion) * (1 - PE_Function(1.0, -GBP_mean_reversion, today,0))

brownian_corr_matrix = Symmetric([1.0 0.75 0.5 0.0;
                                  0.0 1.0 0.5 0.25;
                                  0.0 0.0 1.0 0.25;
                                  0.0 0.0 0.0 1.0])
brownian_ids = ["USD_IR", "GBP_IR", "GBP_FX", "BARC"]
USD_IR_a_ito  = ito_integral("USD_IR", "USD_IR_a", USD_hw_a_curve)
USD_IR_aB_ito = ito_integral("USD_IR", "USD_IR_aB", USD_hw_aB_curve)
GBP_IR_a_ito  = ito_integral("GBP_IR", "GBP_IR_a", GBP_hw_a_curve)
GBP_IR_aB_ito = ito_integral("GBP_IR", "GBP_IR_aB", GBP_hw_aB_curve)
GBP_FX_ito    = flat_ito(    "GBP_FX", "GBP_FX"   , GBP_FX_Vol)
ito_integrals = [USD_IR_a_ito, USD_IR_aB_ito, GBP_IR_a_ito, GBP_IR_aB_ito, GBP_FX_ito]

ito_set_ = ito_set(brownian_corr_matrix, brownian_ids, ito_integrals)
# The next ito integral should have constant vol
abs(get_volatility(ito_set_,  "GBP_FX", Date(2020,1,1)) - GBP_FX_Vol) < tol
abs(get_volatility(ito_set_,  "GBP_FX", Date(2022,1,1)) - GBP_FX_Vol) < tol
# The next ito integral has changing vol
abs(get_volatility(ito_set_,  "USD_IR_a", Date(2020,1,1)) - get_volatility(ito_set_,  "USD_IR_a", Date(2022,1,1)) ) > 0.1
abs(get_volatility(ito_set_,  "USD_IR_a", Date(2020,1,1)) - evaluate(USD_hw_a_curve, Date(2020,1,1)) ) < tol



covar = covariance_at_date(ito_set_, years_from_global_base(today), years_from_global_base(date_2020))
abs(covar.covariance_[5,5] - GBP_FX_Vol * (years_from_global_base(date_2020) - years_from_global_base(today))) < tol
abs(covar.covariance_[1,1] - get_variance(USD_IR_a_ito, years_from_global_base(today), years_from_global_base(date_2020))) < tol

cov_date = covariance_at_date(ito_set_, today, later_date)
abs(cov_date.covariance_[5,5] - GBP_FX_Vol * (years_from_global_base(later_date) - years_from_global_base(today))) < tol

abs(get_volatility(ito_set_,  "USD_IR_a", Date(2020,1,1)) - get_volatility(cov_date,  "USD_IR_a", Date(2020,1,1)) ) < tol
abs(cov_date.covariance_[1,3] - get_covariance(cov_date, "USD_IR_a", "GBP_IR_a")) < tol
abs(get_variance(cov_date, "USD_IR_a") - get_covariance(cov_date, "USD_IR_a", "USD_IR_a")) < tol

# Test correlation.
abs(get_correlation(cov_date, "USD_IR_a", "GBP_IR_a") - (get_covariance(cov_date, "USD_IR_a", "GBP_IR_a")/sqrt(get_variance(cov_date, "USD_IR_a") * get_variance(cov_date, "GBP_IR_a")))) < tol
abs(get_correlation(cov_date, "GBP_IR_aB", "GBP_IR_a") - (get_covariance(cov_date, "GBP_IR_aB", "GBP_IR_a")/sqrt(get_variance(cov_date, "GBP_IR_aB") * get_variance(cov_date, "GBP_IR_a")))) < tol

Random.seed!(1234)
## Test random draws
function abs_value_of_dict_differences(dictA::Dict, dictB::Dict)
   differences = merge(-, dictA, dictB)
   return sum(abs.(values(differences)))
end
abs_value_of_dict_differences(get_normal_draws(cov_date), get_normal_draws(cov_date)) > tol
abs_value_of_dict_differences(get_normal_draws(cov_date;  uniform_draw = rand(5)), get_normal_draws(cov_date; uniform_draw =  rand(5))) > tol

function test_random_points_pdf(covar::covariance_at_date)
    draws = get_normal_draws(covar)
    pdf_val = pdf(covar, draws)
    return (pdf_val >= 0.0)
end
all([test_random_points_pdf(cov_date) for i in 1:1000])

# Distribution Testing
function SplitDicts(dictarray::Array{Dict{String,Float64}})
    return get.(dictarray, "USD_IR_a", 0), get.(dictarray, "USD_IR_aB", 0), get.(dictarray, "GBP_IR_a", 0), get.(dictarray, "GBP_IR_aB", 0), get.(dictarray, "GBP_FX", 0)
end
s = SobolSeq(length(cov_date.ito_set_.ito_integrals_))
normals = get_normal_draws(cov_date,100000)
normal_samples = SplitDicts(normals)

sobols = get_sobol_normal_draws(cov_date, s,100000)
sobol_samples = SplitDicts(sobols)
zero_draws = get_zero_draws(cov_date,2)

abs(var(normal_samples[1]) - get_variance(cov_date, "USD_IR_a"))  < 0.001
abs(var(normal_samples[2]) - get_variance(cov_date, "USD_IR_aB")) < 0.0011
abs(var(normal_samples[3]) - get_variance(cov_date, "GBP_IR_a"))  < 1e-06
abs(var(normal_samples[4]) - get_variance(cov_date, "GBP_IR_aB")) < 1e-04
abs(var(normal_samples[5]) - get_variance(cov_date, "GBP_FX"))    < 0.02

abs(var(sobol_samples[1])  - get_variance(cov_date, "USD_IR_a"))  < 1e-04
abs(var(sobol_samples[2])  - get_variance(cov_date, "USD_IR_aB")) < 0.001
abs(var(sobol_samples[3])  - get_variance(cov_date, "GBP_IR_a"))  < 1e-07
abs(var(sobol_samples[4])  - get_variance(cov_date, "GBP_IR_aB")) < 1e-05
abs(var(sobol_samples[5])  - get_variance(cov_date, "GBP_FX"))    < 0.02

cov(sobol_samples[5], sobol_samples[1])
get_covariance(cov_date, "GBP_FX", "USD_IR_a")


#  Test likelihood
function test_random_points_loglikelihood(covar::covariance_at_date)
    draws = get_normal_draws(covar)
    log_likelihood_val = log_likelihood(covar, draws)
    return log_likelihood_val > -Inf
end
all([test_random_points_loglikelihood(cov_date) for i in 1:1000])