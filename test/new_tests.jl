using Test

@testset "Main Tests" begin
    using UnivariateFunctions
    using StochasticIntegrals
    using Dates
    using LinearAlgebra
    using Distributions: Normal, MersenneTwister, quantile
    using Statistics: var, mean, cov
    using Sobol
    using Random
    tol = 1000*eps()

    USD_mean_reversion  = 0.15
    USD_vol = 0.01
    GBP_mean_reversion = 0.1
    GBP_vol = 0.001
    GBP_FX_Vol = 0.2

    today = Date(2016,12,1)
    tommorow = Date(2016,12,2)
    date_2020 = Date(2020,12,1)
    later_date = Date(2035,12,1)
    later_later_date = Date(2080,12,1)

    USD_hw_a_curve = PE_Function(USD_vol, USD_mean_reversion, today,0)
    USD_hw_aB_curve = USD_hw_a_curve * (1/USD_mean_reversion) * (1 - PE_Function(1.0, -USD_mean_reversion, today,0))
    GBP_hw_a_curve = PE_Function(GBP_vol, GBP_mean_reversion, today,0)
    GBP_hw_aB_curve = GBP_hw_a_curve * (1/GBP_mean_reversion) * (1 - PE_Function(1.0, -GBP_mean_reversion, today,0))

    brownian_corr_matrix = Hermitian([1.0 0.75 0.5 0.0;
                                      0.0 1.0 0.5 0.25;
                                      0.0 0.0 1.0 0.25;
                                      0.0 0.0 0.0 1.0])
    brownian_ids = [:USD_IR, :GBP_IR, :GBP_FX, :BARC]
    USD_IR_a_ito  = ItoIntegral(:USD_IR, USD_hw_a_curve)
    USD_IR_aB_ito = ItoIntegral(:USD_IR, USD_hw_aB_curve)
    GBP_IR_a_ito  = ItoIntegral(:GBP_IR, GBP_hw_a_curve)
    GBP_IR_aB_ito = ItoIntegral(:GBP_IR, GBP_hw_aB_curve)
    GBP_FX_ito    = ItoIntegral(:GBP_FX, GBP_FX_Vol)
    ito_integrals = Dict([:USD_IR_a, :USD_IR_aB, :GBP_IR_a, :GBP_IR_aB, :GBP_FX] .=> [USD_IR_a_ito, USD_IR_aB_ito, GBP_IR_a_ito, GBP_IR_aB_ito, GBP_FX_ito])

    ito_set_ = ItoSet(brownian_corr_matrix, brownian_ids, ito_integrals)
    # The next ito integral should have constant vol
    @test abs(volatility(ito_set_,  :GBP_FX, Date(2020,1,1)) - GBP_FX_Vol) < tol
    @test abs(volatility(ito_set_,  :GBP_FX, Date(2022,1,1)) - GBP_FX_Vol) < tol
    # The next ito integral has changing vol
    @test abs(volatility(ito_set_,  :USD_IR_a, Date(2020,1,1)) - volatility(ito_set_,  :USD_IR_a, Date(2022,1,1)) ) > 0.005
    @test abs(volatility(ito_set_,  :USD_IR_a, Date(2020,1,1)) - evaluate(USD_hw_a_curve, Date(2020,1,1)) ) < tol



    covar = ForwardCovariance(ito_set_, years_from_global_base_date(today), years_from_global_base_date(date_2020))
    @test abs(covar.covariance_[5,5] - GBP_FX_Vol^2 * (years_from_global_base_date(date_2020) - years_from_global_base_date(today))) < tol
    @test abs(covar.covariance_[1,1] - variance(USD_IR_a_ito, years_from_global_base_date(today), years_from_global_base_date(date_2020))) < tol
    @test variance(USD_IR_a_ito, today, today, today) < tol
    @test variance(USD_IR_a_ito, today, tommorow, today) < tol
    @test covariance(USD_IR_a_ito, USD_IR_a_ito, today, today, today, 1.0) < tol

    cov_date = ForwardCovariance(ito_set_, today, later_date)
    @test abs(cov_date.covariance_[5,5] - GBP_FX_Vol^2 * (years_from_global_base_date(later_date) - years_from_global_base_date(today))) < tol

    @test abs(volatility(ito_set_,  :USD_IR_a, Date(2020,1,1)) - volatility(cov_date,  :USD_IR_a, Date(2020,1,1)) ) < tol
    @test abs(cov_date.covariance_[1,3] - covariance(cov_date, :USD_IR_a, :GBP_IR_a)) < tol
    @test abs(variance(cov_date, :USD_IR_a) - covariance(cov_date, :USD_IR_a, :USD_IR_a)) < tol

    # Test correlation.
    @test abs(correlation(cov_date, :USD_IR_a, :GBP_IR_a) - (covariance(cov_date, :USD_IR_a, :GBP_IR_a)/sqrt(variance(cov_date, :USD_IR_a) * variance(cov_date, :GBP_IR_a)))) < tol
    @test abs(correlation(cov_date, :GBP_IR_aB, :GBP_IR_a) - (covariance(cov_date, :GBP_IR_aB, :GBP_IR_a)/sqrt(variance(cov_date, :GBP_IR_aB) * variance(cov_date, :GBP_IR_a)))) < tol

    Random.seed!(1234)
    ## Test random draws
    function abs_value_of_dict_differences(dictA::Dict, dictB::Dict)
       differences = merge(-, dictA, dictB)
       return sum(abs.(values(differences)))
    end
    @test abs_value_of_dict_differences(get_draws(cov_date), get_draws(cov_date)) > tol
    @test abs_value_of_dict_differences(get_draws(cov_date;  uniform_draw = rand(5)), get_draws(cov_date; uniform_draw =  rand(5))) > tol

    function test_random_points_pdf(covar::ForwardCovariance)
        draws = get_draws(covar)
        pdf_val = StochasticIntegrals.pdf(covar, draws)
        return (pdf_val >= 0.0)
    end
    @test all([test_random_points_pdf(cov_date) for i in 1:1000])

    # Distribution Testing
    function SplitDicts(dictarray)
        return get.(dictarray, :USD_IR_a, 0), get.(dictarray, :USD_IR_aB, 0), get.(dictarray, :GBP_IR_a, 0), get.(dictarray, :GBP_IR_aB, 0), get.(dictarray, :GBP_FX, 0)
    end
    defaults = get_draws(cov_date,100000)
    normal_twister = Mersenne(MersenneTwister(123), length(cov_date.covariance_labels_))
    normals = get_draws(cov_date,100000; number_generator = normal_twister)
    normal_samples = SplitDicts(normals)

    s = SobolGen(SobolSeq(length(cov_date.ito_set_.ito_integrals_)))
    sobols = get_draws(cov_date, 100000; number_generator = s)
    sobol_samples = SplitDicts(sobols)
    zero_draws = get_zero_draws(cov_date,2)

    @test abs(var(normal_samples[1]) - variance(cov_date, :USD_IR_a))  < 0.002
    @test abs(var(normal_samples[2]) - variance(cov_date, :USD_IR_aB)) < 0.12
    @test abs(var(normal_samples[3]) - variance(cov_date, :GBP_IR_a))  < 2e-06
    @test abs(var(normal_samples[4]) - variance(cov_date, :GBP_IR_aB)) < 2e-04
    @test abs(var(normal_samples[5]) - variance(cov_date, :GBP_FX))    < 0.03

    @test abs(var(sobol_samples[1])  - variance(cov_date, :USD_IR_a))  < 2e-04
    @test abs(var(sobol_samples[2])  - variance(cov_date, :USD_IR_aB)) < 0.002
    @test abs(var(sobol_samples[3])  - variance(cov_date, :GBP_IR_a))  < 2e-05
    @test abs(var(sobol_samples[4])  - variance(cov_date, :GBP_IR_aB)) < 2e-05
    @test abs(var(sobol_samples[5])  - variance(cov_date, :GBP_FX))    < 0.03

    @test abs(cov(sobol_samples[5], sobol_samples[1]) - covariance(cov_date, :GBP_FX, :USD_IR_a)) < 0.001


    #  Test likelihood
    function test_random_points_loglikelihood(covar::ForwardCovariance)
        draws = get_draws(covar)
        log_likelihood_val = log_likelihood(covar, draws)
        return log_likelihood_val > -Inf
    end
    all([test_random_points_loglikelihood(cov_date) for i in 1:1000])


    # Testing data conversions - From draws
    draws = get_draws(cov_date, 7)
    arr, labs = to_array(draws; labels = [:USD_IR_a, :USD_IR_aB, :GBP_IR_a, :GBP_IR_aB])
    @test size(arr) == (7,4)
    @test labs ==  [:USD_IR_a, :USD_IR_aB, :GBP_IR_a, :GBP_IR_aB]
    arr2, labs2 = to_array(draws)
    @test size(arr2) == (7,5)
    @test Set(labs2) == Set([:USD_IR_a, :USD_IR_aB, :GBP_IR_a, :GBP_IR_aB, :GBP_FX])
    dd = to_dataframe(draws; labels = [:USD_IR_a, :USD_IR_aB, :GBP_IR_a, :GBP_IR_aB])
    @test size(dd) == (7,4)
    dd2 = to_dataframe(draws)
    @test size(dd2) == (7,5)
    @test Set(Symbol.(names(dd2))) == Set([:USD_IR_a, :USD_IR_aB, :GBP_IR_a, :GBP_IR_aB, :GBP_FX])

    # Testing data conversions - From array
    draws = to_draws(arr; labels = labs)
    @test length(draws) == 7
    @test Set(collect(keys(draws[1]))) == Set([:USD_IR_a, :USD_IR_aB, :GBP_IR_a, :GBP_IR_aB])
    draws = to_draws(arr)
    @test length(draws) == 7
    @test Set(collect(keys(draws[1]))) == Set([:x1, :x2, :x3, :x4])
    dd = to_dataframe(arr; labels = labs)
    @test size(dd)[1] == 7
    @test Set(Symbol.(collect(names(dd)))) == Set([:USD_IR_a, :USD_IR_aB, :GBP_IR_a, :GBP_IR_aB])
    dd2 = to_dataframe(arr)
    @test size(dd2)[1] == 7
    @test Set(Symbol.(names(dd2)))== Set([:x1, :x2, :x3, :x4])

    # Testing data conversions - From dataframe
    draws = to_draws(dd; labels = labs)
    @test length(draws) == 7
    @test Set(collect(keys(draws[1]))) == Set([:USD_IR_a, :USD_IR_aB, :GBP_IR_a, :GBP_IR_aB])
    draws = to_draws(dd)
    @test length(draws) == 7
    @test Set(collect(keys(draws[1]))) == Set([:USD_IR_a, :USD_IR_aB, :GBP_IR_a, :GBP_IR_aB])
    X, labs = to_array(dd; labels = labs)
    @test size(X) == (7, 4)
    @test Set(labs) == Set([:USD_IR_a, :USD_IR_aB, :GBP_IR_a, :GBP_IR_aB])
    X, labs = to_array(dd)
    @test size(X) == (7, 4)
    @test Set(Symbol.(labs)) == Set([:USD_IR_a, :USD_IR_aB, :GBP_IR_a, :GBP_IR_aB])

    # # Testing hypercube generation
    # confidence_level = 0.95
    # num = 1000000
    # confidence_hc = get_confidence_hypercube(covar, confidence_level, num)
    # # Now each edge should be same number of standard deviations away from the mean:
    # devs = confidence_hc[:USD_IR_a][2]/sqrt(covar.covariance_[1,1])
    # @test abs(devs - confidence_hc[:GBP_IR_aB][2]/sqrt(covar.covariance_[2,2])) < tol
    # @test abs(devs - confidence_hc[:GBP_IR_a][2]/sqrt(covar.covariance_[3,3]))  < tol
    # @test abs(devs - confidence_hc[:USD_IR_aB][2]/sqrt(covar.covariance_[4,4])) < tol
    # @test abs(devs - confidence_hc[:GBP_FX][2]/sqrt(covar.covariance_[5,5]))    < tol
    # # And the confidence hypercube should contain confidence_level of the distribution.
    # function _sobols(chol, num::Int, sob_seq::SobolGen)
    #     dims = size(chol)[1]
    #     array = Array{Float64,2}(undef, num, dims)
    #     for i in 1:num
    #         sobs = next!(sob_seq)
    #         normal_draw = quantile.(Ref(Normal()), sobs)
    #         scaled_draw = chol * normal_draw
    #         array[i,:] = scaled_draw
    #     end
    #     return array
    # end
    # function estimate_mass_in_hypercube()
    #     dims = length(covar.covariance_labels_)
    #     data = _sobols(covar.chol_, num, SobolGen(SobolSeq(dims)))
    #     number_of_draws = size(data)[1]
    #     in_confidence_area = 0
    #     cutoffs = devs .* sqrt.(diag(covar.covariance_))
    #     for i in 1:number_of_draws
    #         in_confidence_area += all(abs.(data[i,:]) .< cutoffs)
    #     end
    #     mass_in_area = in_confidence_area/number_of_draws
    #     return mass_in_area
    # end
    # @test abs(estimate_mass_in_hypercube() - confidence_level) < 0.01

    # # Testing hypercube generation
    # confidence_level = 0.5
    # num = 1000
    # confidence_hc = get_confidence_hypercube(covar, confidence_level, 500)
    # # Now each edge should be same number of standard deviations away from the mean:
    # devs = confidence_hc[:USD_IR_a][2]/sqrt(covar.covariance_[1,1])
    # @test abs(devs - confidence_hc[:GBP_IR_aB][2]/sqrt(covar.covariance_[2,2])) < tol
    # @test abs(devs - confidence_hc[:GBP_IR_a][2]/sqrt(covar.covariance_[3,3]))  < tol
    # @test abs(devs - confidence_hc[:USD_IR_aB][2]/sqrt(covar.covariance_[4,4])) < tol
    # @test abs(devs - confidence_hc[:GBP_FX][2]/sqrt(covar.covariance_[5,5]))    < tol
    # # And the confidence hypercube should contain confidence_level of the distribution.
    # @test abs(estimate_mass_in_hypercube() - confidence_level) < 0.03




    # Testing Antithetic Variates
    normals_antithetic = get_draws(cov_date,100000; number_generator = normal_twister, antithetic_variates = true)
    normal_samples_antithetic = SplitDicts(normals_antithetic)
    ## Antithetic should have lower variance due to some data copying.
    @test var(normal_samples[1]) > var(normal_samples_antithetic[1]) * 0.9
    ## Antithetic each pair should sum to 0. So whole vector should sum to zero.
    @test sum(normal_samples_antithetic[1]) < tol
    @test sum(normal_samples_antithetic[2]) < 100*tol
    @test sum(normal_samples_antithetic[3]) < tol
    @test sum(normal_samples_antithetic[4]) < tol
    @test sum(normal_samples_antithetic[5]) < tol

    # Now we will do a test for finding the expectation of a lognormal.
    theoretical_expectation = exp(variance(cov_date, :USD_IR_a)/2)
    log_normal_samples = exp.(normal_samples[1])
    log_normal_expectation_estimate = mean(log_normal_samples)
    log_normal_antithetic_samples = exp.(normal_samples_antithetic[1])
    log_normal_antithetic_expectation_estimate = mean(log_normal_antithetic_samples)
    @test abs(log_normal_antithetic_expectation_estimate - theoretical_expectation) < abs(log_normal_expectation_estimate - theoretical_expectation)
end
