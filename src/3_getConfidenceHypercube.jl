function _one_iterate(cutoff_multiplier::Real, target::Real, draws::Array{T,2}, covar_matrix, tuning_parameter::Real) where T<:Real
    cutoffs = cutoff_multiplier .* sqrt.(diag(covar_matrix))
    number_of_draws = size(draws)[1]
    in_confidence_area = 0
    for i in 1:number_of_draws
        in_confidence_area += all(abs.(draws[i,:]) .< cutoffs)
    end
    mass_in_area = in_confidence_area/number_of_draws
    confidence_gap = target - mass_in_area
    return cutoff_multiplier + confidence_gap * tuning_parameter
end

function _randoms(chol, num::Integer, Seed::Integer)
    twist = MersenneTwister(Seed)
    dims = size(chol)[1]
    float_type = typeof(chol[1,1])
    array = Array{float_type,2}(undef, num, dims)
    for i in 1:num
        sobs = rand(twist, dims)
        normal_draw = quantile.(Ref(Normal()), sobs)
        scaled_draw = chol * normal_draw
        array[i,:] = scaled_draw
    end
    return array
end

"""
    get_confidence_hypercube(covar::ForwardCovariance, confidence_level::Real, data::Array{T,2}; tuning_parameter::Real = 1.0)
This returns the endpoints of a hypercube that contains confidence_level (%) of the dataset.
"""
function get_confidence_hypercube(covar::ForwardCovariance, confidence_level::Real, data::Array{T,2}; tuning_parameter::Real = 1.0, ConvergenceMetricThreshold::Real = 1e-10) where T<:Real
    # Using a univariate guess as we can get these pretty cheaply.
    guess = quantile(Normal(), 0.5*(1+confidence_level))
    # This runs once so that any error is explictly thrown and is traceable rather than being obscured by FixedPoint's try-catch
    #_ = _one_iterate.(guess, Ref(confidence_level), Ref(data), Ref(covar.covariance_), Ref(tuning_parameter))
    FP = fixed_point(x -> _one_iterate.(x, Ref(confidence_level), Ref(data), Ref(covar.covariance_), Ref(tuning_parameter)), [guess]; ConvergenceMetricThreshold = ConvergenceMetricThreshold, MaxIter = 10000)
    if ismissing(FP.FixedPoint_)
        error("Could not converge to a solution for the confidence hypercube. Try increasing the number of samples or adjusting the tuning parameter.")
    end
    cutoff_multiplier = FP.FixedPoint_[1]
    cutoffs = vcat(zip(-cutoff_multiplier .* sqrt.(diag(covar.covariance_)) , cutoff_multiplier .* sqrt.(diag(covar.covariance_)))...)
    return Dict{Symbol,Tuple{T,T}}(covar.covariance_labels_ .=> cutoffs)
end

function get_confidence_hypercube(covar::ForwardCovariance, confidence_level::Real, num::Integer; tuning_parameter::Real = 1.0,  ConvergenceMetricThreshold::Real = 1e-10)
    dims = length(covar.covariance_labels_)
    data = _randoms(covar.chol_, num, 1)
    return get_confidence_hypercube(covar, confidence_level, data; tuning_parameter = tuning_parameter, ConvergenceMetricThreshold = ConvergenceMetricThreshold)
end
