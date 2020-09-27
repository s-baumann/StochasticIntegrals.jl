const tol = 10*eps()

"""
ItoIntegral
A struct detailing an ito integral. It contains a UnivariateFunction detailing the integrand as well as a symbol detailing an id of the integral's processes.

Usual (and most general) contructor is:
    ItoIntegral(brownian_id_::Symbol, f_::UnivariateFunction)
Convenience constructor for ItoIntegrals where the integrand is a flat function is:
    ItoIntegral(brownian_id_::Symbol, variance_::Real)
"""
struct ItoIntegral
    brownian_id_::Symbol
    f_::UnivariateFunction
    function ItoIntegral(brownian_id_::Symbol, variance_::Real)
        return new(brownian_id_, PE_Function(variance_, 0.0, 0.0, 0))
    end
    function ItoIntegral(brownian_id_::Symbol, f_::UnivariateFunction)
        return new(brownian_id_, f_)
    end
end

"""
    variance(ito::ItoIntegral, from::Real, to::Real)
    variance(ito::ItoIntegral, base::Union{Date,DateTime}, from::Union{Date,DateTime}, to::Union{Date,DateTime})
Get the variance of an ItoIntegral from one point of time to another.
"""
function variance(ito::ItoIntegral, from::Real, to::Real)
    if from + eps() > to
        return 0.0;
    end
    return evaluate_integral(ito.f_^2, from, to)
end
function variance(ito::ItoIntegral, base::Union{Date,DateTime}, from::Union{Date,DateTime}, to::Union{Date,DateTime})
    from_fl = years_between(from, base)
    to_fl   = years_between(to, base)
    return variance(ito, from_fl, to_fl)
end

"""
    volatility(ito::ItoIntegral, on::Union{Date,DateTime})
Get the volatility of an ItoIntegral on a certain date.
"""
function volatility(ito::ItoIntegral, on::Union{Date,DateTime})
    return ito.f_(on)
end
function volatility(ito::ItoIntegral, on::Real)
    return ito.f_(on)
end

"""
    covariance(ito1::ItoIntegral,ito2::ItoIntegral, from::Real, to::Real, gaussian_correlation::Real)
    covariance(ito1::ItoIntegral,ito2::ItoIntegral, base::Union{Date,DateTime}, from::Union{Date,DateTime}, to::Union{Date,DateTime}, gaussian_correlation::Real)
Get the covariance of two ItoIntegrals over a certain period given the underlying Brownian processes have a correlation of gaussian_correlation.
"""
function covariance(ito1::ItoIntegral,ito2::ItoIntegral, from::Real, to::Real, gaussian_correlation::Real)
    if from + eps() >= to
        return 0.0;
    end
    return gaussian_correlation * evaluate_integral(ito1.f_ * ito2.f_, from, to)
end
function covariance(ito1::ItoIntegral,ito2::ItoIntegral, base::Union{Date,DateTime}, from::Union{Date,DateTime}, to::Union{Date,DateTime}, gaussian_correlation::Real)
    from_fl = years_between(from, base)
    to_fl   = years_between(to, base)
    return covariance(ito1, ito2, from_fl, to_fl, gaussian_correlation)
end

"""
    correlation(ito1::ItoIntegral,ito2::ItoIntegral, from::Real, to::Real, gaussian_correlation::Real)
    correlation(ito1::ItoIntegral,ito2::ItoIntegral,  base::Union{Date,DateTime}, from::Union{Date,DateTime}, to::Union{Date,DateTime}, gaussian_correlation::Real)
Get the correlation of two ItoIntegrals over a certain period given the underlying Brownian processes have a correlation of gaussian_correlation.
"""
function correlation(ito1::ItoIntegral,ito2::ItoIntegral, from::Real, to::Real, gaussian_correlation::Real)
    cov =  covar(ito1,ito2, base, from, to, gaussian_correlation)
    var1 = var(ito1, from, to)
    var2 = var(ito2, from, to)
    return gaussian_correlation * (cov / (var1 * var2))
end
function correlation(ito1::ItoIntegral,ito2::ItoIntegral,  base::Union{Date,DateTime}, from::Union{Date,DateTime}, to::Union{Date,DateTime}, gaussian_correlation::Real)
    from_fl = years_between(from, base)
    to_fl   = years_between(to, base)
    return correlation(ito1, ito2, from_fl, to_fl, gaussian_correlation)
end

"""
    brownians_in_use(itos::Array{ItoIntegral,1}, brownians::Array{Symbol,1})
Determine which Browninan processes are used in an array of ItoIntegrals.
"""
function brownians_in_use(itos::Dict{Symbol,ItoIntegral}, brownians::Array{Symbol,1})
    all_brownians_in_use = unique(map(x -> x.brownian_id_ , values(itos)))
    indices_in_use   = unique(findall(map(x -> x in all_brownians_in_use , brownians)))
    reduced_brownian_list = brownians[indices_in_use]
    return all_brownians_in_use, indices_in_use, reduced_brownian_list
end

"""
    ItoSet
Creates an ItoSet. This contains :
* A correlation matrix of brownian motions.
* A vector giving the axis labels for this correlation matrix.
* A dict of ItoInterals. Here the keys should be ids for the ito integrals and the values should be ItoIntegrals.
Determine which Brownian processes are used in an array of ItoIntegrals.
"""
struct ItoSet{T<:Real}
    brownian_correlation_matrix_::Symmetric{T}
    brownian_ids_::Array{Symbol,1}
    ito_integrals_::Dict{Symbol,ItoIntegral}
    function ItoSet(brownian_corr_matrix::Symmetric{T}, brownian_ids::Array{Symbol,1}, ito_integrals::Dict{Symbol,ItoIntegral}) where T<:Real
        if (size(brownian_ids)[1] != size(brownian_corr_matrix)[1])
            error("The shape of brownian_ids_ must match the number of rows/columns of brownian_correlation_matrix_")
        end
          all_brownians_in_use, used_brownian_indices, brown_ids = brownians_in_use(ito_integrals, brownian_ids)
          if length(setdiff(all_brownians_in_use, brownian_ids)) > 0
              error(string("In creating an ItoSet there are some brownian motions referenced by ito integrals for which there are no corresponding entries in the correlation matrix for brownian motions. Thus an ItoSet cannot be built. These include ", setdiff(all_brownians_in_use, brownian_ids)))
          end
          brownian_corr_matrix_subset      = Symmetric(brownian_corr_matrix[used_brownian_indices,used_brownian_indices])
       return new{T}(brownian_corr_matrix_subset, brown_ids, ito_integrals)
    end
end

"""
    correlation(ito::ItoSet, index1::Integer, index2::Integer)
    correlation(ito::ItoSet, brownian_id1::Symbol, brownian_id2::Symbol)
Get correlation between brownian motions in an ItoSet.
"""
function correlation(ito::ItoSet, index1::Integer, index2::Integer)
    return ito.brownian_correlation_matrix_[index1, index2]
end
function correlation(ito::ItoSet, brownian_id1::Symbol, brownian_id2::Symbol)
    index1 = findall(brownian_id1 .== ito.brownian_ids_)[1]
    index2 = findall(brownian_id2 .== ito.brownian_ids_)[1]
    return correlation(ito, index1, index2)
end

"""
    volatility(ito::ItoSet, index::Integer, on::Union{Date,DateTime})
    volatility(ito::ItoSet, ito_integral_id::Symbol, on::Union{Date,DateTime})
Get volatility of an ito_integral on a date.
"""
function volatility(ito::ItoSet, index::Integer, on::Union{Date,DateTime})
    return volatility(ito.ito_integrals_[index], on)
end

function volatility(ito::ItoSet, ito_integral_id::Symbol, on::Union{Date,DateTime})
    ito_integral = ito.ito_integrals_[ito_integral_id]
    return volatility(ito_integral, on)
end

"""
    make_covariance_matrix(ito_set_::ItoSet, from::Real, to::Real)
Make a covariance matrix given an ItoSet and a period of time.
"""
function make_covariance_matrix(ito_set_::ItoSet{T}, from::Real, to::Real) where T<:Real
    number_of_itos = length(ito_set_.ito_integrals_)
    ito_ids = collect(keys(ito_set_.ito_integrals_))
    cov = Array{T,2}(undef, number_of_itos,number_of_itos)
    for r in 1:number_of_itos
        rito = ito_set_.ito_integrals_[ito_ids[r]]
        for c in r:number_of_itos
            #if c < r
            #    cov[r,c] = 0.0 # Since at the end we use the Symmetric thing, this is discarded so we don't bother computing it.
            #end
            cito = ito_set_.ito_integrals_[ito_ids[c]]
            cr_correlation = correlation(ito_set_, rito.brownian_id_, cito.brownian_id_)
            cov[r,c] = covariance(rito, cito, from, to, cr_correlation)
        end
    end
    return Symmetric(cov), ito_ids
end

"""
    ForwardCovariance
Creates an ForwardCovariance object. This contains :
* An Itoset
* Time From
* Time To
And in the constructor the following items are generated and stored in the object:
* A covariance matrix
* Labels for the covariance matrix.
* The cholesky decomposition of the covariance matrix.
* The inverse of the covariance matrix.
* The determinant of the covariance matrix.
"""
struct ForwardCovariance
    ito_set_::ItoSet
    from_::Real
    to_::Real
    covariance_::Symmetric
    covariance_labels_::Array{Symbol,1}
    chol_::Union{Missing,LowerTriangular}
    inverse_::Union{Missing,Symmetric}
    determinant_::Union{Missing,Real}
    """
    ForwardCovariance(ito_set_::ItoSet, from_::Real, to_::Real;
             calculate_chol::Bool = true, calculate_inverse::Bool = true, calculate_determinant::Bool = true)
    ForwardCovariance(ito_set_::ItoSet, from::Union{Date,DateTime}, to::Union{Date,DateTime})
    ForwardCovariance(old_ForwardCovariance::ForwardCovariance, from::Real, to::Real)
    ForwardCovariance(old_ForwardCovariance::ForwardCovariance, from::Union{Date,DateTime}, to::Union{Date,DateTime})
        These are constructors for a ForwardCovariance struct.
    """
    function ForwardCovariance(ito_set_::ItoSet, from_::Real, to_::Real;
             calculate_chol::Bool = true, calculate_inverse::Bool = true, calculate_determinant::Bool = true)
        covariance_, covariance_labels_ = make_covariance_matrix(ito_set_, from_, to_)
        chol_ = missing
        inverse_ = missing
        determinant_ = missing
        if calculate_chol chol_ = LowerTriangular(cholesky(covariance_).L) end
        if calculate_inverse inverse_           = Symmetric(inv(covariance_)) end
        if calculate_determinant determinant_       = det(covariance_) end
        return new(ito_set_, from_, to_, covariance_, covariance_labels_, chol_, inverse_, determinant_)
    end
    function ForwardCovariance(ito_set_::ItoSet, from::Union{Date,DateTime}, to::Union{Date,DateTime};
             calculate_chol::Bool = true, calculate_inverse::Bool = true, calculate_determinant::Bool = true)
        from_ = years_from_global_base(from)
        to_   = years_from_global_base(to)
        return ForwardCovariance(ito_set_, from_, to_; calculate_chol = calculate_chol,
                  calculate_inverse = calculate_inverse, calculate_determinant = calculate_determinant)
    end
    function ForwardCovariance(old_ForwardCovariance::ForwardCovariance, from::Real, to::Real; recalculate_all::Bool = true)
        if recalculate_all
            return ForwardCovariance(old_ForwardCovariance.ito_set_, from, to)
        else
            old_duration = old_ForwardCovariance.to_ - old_ForwardCovariance.from_
            new_duration = to - from
            covariance_  = old_ForwardCovariance.covariance_ * (new_duration/old_duration)
            covariance_labels_ = old_ForwardCovariance.covariance_labels_

            chol_ = ismissing(old_ForwardCovariance.chol_) ? missing : LowerTriangular(old_ForwardCovariance.chol_ .* sqrt((new_duration/old_duration)))
            inverse_ = ismissing(old_ForwardCovariance.inverse_) ? missing : Symmetric(old_ForwardCovariance.inverse_ ./  ((new_duration/old_duration)))
            determinant_ = ismissing(old_ForwardCovariance.determinant_) ? missing : old_ForwardCovariance.determinant_ *  ((new_duration/old_duration)^length(covariance_labels_))
            return new(old_ForwardCovariance.ito_set_, from, to, covariance_, covariance_labels_, chol_, inverse_, determinant_)
        end
    end
    function ForwardCovariance(old_ForwardCovariance::ForwardCovariance, from::Union{Date,DateTime}, to::Union{Date,DateTime}; recalculate_all::Bool = true)
        from_ = years_from_global_base(from)
        to_   = years_from_global_base(to)
        return ForwardCovariance(old_ForwardCovariance.ito_set_, from_, to_; recalculate_all = recalculate_all)
    end
end

"""
    volatility(covar::ForwardCovariance, index::Integer, on::Union{Date,DateTime})
    volatility(covar::ForwardCovariance, id::Symbol, on::Union{Date,DateTime})
Get the volatility of an ItoIntegral on a date..
"""
function volatility(covar::ForwardCovariance, index::Integer, on::Union{Date,DateTime})
    return volatility(covar.ito_set_, index, on)
end
function volatility(covar::ForwardCovariance, id::Symbol, on::Union{Date,DateTime})
    return volatility(covar.ito_set_, id, on)
end

"""
    variance(covar::ForwardCovariance, id::Symbol)
    variance(covar::ForwardCovariance, index::Integer)
Get the variance of an ItoIntegral over a period.
"""
function variance(covar::ForwardCovariance, id::Symbol)
        index = findall(id .== covar.covariance_labels_)[1]
        return variance(covar, index)
end
function variance(covar::ForwardCovariance, index::Integer)
    return covar.covariance_[index,index]
end

"""
    covariance(covar::ForwardCovariance, index_1::Integer, index_2::Integer)
    covariance(covar::ForwardCovariance, id1::Symbol, id2::Symbol)
Get the covariance of two ItoIntegrals over a period.
"""
function covariance(covar::ForwardCovariance, index_1::Integer, index_2::Integer)
    return covar.covariance_[index_1,index_2]
end
function covariance(covar::ForwardCovariance, id1::Symbol, id2::Symbol)
    index_1 = findall(id1 .== covar.covariance_labels_)[1]
    index_2 = findall(id2 .== covar.covariance_labels_)[1]
    return covariance(covar, index_1, index_2)
end

"""
    correlation(covar::ForwardCovariance, index_1::Integer, index_2::Integer)
    correlation(covar::ForwardCovariance, id1::Symbol, id2::Symbol)
Get the correlation of two ItoIntegrals over a period.
"""
function correlation(covar::ForwardCovariance, index_1::Integer, index_2::Integer)
    cova = covariance(covar, index_1, index_2)
    var1  = variance(covar, index_1)
    var2  = variance(covar, index_2)
    return cova/sqrt(var1 * var2)
end
function correlation(covar::ForwardCovariance, id1::Symbol, id2::Symbol)
    index_1 = findall(id1 .== covar.covariance_labels_)[1]
    index_2 = findall(id2 .== covar.covariance_labels_)[1]
    return correlation(covar, index_1, index_2)
end

## Random draws
"""
    get_draws(covar::ForwardCovariance; uniform_draw::Array{T,1} = rand(length(covar.covariance_labels_))) where T<:Real
    get_draws(covar::ForwardCovariance, num::Integer; twister::MersenneTwister = MersenneTwister(1234), antithetic_variates = false)
get pseudorandom draws from a ForwardCovariance struct. Other schemes (like quasirandom) can be done by inserting quasirandom
numbers in as the uniform_draw.
If the antithetic_variates control is set to true then every second set of draws will be antithetic to the previous.
"""
function get_draws(covar::ForwardCovariance; uniform_draw::Array{T,1} = rand(length(covar.covariance_labels_))) where T<:Real
    number_of_itos = length(covar.covariance_labels_)
    normal_draw = quantile.(Ref(Normal()), uniform_draw)
    scaled_draw = covar.chol_ * normal_draw
    first_set_of_draws = Dict{Symbol,Real}(covar.covariance_labels_ .=> scaled_draw)
    return first_set_of_draws
end
function get_draws(covar::ForwardCovariance, num::Integer; number_generator::NumberGenerator = Mersenne(MersenneTwister(1234), length(covar.covariance_labels_)), antithetic_variates = false)
    if antithetic_variates
        half_num = convert(Int, round(num/2))
        array_of_dicts = Array{Dict{Symbol,Real}}(undef, half_num*2)
        number_of_itos = length(covar.covariance_labels_)
        for i in 1:half_num
            first_entry = 2*(i-1)+1
            draw = next!(number_generator)
            array_of_dicts[first_entry] = get_draws(covar; uniform_draw = draw)
            array_of_dicts[first_entry + 1] = get_draws(covar; uniform_draw = 1.0 .- draw)
        end
        return array_of_dicts
    else
        array_of_dicts = Array{Dict{Symbol,Real}}(undef, num)
        number_of_itos = length(covar.covariance_labels_)
        for i in 1:num
            draw =  next!(number_generator)
            array_of_dicts[i] = get_draws(covar; uniform_draw = draw)
        end
        return array_of_dicts
    end
end

# This is most likely useful for bug hunting.
"""
    get_zero_draws(covar::ForwardCovariance)
get a draw of zero for all ito_integrals. May be handy for bug hunting.
"""
function get_zero_draws(covar::ForwardCovariance)
    return Dict{Symbol,Real}(covar.covariance_labels_ .=> 0.0)
end
# This is most likely useful for bug hunting.
"""
    get_zero_draws(covar::ForwardCovariance, num::Integer)
get an array of zero draws for all ito_integrals. May be handy for bug hunting.
"""
function get_zero_draws(covar::ForwardCovariance, num::Integer)
    array_of_dicts = Array{Dict{Symbol,Real}}(undef, num)
    for i in 1:num
        array_of_dicts[i] = get_zero_draws(covar)
    end
    return array_of_dicts
end

"""
    pdf(covar::ForwardCovariance, coordinates::Dict{Symbol,Real})
get the value of the pdf at some coordinates.
"""
function pdf(covar::ForwardCovariance, coordinates::Dict{Symbol,Real})
    # The pdf is det(2\pi\Sigma)^{-0.5}\exp(-0.5(x - \mu)^\prime \Sigma^{-1} (x - \mu))
    # Where Sigma is covariance matrix, \mu is means (0 in this case) and x is the coordinates.
    rank_of_matrix = length(covar.covariance_labels_)
    x = get.(Ref(coordinates), covar.covariance_labels_, 0)
    one_on_sqrt_of_det_two_pi_covar     = 1/(sqrt(covar.determinant_) * (2*pi)^(rank_of_matrix/2))
    return one_on_sqrt_of_det_two_pi_covar * exp(-0.5 * x' * covar.inverse_ * x)
end

"""
    log_likelihood(covar::ForwardCovariance, coordinates::Dict{Symbol,Real})
get the log likelihood at some coordinates.
"""
function log_likelihood(covar::ForwardCovariance, coordinates::Dict{Symbol,Real})
    # The pdf is det(2\pi\Sigma)^{-0.5}\exp(-0.5(x - \mu)^\prime \Sigma(x - \mu))
    # Where Sigma is covariance matrix, \mu is means (0 in this case) and x is the coordinates.
    rank_of_matrix = length(covar.covariance_labels_)
    x = get.(Ref(coordinates), covar.covariance_labels_, 0)
    one_on_sqrt_of_det_two_pi_covar     = -0.5*log(covar.determinant_)  +  (rank_of_matrix/2)*log(2*pi)
    return one_on_sqrt_of_det_two_pi_covar + (-0.5 * x' * covar.inverse_ * x)
end
