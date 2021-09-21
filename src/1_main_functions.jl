const tol = 10*eps()

"""
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
### Inputs
* `ito` - The ito integral you want the variance for.
* `from` - The time at which the integration starts.
* `to` - The time at which the integration ends.
### Outputs
* A scalar
"""
function variance(ito::ItoIntegral, from::Real, to::Real)
    if from + eps() > to
        return 0.0;
    end
    return evaluate_integral(ito.f_^2, from, to)
end
function variance(ito::ItoIntegral, base::Union{Date,DateTime},
                           from::Union{Date,DateTime}, to::Union{Date,DateTime})
    from_fl = years_between(from, base)
    to_fl   = years_between(to, base)
    return variance(ito, from_fl, to_fl)
end

"""
    volatility(ito::ItoIntegral, on::Union{Date,DateTime})
Get the volatility of an ItoIntegral on a certain date.
### Inputs
* `ito` - The ito integral you want the volatility for.
* `on` - What instant do you want the volatility for.
### Outputs
* A scalar
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
### Inputs
* `ito1` - The first ito integral
* `ito2` - The second ito integral
* `from` - The start of the period
* `to` - The end of the period
* `gaussian_correlation` - The correlation between the brownians for each of the two itos. This should be in the range [-1,1].
* `on` - What instant do you want the volatility for.
### Outputs
* A scalar
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
    brownians_in_use(itos::Array{ItoIntegral,1}, brownians::Array{Symbol,1})
Determine which Browninan processes are used in an array of ItoIntegrals.
### Inputs
* itos - A dict containing each of the ito integrals
* brownians - All possible brownians
### Outputs
* A `Vector` of what brownians are in use in itos
* A `Vector` with the indices of these brownians.
"""
function brownians_in_use(itos::Dict{Symbol,ItoIntegral}, brownians::Array{Symbol,1})
    all_brownians_in_use = unique(map(x -> x.brownian_id_ , values(itos)))
    indices_in_use   = unique(findall(map(x -> x in all_brownians_in_use , brownians)))
    reduced_brownian_list = brownians[indices_in_use]
    return all_brownians_in_use, indices_in_use, reduced_brownian_list
end

"""
Creates an ItoSet. This contains :
* A correlation matrix of brownian motions.
* A vector giving the axis labels for this correlation matrix.
* A dict of ItoInterals. Here the keys should be ids for the ito integrals and the values should be ItoIntegrals.
Determine which Brownian processes are used in an array of ItoIntegrals.
"""
struct ItoSet{T<:Real}
    brownian_correlation_matrix_::Hermitian{T}
    brownian_ids_::Array{Symbol,1}
    ito_integrals_::Dict{Symbol,ItoIntegral}
    function ItoSet(brownian_corr_matrix::Hermitian{T}, brownian_ids::Array{Symbol,1},
                    ito_integrals::Dict{Symbol,ItoIntegral}) where T<:Real
        if size(brownian_ids)[1] != size(brownian_corr_matrix)[1]
            error("The shape of brownian_ids_ must match the number of rows/columns of brownian_correlation_matrix_")
        end
        all_brownians_in_use, used_brownian_indices, brown_ids = brownians_in_use(ito_integrals, brownian_ids)
        if length(setdiff(all_brownians_in_use, brownian_ids)) > 0
            error(string("In creating an ItoSet there are some brownian motions referenced by ito integrals for which there are no corresponding entries in the correlation matrix for brownian motions. Thus an ItoSet cannot be built. These include ", setdiff(all_brownians_in_use, brownian_ids)))
        end
        brownian_corr_matrix_subset = Hermitian(brownian_corr_matrix[used_brownian_indices,used_brownian_indices])
        return new{T}(brownian_corr_matrix_subset, brown_ids, ito_integrals)
    end
end

"""
    correlation(ito::ItoSet, index1::Integer, index2::Integer)
    correlation(ito::ItoSet, brownian_id1::Symbol, brownian_id2::Symbol)
Get correlation between brownian motions in an ItoSet.
### Inputs
* `ito` - An ItoSet that you want the correlation for two itos within.
* `index1` or `brownian_id1` - The index/key for the first ito integral.
* `index2` or `brownian_id2` - The index/key for the second ito integral.
### Returns
* A scalar
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
### Inputs
* `ito` - An ItoSet that you want the volatility for.
* `index` - The key of the ito dict that you are interested in
* `on` The time or instant you want the volatility for.
### Returns
* A scalar
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
Make a covariance matrix given an ItoSet and a period of time. This returns a
Hermitian covariance matrix as well as a vector of symbols representing the axis
labelling on this Hermitian.

### Inputs
* `ito_set_` - An ItoSet you want to make a covariance matrix from.
* `from` - The (numeric) time from which the covariance span starts.
* `to` - The (numeric) time at which the covariance span ends.
### Returns
* A `Hermitian` covariance matrix.
* A `Vector{Symbol}` of labels for the covariance matrix.
"""
function make_covariance_matrix(ito_set_::ItoSet{T}, from::Real, to::Real) where T<:Real
    number_of_itos = length(ito_set_.ito_integrals_)
    ito_ids = collect(keys(ito_set_.ito_integrals_))
    cov = Array{T,2}(undef, number_of_itos,number_of_itos)
    for r in 1:number_of_itos
        rito = ito_set_.ito_integrals_[ito_ids[r]]
        for c in r:number_of_itos
            cito = ito_set_.ito_integrals_[ito_ids[c]]
            cr_correlation = correlation(ito_set_, rito.brownian_id_, cito.brownian_id_)
            cov[r,c] = covariance(rito, cito, from, to, cr_correlation)
        end
    end
    return Hermitian(cov), ito_ids
end

"""
    StochasticIntegralsCovariance
This is an abstract type which represents structs that represent covariances over spans of time.
The concrete instances of this type should support extracting correlations, covariances, volatilities
and random number generation.
"""
abstract type StochasticIntegralsCovariance end

"""
    ForwardCovariance
Creates an ForwardCovariance struct. This contains :
* An Itoset
* Time From
* Time To
And in the constructor the following items are generated and stored in the object:
* A covariance matrix
* Labels for the covariance matrix.
* The cholesky decomposition of the covariance matrix.
* The inverse of the covariance matrix.
* The determinant of the covariance matrix.

The constructors are:
    ForwardCovariance(ito_set_::ItoSet, from_::Real, to_::Real;
         calculate_chol::Bool = true, calculate_inverse::Bool = true, calculate_determinant::Bool = true)
    ForwardCovariance(ito_set_::ItoSet, from::Union{Date,DateTime}, to::Union{Date,DateTime})
    ForwardCovariance(old_ForwardCovariance::ForwardCovariance, from::Real, to::Real)
    ForwardCovariance(old_ForwardCovariance::ForwardCovariance, from::Union{Date,DateTime}, to::Union{Date,DateTime})
"""
struct ForwardCovariance <:StochasticIntegralsCovariance
    ito_set_::ItoSet
    from_::Real
    to_::Real
    covariance_::Hermitian
    covariance_labels_::Array{Symbol,1}
    chol_::Union{Missing,LowerTriangular}
    inverse_::Union{Missing,Hermitian}
    determinant_::Union{Missing,Real}
    function ForwardCovariance(ito_set_::ItoSet, from_::Real, to_::Real;
             calculate_chol::Bool = true, calculate_inverse::Bool = true, calculate_determinant::Bool = true)
        covariance_, covariance_labels_ = make_covariance_matrix(ito_set_, from_, to_)
        chol_ = missing
        inverse_ = missing
        determinant_ = missing
        if calculate_chol chol_ = LowerTriangular(cholesky(covariance_).L) end
        if calculate_inverse inverse_           = Hermitian(inv(covariance_)) end
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
            relative_duration = new_duration/old_duration
            covariance_  = old_ForwardCovariance.covariance_ * (relative_duration)
            covariance_labels_ = old_ForwardCovariance.covariance_labels_

            chol_ = ismissing(old_ForwardCovariance.chol_) ? missing : LowerTriangular(old_ForwardCovariance.chol_ .* sqrt((relative_duration)))
            inverse_ = ismissing(old_ForwardCovariance.inverse_) ? missing : Hermitian(old_ForwardCovariance.inverse_ ./  ((relative_duration)))
            determinant_ = ismissing(old_ForwardCovariance.determinant_) ? missing : old_ForwardCovariance.determinant_ *  ((relative_duration)^length(covariance_labels_))
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
    SimpleCovariance
Creates an SimpleCovariance struct. This is a simplified version of ForwardCovariance
and is designed for Ito Integrals that are constant (so correlations do not need to be recalculated).
This computational saving is the only advantage - ForwardCovariance is more general.
It contains the same elements as a ForwardCovariance:
* An Itoset
* Time From
* Time To
And in the constructor the following items are generated and stored in the object:
* A covariance matrix
* Labels for the covariance matrix.
* The cholesky decomposition of the covariance matrix.
* The inverse of the covariance matrix.
* The determinant of the covariance matrix.

The constructors are:
    SimpleCovariance(ito_set_::ItoSet, from_::Real, to_::Real;
         calculate_chol::Bool = true, calculate_inverse::Bool = true, calculate_determinant::Bool = true)
"""
mutable struct SimpleCovariance <: StochasticIntegralsCovariance
    ito_set_::ItoSet
    from_::Real
    to_::Real
    covariance_::Hermitian
    covariance_labels_::Array{Symbol,1}
    chol_::Union{Missing,LowerTriangular}
    inverse_::Union{Missing,Hermitian}
    determinant_::Union{Missing,Real}
    function SimpleCovariance(ito_set_::ItoSet, from_::Real, to_::Real;
             calculate_chol::Bool = true, calculate_inverse::Bool = true, calculate_determinant::Bool = true)
        fc = ForwardCovariance(ito_set_, from_, to_; calculate_chol = calculate_chol, calculate_inverse = calculate_inverse, calculate_determinant = calculate_determinant)
        return new(fc.ito_set_, fc.from_, fc.to_, fc.covariance_, fc.covariance_labels_, fc.chol_, fc.inverse_, fc.determinant_)
    end
end

"""
    update!(sc::SimpleCovariance, from::Real, to::Real)
This takes a SimpleCovariance and updates it for a new  span in time.
The new span in time is between from and to. For SimpleCovariance this is done by
just adjusting the covariances for the new time span (with corresponding adjustments)
to the cholesky, inverse, etc.

The corresponding technique for a ForwardCovariance (which is also a StochasticIntegralsCovariance)
is to feed it into a new ForwardCovariance constructor which will recalculate for the new span.
### Inputs
* `sc` - The SimpleCovariance struct.
* `from` - The time from which you want the covariance for.
* `to` - The time to which you want the covariance for.
### Returns
Nothing. It juts updates the sc struct you pass in as an input.
"""
function update!(sc::SimpleCovariance, from::Real, to::Real)
    old_duration = sc.to_ - sc.from_
    new_duration = to - from
    relative_duration = new_duration/old_duration
    sc.covariance_  = sc.covariance_ * (relative_duration)
    sc.chol_ = ismissing(sc.chol_) ? missing : LowerTriangular(sc.chol_ .* sqrt(relative_duration))
    sc.inverse_ = ismissing(sc.inverse_) ? missing : Hermitian(sc.inverse_ ./  (relative_duration))
    sc.determinant_ = ismissing(sc.determinant_) ? missing : sc.determinant_ *  ((relative_duration)^length(covariance_labels_))
    sc.from_ = from
    sc.to_ = to
end


"""
    volatility(covar::ForwardCovariance, index::Integer, on::Union{Date,DateTime})
    volatility(covar::ForwardCovariance, id::Symbol, on::Union{Date,DateTime})
Get the volatility of an ForwardCovariance on a date.
### Inputs
* `covar` - An ForwardCovariance that you want the volatility for.
* `index` - The key of the ito dict that you are interested in
* `on` The time or instant you want the volatility for.
### Returns
* A scalar
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
Get the variance of an ForwardCovariance over a period.
### Inputs
* `covar` - An ForwardCovariance that you want the variance for.
* `id` or `index` - The key/index of the ito dict that you are interested in
### Returns
* A scalar
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
Get the covariance of two ito integrals in a ForwardCovariance over a period.
### Inputs
* `covar` - An ForwardCovariance that you want the covariance for.
* `index_1` or `id1` - The key/index of the first ito that you are interested in
* `index_2` or `id2` - The key/index of the second ito that you are interested in
### Returns
* A scalar
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
### Inputs
* `covar` - An ForwardCovariance that you want the correlation for.
* `index_1` or `id1` - The key/index of the first ito that you are interested in
* `index_2` or `id2` - The key/index of the second ito that you are interested in
### Returns
* A scalar
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

### Inputs
* `covar` - An ForwardCovariance or SimpleCovariance struct that you want to draw from.
* `uniform_draw`- The draw vector (from the uniform distribution)
### Returns
* A Dict of draws.
"""
function get_draws(covar::Union{ForwardCovariance,SimpleCovariance}; uniform_draw::Array{T,1} = rand(length(covar.covariance_labels_))) where T<:Real
    number_of_itos = length(covar.covariance_labels_)
    normal_draw = quantile.(Ref(Normal()), uniform_draw)
    scaled_draw = covar.chol_ * normal_draw
    first_set_of_draws = Dict{Symbol,Real}(covar.covariance_labels_ .=> scaled_draw)
    return first_set_of_draws
end
"""
    get_draws(covar::ForwardCovariance, num::Integer; twister::MersenneTwister = MersenneTwister(1234), antithetic_variates = false)
get pseudorandom draws from a ForwardCovariance struct. Other schemes (like quasirandom) can be done by inserting quasirandom
numbers in as the uniform_draw.
If the antithetic_variates control is set to true then every second set of draws will be antithetic to the previous.

### Inputs
* `covar` - An ForwardCovariance or SimpleCovariance struct that you want to draw from.
* `num`- The number of draws you want
* `twister`- The number of draws you want
### Returns
* A `Vector` of `Dict`s of draws.
"""
function get_draws(covar::Union{ForwardCovariance,SimpleCovariance}, num::Integer; number_generator::NumberGenerator = Mersenne(MersenneTwister(1234), length(covar.covariance_labels_)), antithetic_variates = false)
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
### Inputs
* covar - the ForwardCovariance struct that you want zero draws from
### Outputs
* A `Dict` of zero draws
"""
function get_zero_draws(covar::ForwardCovariance)
    return Dict{Symbol,Real}(covar.covariance_labels_ .=> 0.0)
end
# This is most likely useful for bug hunting.
"""
    get_zero_draws(covar::ForwardCovariance, num::Integer)
get an array of zero draws for all ito_integrals. May be handy for bug hunting.
### Inputs
* `covar` - the ForwardCovariance struct that you want zero draws from
* `num` - The number of zero draws you want.
### Outputs
* A `Vector` of `Dict`s of zero draws
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
get the value of the pdf at some coordinates. Note that it is assumed that
the mean of the multivariate gaussian is the zero vector.
### Inputs
* `covar` - the ForwardCovariance struct that you want to evaluate the pdf
* `coordinates` - The coordinates you want to examine.
### Outputs
* A scalar
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
get the log likelihood at some coordinates. Note that it is assumed that
the mean of the multivariate gaussian is the zero vector.
### Inputs
* `covar` - the ForwardCovariance struct that you want to evaluate the log likelihood.
* `coordinates` - The coordinates you want to examine.
### Outputs
* A scalar
"""
function log_likelihood(covar::ForwardCovariance, coordinates::Dict{Symbol,R}) where R<:Real
    # The pdf is det(2\pi\Sigma)^{-0.5}\exp(-0.5(x - \mu)^\prime \Sigma(x - \mu))
    # Where Sigma is covariance matrix, \mu is means (0 in this case) and x is the coordinates.
    rank_of_matrix = length(covar.covariance_labels_)
    x = get.(Ref(coordinates), covar.covariance_labels_, 0)
    one_on_sqrt_of_det_two_pi_covar     = -0.5*log(covar.determinant_)  +  (rank_of_matrix/2)*log(2*pi)
    return one_on_sqrt_of_det_two_pi_covar + (-0.5 * x' * covar.inverse_ * x)
end
