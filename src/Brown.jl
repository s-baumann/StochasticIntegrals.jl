const tol = 10*eps()

struct ito_integral
    brownian_id_::String
    ito_integral_id_::String
    f_::UnivariateFunction
end

function flat_ito(brownian_id_, ito_integral_id_, variance_)
    return ito_integral(brownian_id_, ito_integral_id_, PE_Function(variance_, 0.0, 0.0, 0))
end

function get_variance(ito::ito_integral, from::Float64, to::Float64)
    return evaluate_integral(ito.f_^2,from,to)
end

function get_variance(ito::ito_integral, base::Date, from::Date, to::Date)
    from_fl = years_between(from, base)
    to_fl   = years_between(to, base)
    return get_variance(ito, from_fl, to_fl)
end

function get_volatility(ito::ito_integral, on::Date)
    return evaluate(ito.f_, on)
end

function get_covariance(ito1::ito_integral,ito2::ito_integral, from::Float64, to::Float64, gaussian_correlation::Float64)
    return gaussian_correlation * evaluate_integral(ito1.f_ * ito2.f_, from, to)
end


function get_covariance(ito1::ito_integral,ito2::ito_integral, base::Date, from::Date, to::Date, gaussian_correlation::Float64)
    from_fl = years_between(from, base)
    to_fl   = years_between(to, base)
    return get_covariance(ito1, ito2, from_fl, to_fl, gaussian_correlation)
end

function get_correlation(ito1::ito_integral,ito2::ito_integral, from::Float64, to::Float64, gaussian_correlation::Float64)
    cov =  covar(ito1,ito2, base, from, to, gaussian_correlation)
    var1 = var(ito1, from, to)
    var2 = var(ito2, from, to)
    return gaussian_correlation * (cov / (var1 * var2))
end

function get_correlation(ito1::ito_integral,ito2::ito_integral,  base::Date, from::Date, to::Date, gaussian_correlation::Float64)
    from_fl = years_between(from, base)
    to_fl   = years_between(to, base)
    return get_correlation(ito1, ito2, from_fl, to_fl, gaussian_correlation)
end

function brownians_in_use(itos::Array{ito_integral}, brownians::Array{String})
    brownians_in_use = unique(map(x -> x.brownian_id_ , itos))
    indices_in_use   = unique(findall(map(x -> x in brownians_in_use , brownians)))
    reduced_brownian_list = brownians[indices_in_use]
    return indices_in_use, reduced_brownian_list
end

struct ito_set
    brownian_correlation_matrix_::Symmetric
    brownian_ids_::Array{String}
    ito_integrals_::Array{ito_integral}
    function ito_set(brownian_corr_matrix::Symmetric, brownian_ids::Array{String}, ito_integrals::Array{ito_integral})
        if (size(brownian_ids)[1] != size(brownian_corr_matrix)[1])
            error("The shape of brownian_ids_ must match the number of rows/columns of brownian_correlation_matrix_")
        end
          used_brownian_indices, brown_ids = brownians_in_use(ito_integrals, brownian_ids)
          brownian_corr_matrix_subset      = Symmetric(brownian_corr_matrix[used_brownian_indices,used_brownian_indices])
       return new(brownian_corr_matrix_subset, brown_ids, ito_integrals)
    end
end

function get_ito_integral_ids(ito_integrals_::Array{ito_integral})
    return map(p->p.ito_integral_id_, ito_integrals_)
end

function get_correlation(ito::ito_set, index1::Int, index2::Int)
    return ito.brownian_correlation_matrix_[index1, index2]
end

function get_correlation(ito::ito_set, brownian_id1::String, brownian_id2::String)
    index1 = findall(brownian_id1 .== ito.brownian_ids_)[1]
    index2 = findall(brownian_id2 .== ito.brownian_ids_)[1]
    return get_correlation(ito, index1, index2)
end

function get_volatility(ito::ito_set, index::Int, on::Date)
    return get_volatility(ito.ito_integrals_[index], on)
end

function get_volatility(ito::ito_set, ito_integral_id::String, on::Date)
    index = findall(ito_integral_id .== map(p->p.ito_integral_id_, ito.ito_integrals_))[1]
    return get_volatility(ito, index, on)
end

function make_covariance_matrix(ito_set_::ito_set, from::Float64, to::Float64)
    number_of_itos = size(ito_set_.ito_integrals_)[1]
    cov = Array{Float64}(undef, number_of_itos,number_of_itos)
    for r in 1:number_of_itos
        rito = ito_set_.ito_integrals_[r]
        for c in 1:number_of_itos
            if c < r
                cov[r,c] = 0.0 # Since at the end we use the Symmetric thing, this is discarded so we don't bother computing it.
            end
            cito = ito_set_.ito_integrals_[c]
            cr_correlation = get_correlation(ito_set_, rito.brownian_id_, cito.brownian_id_)
            cov[r,c] = get_covariance(rito, cito, from, to, cr_correlation)
        end
    end
    return Symmetric(cov)
end

struct covariance_at_date
    ito_set_::ito_set
    from_::Float64
    to_::Float64
    covariance_labels_::Array{String}
    covariance_::Symmetric
    chol_::LowerTriangular
    inverse_::Symmetric
    determinant_::Float64
    function covariance_at_date(ito_set_::ito_set, from_::Float64, to_::Float64)
        covariance_labels_ = get_ito_integral_ids(ito_set_.ito_integrals_)
        covariance_        = make_covariance_matrix(ito_set_, from_, to_)
        chol_              = LowerTriangular(cholesky(covariance_).L)
        inverse_           = Symmetric(inv(covariance_))
        determinant_       = det(covariance_)
        return new(ito_set_, from_, to_, covariance_labels_, covariance_, chol_, inverse_, determinant_)
    end
    function covariance_at_date(ito_set_::ito_set, from::Date, to::Date)
        from_ = years_from_global_base(from)
        to_   = years_from_global_base(to)
        return covariance_at_date(ito_set_, from_, to_)
    end
    function covariance_at_date(old_covariance_at_date::covariance_at_date, from::Float64, to::Float64)
        return covariance_at_date(old_covariance_at_date.ito_set_, from, to)
    end
    function covariance_at_date(old_covariance_at_date::covariance_at_date, from::Date, to::Date)
        return covariance_at_date(old_covariance_at_date.ito_set_, from, to)
    end
end

function get_volatility(covar::covariance_at_date, index::Int, on::Date)
    return get_volatility(covar.ito_set_, index, on)
end

function get_volatility(covar::covariance_at_date, id::String, on::Date)
    return get_volatility(covar.ito_set_, id, on)
end

function get_variance(covar::covariance_at_date, id::String)
        index = findall(id .== covar.covariance_labels_)[1]
        return get_variance(covar, index)
end

function get_variance(covar::covariance_at_date, index::Int)
    return covar.covariance_[index,index]
end

function get_covariance(covar::covariance_at_date, index_1::Int, index_2::Int)
    return covar.covariance_[index_1,index_2]
end

function get_covariance(covar::covariance_at_date, id1::String, id2::String)
    index_1 = findall(id1 .== covar.covariance_labels_)[1]
    index_2 = findall(id2 .== covar.covariance_labels_)[1]
    return get_covariance(covar, index_1, index_2)
end

function get_correlation(covar::covariance_at_date, index_1::Int, index_2::Int)
    covariance = get_covariance(covar, index_1, index_2)
    var1  = get_variance(covar, index_1)
    var2  = get_variance(covar, index_2)
    return covariance/sqrt(var1 * var2)
end

function get_correlation(covar::covariance_at_date, id1::String, id2::String)
    index_1 = findall(id1 .== covar.covariance_labels_)[1]
    index_2 = findall(id2 .== covar.covariance_labels_)[1]
    return get_correlation(covar, index_1, index_2)
end

## Random draws
function get_normal_draws(covar::covariance_at_date; uniform_draw::Array{Float64} = rand(length(covar.covariance_labels_)))
    number_of_itos = length(covar.covariance_labels_)
    normal_draw = quantile.(Ref(Normal()), uniform_draw)
    scaled_draw = covar.chol_ * normal_draw
    return Dict{String,Float64}(covar.covariance_labels_ .=> scaled_draw)
end
function get_normal_draws(covar::covariance_at_date, num::Int)
    array_of_dicts = Array{Dict{String,Float64}}(undef, num)
    for i in 1:num
        array_of_dicts[i] = get_normal_draws(covar)
    end
    return array_of_dicts
end

function get_sobol_normal_draws(covar::covariance_at_date, sob_seq::SobolSeq)
    sobol_draw = next!(sob_seq)
    return get_normal_draws(covar; uniform_draw = sobol_draw)
end
function get_sobol_normal_draws(covar::covariance_at_date, sob_seq::SobolSeq, num::Int)
    array_of_dicts = Array{Dict{String,Float64}}(undef, num)
    for i in 1:num
        array_of_dicts[i] = get_sobol_normal_draws(covar,sob_seq)
    end
    return array_of_dicts
end



# This is most likely useful for bug hunting.
function get_zero_draws(covar::covariance_at_date)
    return Dict{String,Float64}(covar.covariance_labels_ .=> 0.0)
end
# This is most likely useful for bug hunting.
function get_zero_draws(covar::covariance_at_date, num::Int)
    array_of_dicts = Array{Dict{String,Float64}}(undef, num)
    for i in 1:num
        array_of_dicts[i] = get_zero_draws(covar)
    end
    return array_of_dicts
end



function pdf(covar::covariance_at_date, coordinates::Dict{String,Float64})
    # The pdf is det(2\pi\Sigma)^{-0.5}\exp(-0.5(x - \mu)^\prime \Sigma(x - \mu))
    # Where Sigma is covariance matrix, \mu is means (0 in this case) and x is the coordinates.
    rank_of_matrix = length(covar.covariance_labels_)
    x = get.(Ref(coordinates), covar.covariance_labels_, 0)
    one_on_sqrt_of_det_two_pi_covar     = 1/(sqrt(covar.determinant_) * (2*pi)^(rank_of_matrix/2))
    return one_on_sqrt_of_det_two_pi_covar * exp(-0.5 * x' * covar.covariance_ * x)
end

function log_likelihood(covar::covariance_at_date, coordinates::Dict{String,Float64})
    # The pdf is det(2\pi\Sigma)^{-0.5}\exp(-0.5(x - \mu)^\prime \Sigma(x - \mu))
    # Where Sigma is covariance matrix, \mu is means (0 in this case) and x is the coordinates.
    rank_of_matrix = length(covar.covariance_labels_)
    x = get.(Ref(coordinates), covar.covariance_labels_, 0)
    one_on_sqrt_of_det_two_pi_covar     = -0.5*log(covar.determinant_)  +  (rank_of_matrix/2)*log(2*pi)
    return one_on_sqrt_of_det_two_pi_covar + (-0.5 * x' * covar.covariance_ * x)
end
