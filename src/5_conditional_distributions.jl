
"""
    generate_conditioned_dist(covar::ForwardCovariance, conditioning_draws::Array{Symbol,T}) where T<:Real
Given some subset of known stochastic integral values this generates a conditional multivariate normal distribution.
"""
function generate_conditioned_distribution(covar::ForwardCovariance, conditioning_draws::Dict{Symbol,T}) where T<:Real
    # https://stats.stackexchange.com/questions/30588/deriving-the-conditional-distributions-of-a-multivariate-normal-distribution
    what_conditioned_on = collect(keys(conditioning_draws))
    len = length(what_conditioned_on)
    if len < 1 error("Nothing that was input was in the covariance matrix. So nothing to condition on.") end
    conditioning_indices = Array{typeof(Integer(1)),1}()
    for i in 1:len
        append!(conditioning_indices, findall(what_conditioned_on[i] .==  covar.covariance_labels_))
    end
    other_indices = setdiff(1:length(covar.covariance_labels_), conditioning_indices)
    labels = covar.covariance_labels_[other_indices]
    # Segmenting the covariance matrix.
    sigma11 = covar.covariance_[other_indices,other_indices]
    sigma12 = covar.covariance_[other_indices,conditioning_indices]
    sigma21 = covar.covariance_[conditioning_indices,other_indices]
    sigma22 = covar.covariance_[conditioning_indices,conditioning_indices]
    mu1     = zeros(length(other_indices))
    mu2     = zeros(length(conditioning_indices))
    conditioned_values = map( x -> conditioning_draws[x], what_conditioned_on)
    sigma12_invsigma22 = sigma12 * inv(sigma22)
    conditional_mu = mu1 + sigma12_invsigma22 * (conditioned_values - mu2)
    conditional_sigma = sigma11 -  sigma12_invsigma22 * sigma21
    return conditional_mu, conditional_sigma, labels
end
