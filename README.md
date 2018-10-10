# StochasticIntegrals.jl

This generates covariance matrices and cholesky decompositions for a set of stochastic integrals.
At the moment it only supports Ito integrals. Users specify the UnivariateFunction that is the integrand of the ito integral and a covariance matrix will be made of all such ito integrals.

There are a large number of convenience functions. This includes finding the variance and instantaneous volatility of an ito integral; for extracting the terminal correlation & covariance of a pair of stochastic integrals over a period of time; for generation of random draws from the set of ito integrals (either pseudorandom or quasirandom). Given a draw of stochastic integrals, it is also possible to find the density of the multivariate normal distribution distribution at this point. See the testing files for code examples.