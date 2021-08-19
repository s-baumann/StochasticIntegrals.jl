# StochasticIntegrals.jl

| Build | Coverage | Documentation |
|-------|----------|---------------|
| [![Build status](https://github.com/s-baumann/StochasticIntegrals.jl/workflows/CI/badge.svg)](https://github.com/s-baumann/StochasticIntegrals.jl/actions) | [![codecov](https://codecov.io/gh/s-baumann/StochasticIntegrals.jl/branch/master/graph/badge.svg?token=al5s5iJTeL)](https://codecov.io/gh/s-baumann/StochasticIntegrals.jl) | [![docs-latest-img](https://img.shields.io/badge/docs-latest-blue.svg)](https://s-baumann.github.io/StochasticIntegrals.jl/dev/index.html) |

This generates covariance matrices and Cholesky decompositions for a set of stochastic integrals.
At the moment it only supports Ito integrals. Users specify the [UnivariateFunction](https://github.com/s-baumann/UnivariateFunctions.jl) that is the integrand of the Ito integral and a covariance matrix will be made of all such Ito integrals.

There are a large number of convenience functions. This includes finding the variance and instantaneous volatility of an Ito integral; for extracting the terminal correlation & covariance of a pair of stochastic integrals over a period of time; for generation of random draws from the set of Ito integrals (either pseudorandom or quasirandom). Given a draw of stochastic integrals, it is also possible to find the density of the multivariate normal distribution at this point. See the testing files for code examples.
