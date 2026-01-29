# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

StochasticIntegrals.jl is a Julia package for computing covariance matrices of Ito integrals and generating correlated random samples for Monte Carlo simulations. It builds on UnivariateFunctions.jl for representing integrand functions.

## Development Commands

```bash
# Install dependencies
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Run all tests
julia --project=. -e 'using Pkg; Pkg.test()'

# Run a specific test file
julia --project=. -e 'include("test/new_tests.jl")'

# Start REPL for development
julia --project=.

# Build documentation
julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

## Architecture

### Source File Organization

Files are numbered to indicate dependency order:
1. `1_main_functions.jl` - Core types (ItoIntegral, ItoSet, ForwardCovariance, SimpleCovariance) and covariance computation
2. `2_data_conversions.jl` - DataFrame/Array conversion utilities
3. `3_getConfidenceHypercube.jl` - Confidence region calculation
4. `4_number_generators.jl` - Abstract NumberGenerator with Mersenne, SobolGen, Stable_RNG implementations
5. `5_conditional_distributions.jl` - Conditional multivariate normal generation
6. `6_ItoProcesses.jl` - ItoProcess struct and time evolution functions

### Core Types

- **ItoIntegral**: Single stochastic integral with Brownian ID and UnivariateFunction integrand
- **ItoSet**: Container for multiple ItoIntegrals with a Brownian correlation matrix
- **ForwardCovariance**: Computes and caches covariance matrix, Cholesky decomposition, inverse, and determinant for a time period
- **SimpleCovariance**: Mutable variant for constant integrands; scales efficiently via `update!()`
- **ItoProcess**: Mutable struct for evolving processes forward in time with drift and stochastic components

### Key Design Patterns

- Operator overloading on UnivariateFunction types for mathematical composition
- Abstract types (`StochasticIntegralsCovariance`, `NumberGenerator`) enable extensibility
- Optional computation flags in ForwardCovariance to skip unused matrix operations
- Hermitian matrix specialization for covariance matrices

### Dependencies

- **UnivariateFunctions.jl**: Required for defining integrand functions (users must import this)
- **Distributions.jl**: Normal distribution operations
- **Sobol.jl**: Quasi-random number generation
- **FixedPointAcceleration.jl**: Confidence hypercube solver

## Testing

Test files mirror the functionality areas:
- `test/new_tests.jl` - Main test suite covering covariance computation and random draws
- `test/conditioning.jl` - Conditional distribution tests
- `test/test_ito_processes.jl` - Ito process evolution tests

The docs/src/index.md file contains worked examples that serve as additional documentation.
