```@meta
CurrentModule = StochasticIntegrals
```

# Internal Functions

```@index
Pages = ["api.md"]
```

### Main Structs

```@docs
ItoIntegral
ItoSet
ForwardCovariance
SimpleCovariance
```

### Random Number Generation

```@docs
NumberGenerator
Mersenne
SobolGen
next!
get_draws
get_zero_draws
```

### Ito Processes

```@docs
ItoProcess
evolve
evolve_covar_and_ito_processes
make_ito_process_syncronous_time_series
make_ito_process_non_syncronous_time_series
```

### Helper Functions

```@docs
volatility
variance
covariance
correlation
pdf
make_covariance
log_likelihood
brownians_in_use
update!
```

## Internal Functions

```@docs
to_draws
to_dataframe
to_array
get_confidence_hypercube
generate_conditioned_distribution
```
