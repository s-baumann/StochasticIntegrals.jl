"""
A struct representing an Itoprocess.
### Members
* `t0` - The initial time of the `ItoProcess`
* `value` - The initial value of the `ItoProcess`
* `drift` - The drift of the `ItoProcess`.
* `stochastic` - The stochastic process.
"""
mutable struct ItoProcess{R<:Real}
  t0::R
  value::R
  drift::UnivariateFunction
  stochastic::ItoIntegral
end

"""
    evolve!(itoprocess::ItoProcess, stochastic::Real, new_time::Real)
Evolve the `ItoProcess` forward. This changes the time (`t0`) as well as the `value`.
### Inputs
* `itoprocess` - The `ItoProcess` to be evolved.
* `stochastic` - A draw from the `ItoIntegral`.
* `new_time` - The new time.
### Return
Nothing. It changes the input `ItoProcess`
"""
function evolve!(itoprocess::ItoProcess, stochastic::Real, new_time::Real)
  old_const = itoprocess.value
  realised_drift = evaluate_integral(itoprocess.drift,itoprocess.t0, new_time)
  itoprocess.value = old_const + realised_drift + stochastic
  itoprocess.t0 = new_time
end

"""
    evolve!(itoprocesses::Union{Dict{Symbol,ItoProcess{R}},Dict{Symbol,ItoProcess}}, stochastics::Union{Dict{Symbol,R},Dict{Symbol,Real}}, new_time::Real) where R<:Real
Evolve the `ItoProcess` forward. This changes the time (`t0`) as well as the `value`.
### Inputs
* `itoprocesses` - A `Dict` of `ItoProcess`es
* `stochastic` - A `Dict` of draws from the `ItoIntegral`.
* `new_time` - The new time.
### Return
Nothing. It changes the input `ItoProcess`
"""
function evolve!(itoprocesses::Union{Dict{Symbol,ItoProcess{R}},Dict{Symbol,ItoProcess}}, stochastics::Union{Dict{Symbol,R},Dict{Symbol,Real}}, new_time::Real) where R<:Real
  for k in keys(itoprocesses)
    evolve!(itoprocesses[k], stochastics[k], new_time)
  end
end

"""
    evolve_covar_and_ito_processes!(itoprocesses::Union{Dict{Symbol,ItoProcess{R}},Dict{Symbol,ItoProcess}}, covar::ForwardCovariance, new_time::Real; number_generator::NumberGenerator) where R<:Real
Evolve the `ItoProcess`es forward.
### Inputs
* `itoprocesses` - A `Dict` of `ItoProcess`es
* `covar` - The covariance matrix.
* `new_time` - The new time.
* `number_generator` - The number generator
### Return
* The updated `ItoProcess`es
* The Covariance matrix
"""
function evolve_covar_and_ito_processes!(itoprocesses::Union{Dict{Symbol,ItoProcess{R}},Dict{Symbol,ItoProcess}}, covar::ForwardCovariance, new_time::Real; number_generator::NumberGenerator) where R<:Real
  if new_time - covar.to_ < 1000*eps()
    stochastic_draws  = Dict{Symbol,Real}(covar.covariance_labels_ .=> zeros(length(covar.covariance_labels_)))
    evolve!(itoprocesses, stochastic_draws, new_time)
    return itoprocesses, covar
  end
  new_covar = ForwardCovariance(covar, covar.to_, new_time)
  stochastic_draws = get_draws(new_covar; uniform_draw = next!(number_generator))
  evolve!(itoprocesses, stochastic_draws, new_time)
  return itoprocesses, new_covar
end

"""
    evolve_covar_and_ito_processes!(itoprocesses::Union{Dict{Symbol,ItoProcess{R}},Dict{Symbol,ItoProcess}}, covar::SimpleCovariance, new_time::Real; number_generator::NumberGenerator) where R<:Real
Evolve the `ItoProcess`es forward.
### Inputs
* `itoprocesses` - A `Dict` of `ItoProcess`es
* `covar` - The covariance matrix.
* `new_time` - The new time.
* `number_generator` - The number generator
### Return
* The updated `ItoProcess`es
* The Covariance matrix
"""
function evolve_covar_and_ito_processes!(itoprocesses::Union{Dict{Symbol,ItoProcess{R}},Dict{Symbol,ItoProcess}}, covar::SimpleCovariance, new_time::Real; number_generator::NumberGenerator) where R<:Real
  if new_time - covar.to_ < 1000*eps()
    update!(covar, covar.from_, new_time) # Note that as newtime and to are about the same I am just trying to adjust the from time here.
    stochastic_draws  = Dict{Symbol,Real}(covar.covariance_labels_ .=> zeros(length(covar.covariance_labels_)))
    evolve!(itoprocesses, stochastic_draws, new_time)
    return itoprocesses, covar
  end
  update!(covar, covar.to_, new_time)
  stochastic_draws = get_draws(covar; uniform_draw = next!(number_generator))
  evolve!(itoprocesses, stochastic_draws, new_time)
  return itoprocesses, covar
end

"""
    make_ito_process_syncronous_time_series(ito_processes::Union{Dict{Symbol,ItoProcess{T}},Dict{Symbol,ItoProcess}},
                                                     covar::Union{ForwardCovariance,SimpleCovariance}, timegap::R, total_number_of_ticks::Integer;
                                                     number_generator::NumberGenerator = Mersenne(MersenneTwister(2), length(collect(keys(ito_processes))))) where R<:Real where T<:Real

Evolve the `ItoProcess`es forward.
### Inputs
* `itoprocesses` - A `Dict` of `ItoProcess`es
* `covar` - The covariance matrix.
* `timegap` - The time gap between ticks.
* `total_number_of_ticks` - The total number of ticks.
* `number_generator` The `NumberGenerator` used for the RNG.
### Return
* A DataFrame with the ticks.
"""
function make_ito_process_syncronous_time_series(ito_processes::Union{Dict{Symbol,ItoProcess{T}},Dict{Symbol,ItoProcess}},
                                                     covar::Union{ForwardCovariance,SimpleCovariance}, timegap::R, total_number_of_ticks::Integer;
                                                     number_generator::NumberGenerator = Mersenne(MersenneTwister(2), length(ito_processes))) where R<:Real where T<:Real
    assets = collect(keys(ito_processes))
    num_assets = length(assets)
    total_rows = total_number_of_ticks * num_assets
    times_col = Vector{R}(undef, total_rows)
    names_col = Vector{Symbol}(undef, total_rows)
    values_col = Vector{R}(undef, total_rows)
    at_time = covar.to_
    row = 0
    for i in 1:total_number_of_ticks
      next_tick = at_time + timegap
      ito_processes, covar = evolve_covar_and_ito_processes!(ito_processes, covar, next_tick; number_generator = number_generator)
      for stock in assets
        row += 1
        times_col[row] = next_tick
        names_col[row] = stock
        values_col[row] = ito_processes[stock].value
      end
      at_time = next_tick
    end
    return DataFrame(Time = times_col, Name = names_col, Value = values_col)
end

"""
    make_ito_process_non_syncronous_time_series(ito_processes::Union{Dict{Symbol,ItoProcess{R}},Dict{Symbol,ItoProcess}},
                                                     covar::Union{ForwardCovariance,SimpleCovariance}, update_rates::Union{OrderedDict{Symbol,D},Dict{Symbol,D},OrderedDict{Symbol,Distribution},Dict{Symbol,Distribution}},
                                                     total_number_of_ticks::Integer;
                                                     timing_twister::Union{StableRNG,MersenneTwister} = MersenneTwister(1),
                                                     ito_number_generator::NumberGenerator = Mersenne(MersenneTwister(2), length(collect(keys(ito_processes))))
                                                     ) where R<:Real where D<:Distribution
Evolve the `ItoProcess`es forward.
### Inputs
* `itoprocesses` - A `Dict` of `ItoProcess`es
* `covar` - The covariance matrix.
* `update_rates` - The update rates of the exponential waiting times between ticks for each asset.
* `total_number_of_ticks` - The total number of ticks.
* `ito_twister` The `MersenneTwister` used for the RNG.
### Return
* A DataFrame with the ticks.
"""
function make_ito_process_non_syncronous_time_series(ito_processes::Union{Dict{Symbol,ItoProcess{R}},Dict{Symbol,ItoProcess}},
                                                     covar::Union{ForwardCovariance,SimpleCovariance}, update_rates::Union{OrderedDict{Symbol,D},Dict{Symbol,D},OrderedDict{Symbol,Distribution},Dict{Symbol,Distribution}},
                                                     total_number_of_ticks::Integer;
                                                     timing_twister::Union{StableRNG,MersenneTwister} = MersenneTwister(1),
                                                     ito_number_generator::NumberGenerator = Mersenne(MersenneTwister(2), length(ito_processes))
                                                     ) where R<:Real where D<:Distribution
    update_rates = OrderedDict(update_rates)
    assets = collect(keys(ito_processes))
    times_col = Vector{R}(undef, total_number_of_ticks)
    names_col = Vector{Symbol}(undef, total_number_of_ticks)
    values_col = Vector{R}(undef, total_number_of_ticks)
    update_keys = collect(keys(update_rates))
    update_dists = collect(values(update_rates))
    at_time = covar.to_
    for i in 1:total_number_of_ticks
      starts = vcat(rand.(Ref(timing_twister), update_dists, 1)...)
      next_tick = at_time + minimum(starts)
      ito_processes, covar = evolve_covar_and_ito_processes!(ito_processes, covar, next_tick; number_generator = ito_number_generator)
      # Moving everything up to the tick.
      what_stock = update_keys[argmin(starts)]
      times_col[i] = next_tick
      names_col[i] = what_stock
      values_col[i] = ito_processes[what_stock].value
      at_time = next_tick
    end
    return DataFrame(Time = times_col, Name = names_col, Value = values_col)
end
