struct ItoProcess{R<:Real}
  t0::R
  value::R
  drift::UnivariateFunction
  stochastic::ItoIntegral
end

function evolve(itoprocess::ItoProcess, stochastic::Real, new_time::Real)
  old_const = itoprocess.value
  realised_drift = evaluate_integral(itoprocess.drift,itoprocess.t0, new_time)
  new_const = old_const + realised_drift + stochastic
  return ItoProcess(new_time, new_const, itoprocess.drift, itoprocess.stochastic)
end

function evolve(itoprocesses::Dict{Symbol,ItoProcess{R}}, stochastics::Dict{Symbol,R}, new_time::Real) where R<:Real
  for k in keys(itoprocesses)
    itoprocesses[k] = evolve(itoprocesses[k], stochastics[k], new_time)
  end
  return itoprocesses
end

function evolve(itoprocesses::Dict{Symbol,ItoProcess{R}}, stochastics::Union{Dict{Symbol,R},Dict{Symbol,Real}}, new_time::Real) where R<:Real
  for k in keys(itoprocesses)
    itoprocesses[k] = evolve(itoprocesses[k], stochastics[k], new_time)
  end
  return itoprocesses
end

function evolve_covar_and_ito_processes(itoprocesses::Dict{Symbol,ItoProcess{R}}, covar::ForwardCovariance, new_time::Real; number_generator::NumberGenerator, recalculate_all::Bool = true) where R<:Real
  new_covar = ForwardCovariance(covar, covar.to_, new_time; recalculate_all = recalculate_all)
  stochastic_draws = get_draws(new_covar, 1; number_generator = number_generator)[1]
  evolved_itos = evolve(itoprocesses, stochastic_draws, new_time)
  return evolved_itos, new_covar
end

function make_ito_process_syncronous_time_series(ito_processes::Dict{Symbol,ItoProcess{R}},
                                                     covar::ForwardCovariance, timegap::Real, total_number_of_ticks::Integer; ito_twister = MersenneTwister(2)) where R<:Real
    assets = collect(keys(ito_processes))
    d1 = DataFrame(Time = Array{R}([]),Name = Array{Symbol}([]),Value = Array{R}([]))
    at_time = covar.to_
    number_generator = Mersenne(ito_twister, length(assets))
    for i in 1:total_number_of_ticks
      next_tick = at_time + timegap
      ito_processes, covar = evolve_covar_and_ito_processes(ito_processes, covar, next_tick; recalculate_all = false, number_generator = number_generator)
      for stock in assets
        d2 = Dict([:Time, :Name, :Value] .=> [next_tick, stock, ito_processes[stock].value])
        d1 = push!(d1,d2)
      end
      at_time = next_tick
    end
    return d1
end

function make_ito_process_non_syncronous_time_series(ito_processes::Dict{Symbol,ItoProcess{R}},
                                                     covar::ForwardCovariance, update_rates::Union{OrderedDict{Symbol,D},Dict{Symbol,D}},
                                                     total_number_of_ticks::Integer; timing_twister::MersenneTwister = MersenneTwister(1), ito_twister = MersenneTwister(2)) where R<:Real where D<:Distribution
    update_rates = OrderedDict(update_rates)
    assets = collect(keys(ito_processes))
    d1 = DataFrame(Time = Array{R}([]),Name = Array{Symbol}([]),Value = Array{R}([]))
    at_time = covar.to_
    number_generator = Mersenne(ito_twister, length(assets))
    for i in 1:total_number_of_ticks
      starts = vcat(rand.(Ref(timing_twister), values(update_rates), 1)...)
      next_tick = at_time + minimum(starts)
      ito_processes, covar = evolve_covar_and_ito_processes(ito_processes, covar, next_tick; recalculate_all = false, number_generator = number_generator)
      # Moving everything up to the tick.
      what_stock = collect(keys(update_rates))[findall(abs.(starts .- minimum(starts)) .< 1e-15)[1]]
      d2 = Dict([:Time, :Name, :Value] .=> [next_tick, what_stock, ito_processes[what_stock].value])
      d1 = push!(d1,d2)
      at_time = next_tick
    end
    return d1
end


function make_ito_process_non_syncronous_time_series_wide(ito_processes::Dict{Symbol,ItoProcess{R}},
                                                     covar::ForwardCovariance, update_rates::Union{OrderedDict{Symbol,D},Dict{Symbol,D}},
                                                     total_number_of_ticks::Integer; timing_twister::MersenneTwister = MersenneTwister(1), ito_twister = MersenneTwister(2)) where R<:Real where D<:Distribution
    update_rates = OrderedDict(update_rates)
    assets = collect(keys(ito_processes))
    d1 = DataFrame(Time = Array{R}([]))
    for c in assets
        d1[!,c] = Array{Union{Missing,R}}([])
    end
    at_time = covar.to_
    number_generator = Mersenne(ito_twister, length(assets))
    for i in 1:total_number_of_ticks
      starts = vcat(rand.(Ref(timing_twister), values(update_rates), 1)...)
      next_tick = at_time + minimum(starts)
      ito_processes, covar = evolve_covar_and_ito_processes(ito_processes, covar, next_tick; recalculate_all = false, number_generator = number_generator)
      # Moving everything up to the tick.
      what_stock = collect(keys(update_rates))[findall(abs.(starts .- minimum(starts)) .< 1e-15)[1]]
      d2 = Dict([:Time, what_stock, setdiff(assets, [what_stock])...] .=> [next_tick, ito_processes[what_stock].value, repeat([missing], length(assets)-1)...])
      d1 = push!(d1,d2)
      at_time = next_tick
    end
    return d1
end
