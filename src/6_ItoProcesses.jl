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

function evolve(itoprocesses::Dict{Symbol,ItoProcess{R}}, stochastics::Union{Dict{Symbol,R},Dict{Symbol,Real}}, new_time::Real) where R<:Real
  for k in keys(itoprocesses)
    itoprocesses[k] = evolve(itoprocesses[k], stochastics[k], new_time)
  end
  return itoprocesses
end

function evolve_covar_and_ito_processes(itoprocesses::Dict{Symbol,ItoProcess{R}}, covar::ForwardCovariance, new_time::Real; number_generator::NumberGenerator) where R<:Real
  if new_time - covar.to_ < 1000*eps()
    stochastic_draws  = Dict{Symbol,Real}(covar.covariance_labels_ .=> Float64.(zeros(length(covar.covariance_labels_))))
    evolved_itos = evolve(itoprocesses, stochastic_draws, new_time)
    return evolved_itos, covar
  end
  new_covar = ForwardCovariance(covar, covar.to_, new_time; recalculate_all = true)
  stochastic_draws = get_draws(new_covar, 1; number_generator = number_generator)[1]
  evolved_itos = evolve(itoprocesses, stochastic_draws, new_time)
  return evolved_itos, new_covar
end

function evolve_covar_and_ito_processes(itoprocesses::Dict{Symbol,ItoProcess{R}}, covar::SimpleCovariance, new_time::Real; number_generator::NumberGenerator) where R<:Real
  if new_time - covar.to_ < 1000*eps()
    update!(covar, covar.from_, new_time) # Note that as newtime and to are about the same I am just trying to adjust the from time here.
    stochastic_draws  = Dict{Symbol,Real}(covar.covariance_labels_ .=> Float64.(zeros(length(covar.covariance_labels_))))
    evolved_itos = evolve(itoprocesses, stochastic_draws, new_time)
    return evolved_itos, covar
  end
  update!(covar, covar.to_, new_time)
  stochastic_draws = get_draws(covar, 1; number_generator = number_generator)[1]
  evolved_itos = evolve(itoprocesses, stochastic_draws, new_time)
  return evolved_itos, covar
end



function make_ito_process_syncronous_time_series(ito_processes::Dict{Symbol,ItoProcess{R}},
                                                     covar::Union{ForwardCovariance,SimpleCovariance}, timegap::Real, total_number_of_ticks::Integer; ito_twister = MersenneTwister(2)) where R<:Real
    assets = collect(keys(ito_processes))
    d1 = DataFrame(Time = Array{R}([]),Name = Array{Symbol}([]),Value = Array{R}([]))
    at_time = covar.to_
    number_generator = Mersenne(ito_twister, length(assets))
    for i in 1:total_number_of_ticks
      next_tick = at_time + timegap
      ito_processes, covar = evolve_covar_and_ito_processes(ito_processes, covar, next_tick; number_generator = number_generator)
      for stock in assets
        d2 = Dict([:Time, :Name, :Value] .=> [next_tick, stock, ito_processes[stock].value])
        d1 = push!(d1,d2)
      end
      at_time = next_tick
    end
    return d1
end

function make_ito_process_non_syncronous_time_series(ito_processes::Dict{Symbol,ItoProcess{R}},
                                                     covar::Union{ForwardCovariance,SimpleCovariance}, update_rates::Union{OrderedDict{Symbol,D},Dict{Symbol,D}},
                                                     total_number_of_ticks::Integer; timing_twister::MersenneTwister = MersenneTwister(1), ito_twister = MersenneTwister(2)) where R<:Real where D<:Distribution
    update_rates = OrderedDict(update_rates)
    assets = collect(keys(ito_processes))
    d1 = DataFrame(Time = Array{R}([]),Name = Array{Symbol}([]),Value = Array{R}([]))
    at_time = covar.to_
    number_generator = Mersenne(ito_twister, length(assets))
    for i in 1:total_number_of_ticks
      starts = vcat(rand.(Ref(timing_twister), values(update_rates), 1)...)
      next_tick = at_time + minimum(starts)

      #oldito = deepcopy(ito_processes)
      #old_covar = deepcopy(covar)

      ito_processes, covar = evolve_covar_and_ito_processes(ito_processes, covar, next_tick; number_generator = number_generator)

      #vals = map( a -> ito_processes[a].value, assets )
      #if any(isnan.(vals))
      #  bson(string("C:\\Dropbox\\Julia_Library\\debug\\make_ito_process_non_syncronous_time_series_not_good.bson"),
      #       Dict{String,Any}(["ito_processes", "covar", "update_rates", "total_number_of_ticks", "timing_twister", "starts", "next_tick", "oldito", "old_covar", "d1", "i", "assets", "at_time", "number_generator" ] .=>
      #              [ito_processes, covar, update_rates, total_number_of_ticks, timing_twister, starts, next_tick, oldito, old_covar, d1, i, assets, at_time, number_generator ]))
      #  error("make_ito_process_non_syncronous_time_series_not_good")
      #end

      # Moving everything up to the tick.
      what_stock = collect(keys(update_rates))[findall(abs.(starts .- minimum(starts)) .< 1e-15)[1]]
      d2 = Dict([:Time, :Name, :Value] .=> [next_tick, what_stock, ito_processes[what_stock].value])
      d1 = push!(d1,d2)
      at_time = next_tick
    end
    return d1
end
