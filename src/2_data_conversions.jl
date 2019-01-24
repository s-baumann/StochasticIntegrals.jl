
# From draw
function draws_to_array(draws::Array{Dict{Symbol,Float64},1}; labels::Array{Symbol,1}  = collect(keys(draws[1])))
    X = Array{Float64,2}(undef,length(draws), length(labels))
    for i in 1:length(draws)
        for j in 1:length(labels)
            X[i,j] = draws[i][labels[j]]
        end
    end
    return X, labels
end
function draws_to_dataframe(draws::Array{Dict{Symbol,Float64},1}; labels::Array{Symbol,1}  = collect(keys(draws[1])))
    dd = DataFrame()
    X = Array{Float64,2}(undef,length(draws), length(labels))
    for lab in labels
        q = Array{Float64,1}(undef, length(draws))
        for j in 1:length(draws)
            q[j] = draws[j][lab]
        end
        dd[lab] = q
    end
    return dd
end

# From Array
function array_to_draws(X::Array{Float64,2}; labels::Array{Symbol,1} = Symbol.("x", 1:size(X)[2]))
    draws = Array{Dict{Symbol,Float64},1}()
    for i in 1:size(X)[1]
        draw = Dict{Symbol,Float64}()
        for j in 1:length(labels)
            draw[labels[j]] = X[i,j]
        end
        draws = vcat(draws, draw)
    end
    return draws
end
function array_to_dataframe(X::Array{Float64,2}; labels::Array{Symbol,1} = Symbol.("x", 1:size(X)[2]))
    dd = DataFrame()
        for j in 1:length(labels)
            dd[labels[j]] = X[:,j]
        end
    return dd
end

# From dataframe
function dataframe_to_draws(dd::DataFrame; labels::Array{Symbol,1} = names(dd))
    draws = Array{Dict{Symbol,Float64},1}()
    for i in 1:size(dd)[1]
        draw = Dict{Symbol,Float64}()
        for j in labels
            draw[j] = dd[j][i]
        end
        draws = vcat(draws, draw)
    end
    return draws
end
function dataframe_to_array(dd::DataFrame; labels::Array{Symbol,1} = names(dd))
    X = Array{Float64,2}(undef,size(dd)[1], length(labels))
    for c in 1:length(labels)
        X[:,c] = dd[labels[c]]
    end
    return X, labels
end
