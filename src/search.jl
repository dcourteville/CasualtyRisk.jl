export compute_reentry_distibution, compute_casualty_risk
export mc_reentry_simulation, mc_casualty_risk

"""
Compute reentry distribution
"""
function compute_reentry_distibution(model, X, Σ)
    return unscented_transform(Xi -> reentry_shooting(model, Xi), X, Σ, similar_type(X, Size(5)))
end

"""
Compute total casualty risk
"""
function compute_casualty_risk(model, X, Σ, pop_data, args...; kwargs...)
    μr, Σr = compute_reentry_distibution(model, X, Σ)
    return integrate_casualty_risk(pop_data, μr[2:3], Σr[2:3,2:3], μr[1], args...; kwargs...)
end

"""
Perform a Monte-Carlo simulation of the reentry and return the set of reentry coordinates
"""
function mc_reentry_simulation(model, X, Σ; nsamples=100000)
    dist = MvNormal(X, Σ)
    N = length(X)
    points = ThreadsX.map(Xi -> reentry_shooting(model, SVector{N}(Xi)), eachcol(rand(dist, nsamples)))
    return points
end

"""
Compute the casualty risk using Monte-Carlo simulations
"""
function mc_casualty_risk(points, pop_data)
    return mean(points) do xr
        geo = reentry2geo(xr)
        cell = find_pop_cell(pop_data, geo)
        return pop_density(cell)
    end
end

function mc_casualty_risk(model, X, Σ, pop_data; kwargs...)
    points = mc_reentry_simulation(model, X, Σ; kwargs...)
    return mc_casualty_risk(points, pop_data)
end

function mapΛ(model, X, Σ, pop_data, Λgrid=0:359, args...; kwargs...)
    Rgrid = ThreadsX.map(Λgrid) do Λ
        XΛ = @set X[5] = deg2rad(Λ)
        return compute_casualty_risk(model, XΛ, Σ, pop_data, args...; kwargs...)
    end
    return Rgrid
end