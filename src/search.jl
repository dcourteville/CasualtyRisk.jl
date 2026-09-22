export compute_reentry_distibution, compute_casualty_risk
export mc_reentry_simulation, mc_casualty_risk

"""
Compute reentry distribution
"""
function compute_reentry_distibution(model, X, Σ)
    μr, Σr = unscented_transform(Xi -> reentry_shooting(model, Xi), X, Σ, similar_type(X, Size(5)))
    return MvNormal(μr, Σr)
end

"""
Compute total casualty risk
"""
function compute_casualty_risk(model, X, Σ, pop_data; σcut=2, casualty_area=1.0, n_quad=3)
    reentry_dist = compute_reentry_distibution(model, X, Σ)
    proj_dist = ProjectedDistribution(reentry_dist; σcut)
    return integrate_casualty_risk(proj_dist, pop_data; casualty_area, n_quad)
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

function map_longitude_shifts(model, X, Σ, pop_data, shifts=0:359; kwargs...)
    Rgrid = ThreadsX.map(shifts) do Λ
        # Shifting time instead of RAAN is more accurate
        # It preserves the same orientation relative to the Sun
        #XΛ = @set X[5] = X[5] + mod2pi(deg2rad(Λ)) # Shift RAAN
        XΛ = @set X[8] = X[8] - mod2pi(deg2rad(Λ)) / EARTH_ANGULAR_SPEED # Shift time
        return compute_casualty_risk(model, XΛ, Σ, pop_data; kwargs...)
    end
    return Rgrid
end

function map_longitude_shifts_fast(model, X, Σ, pop_data; σcut=2, casualty_area=1.0, n_quad=3)
    reentry_dist = compute_reentry_distibution(model, X, Σ)
    proj_dist = ProjectedDistribution(reentry_dist; σcut)
    grid_dist = ProbabilityGrid(proj_dist, pop_data; n_quad)

    shifts = (0:(size(grid_dist.grid, 2)-1))
    return map(
        shift -> integrate_casualty_risk(grid_dist, pop_data, shift; casualty_area),
        shifts
    )
end