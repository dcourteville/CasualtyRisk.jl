using FastGaussQuadrature, LinearAlgebra

export compute_casualty_risk

const SUM_CUT = 10

"""
Generate bivariate Gauss-Legendre quadrature node and weights
"""
function bivariate_gausslegendre(n)
    x, w = gausslegendre(n)
    return [SVector(y, z) for (y, z) in Iterators.product(x, x)], w .* w'
end

"""
Compute the probability density of impacting the given coordinates
- `ϕ`: Latitude
- `λ`: Longitude
- `i`: Inclination
- `sum_range`: Sum truncation limits
- `dist`: Impact distribution
"""
function impact_density(x, i, dist, sum_range)
    λ, ϕ = x
    sin_λ, cos_λ = sincos(λ)
    sin_i, cos_i = sincos(i)

    α0 = asin(sin(ϕ) / sin_i)
    α1 = π - α0
    sin_α0, cos_α0 = sincos(α0)
    sin_α1 = sin_α0
    cos_α1 = -cos_α0

    num = cos_α0 * sin_λ - cos_i * sin_α0 * cos_λ
    den = cos_α0 * cos_λ + cos_i * sin_α0 * sin_λ
    Λ0 = atan(num, den)
    num = cos_α1 * sin_λ - cos_i * sin_α1 * cos_λ
    den = cos_α1 * cos_λ + cos_i * sin_α1 * sin_λ
    Λ1 = atan(num, den)

    pdf_sum = 0.0
    for n in (-sum_range[1]):sum_range[1]
        for m in (-sum_range[2]):sum_range[2]
            pdf_sum += Distributions.pdf(dist, SA[Λ0+2π*n, α0+2π*m])
            pdf_sum += Distributions.pdf(dist, SA[Λ1+2π*n, α1+2π*m])
        end
    end
    return pdf_sum
end

struct DistributionProjection{T}
    base_dist::T
    i::Float64
    sum_range::Tuple{Int64,Int64}
end

function impact_density(x, dist::DistributionProjection)
    impact_density(x, dist.i, dist.base_dist, dist.sum_range)
end

"""
Compute the probability of impacting the defined cell
"""
function impact_probability(ϕmin, ϕmax, λmin, λmax, dist, nodes)
    lb = SA[λmin, ϕmin]
    ub = SA[λmax, ϕmax]
    scale = (ub .- lb) ./ 2
    shift = (lb .+ ub) ./ 2
    function fnode((x, w),)
        w * impact_density(scale .* x .+ shift, dist)
    end
    return sum(fnode, nodes) * prod(scale)
end

"""
Compute the casualty risk in the cell
- `cell`: Cell definition `((lat_min, lat_max, lon_min, lon_max), pop)`
- `ϕlim`: Maximum reachable latitude
- `dist`: Probability density function
"""
function cell_casualty_risk(cell, ϕlim, dist, algo)
    ((ϕmin, ϕmax, λmin, λmax), pop) = cell
    if pop == 0.0 || ϕmin >= ϕlim || ϕmax <= -ϕlim
        # Cell is uninhabitated or outside the reachable latitudes
        return 0.0
    else
        # Restrict latitude range to reachable values according to inclination
        ϕmin = max(ϕmin, -ϕlim)
        ϕmax = min(ϕmax, ϕlim)
        # Compute impact probability
        cell_proba = impact_probability(ϕmin, ϕmax, λmin, λmax, dist, algo)
        # Compute casualty risk
        return pop_density(cell) * cell_proba
    end
end

"""
Compute the casualty risk over all the given cells
- `cells`: List of cells, each cell is a tuple `((lat_min, lat_max, lon_min, lon_max), pop)`
- `μ`: Mean vector of the `[Λ, α]` gaussian distribution
- `Σ`: Covariance matrix of the `[Λ, α]` gaussian distribution
- `i`: Inclination
- `casualty_area`: (optional) Casualty area of surviving debris
- `n_quad`: Order of the gauss quadrature for each cell
- `σcut`: Truncation limit of the gaussian distribution
"""
function integrate_casualty_risk(pop_data, μ, Σ, i; casualty_area=1.0, n_quad=3, σcut=2)
    # Integration problem
    smean = SVector{2}(mod2pi.(μ))
    scov = SMatrix{2,2}(Σ)
    dist = MvNormal(smean, scov)
    sum_range = (
        ceil(Int, σcut * sqrt(scov[1, 1]) / π),
        ceil(Int, σcut * sqrt(scov[2, 2]) / π)
    )
    algo = zip(bivariate_gausslegendre(n_quad)...)
    map_dist = DistributionProjection(dist, i, sum_range)

    # Maximum reachable latitude
    ϕlim = asin(sin(i))

    # Integrate for each cell
    total_risk = sum(cell -> cell_casualty_risk(cell, ϕlim, map_dist, algo), pop_data)
    return casualty_area * total_risk
end