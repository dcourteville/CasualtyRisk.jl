using Makie, GeoMakie, ThreadsX

export impact_map

function impact_map(mean, cov, inc)
    # Scaled impact density function (f(mean) == 1)
    dist = MvNormal(SVector{2}(mod2pi.(mean)), SMatrix{2,2}(cov))
    norm = 1 / (2π * sqrt(det(cov)))
    f = (x) -> impact_density(deg2rad.(x), inc, dist, (1,5)) / norm

    # Earth map
    fig = Figure()
    ax = GeoAxis(fig[1,1], dest = "+proj=wintri")
    lines!(ax, GeoMakie.coastlines(); color = :black)

    # Plot reentry area
    ideg = rad2deg(asin(sin(inc)))
    coord = [SA[λ,ϕ] for λ in LinRange(-180, 180, 10*360), ϕ in LinRange(-ideg, +ideg, 10*180)]
    r = ThreadsX.map(f, coord)
    cut = exp(-0.5*2^3) # Cut at 3σ
    hm = meshimage!(ax, -180 .. 180, -ideg .. +ideg, r; colorrange=(cut, 1.0), lowclip=:transparent, colormap = [:yellow, :red], alpha=0.75)
    #Colorbar(fig[:, end+1], hm)
    return fig, ax
end

function impact_map(reentry_dist)
    μr = mean(reentry_dist)
    Σr = cov(reentry_dist)
    return impact_map(μr[2:3], Σr[2:3,2:3], μr[1])
end