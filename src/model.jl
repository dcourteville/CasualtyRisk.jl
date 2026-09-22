using OrbitalElementModels
using OrdinaryDiffEqTsit5

export ReentryModel, nb_revolutions

function apsalt2mce(ha, hp, ω, i, Ω, ν, body)
    # Convert to keplerian
    a = (ha + hp + 2*body.ae) / 2
    e = (ha - hp) / 2a
    # Convert to MCE
    return kep2mce(a, e, i, ω, Ω, ν)
end

function mce2reentry(x, t, body)
    p, ex, ey, i, Ω, α = x
    Λ = Ω - sideral_time(t, body)

    # velocity
    sα, cα = sincos(α)
    v = sqrt(body.μ / p) * SA[ex*sα-ey*cα, 1+ex*cα+ey*sα]
    return SA[i, Λ, α, v[1], v[2]]
end

function reentry2geo(x)
    i, Λ, α = x
    si, ci = sincos(i)
    sΛ, cΛ = sincos(Λ)
    sα, cα = sincos(α)
    ϕ = asin(si*sα)
    λ = atan(sΛ*cα + cΛ*ci*sα, cΛ*cα - sΛ*ci*sα)
    return SA[ϕ, λ]
end

struct ReentryModel{E,B,Z,D}
    elem::E
    body::B
    zonal::Z
    drag::D
end

function state_dynamics(model::ReentryModel, x, p, t)
    B, b = gauss(x, model.body, model.elem)
    az = model.zonal(x, t, model.body, model.elem)
    ad = model.drag(x, t, model.body, model.elem)

    dx = B * (az + ad * (1 + p[1])) + b
    return vcat(dx, 0.0)
end

function reentry_shooting(model::ReentryModel, X)
    N = 7
    x0 = convert_orbit(apsalt2mce, X[SOneTo(N)], model.body)
    t0 = X[N+1]
    p = SA[X[N+2]]

    hstop = 80e3
    tmax = t0 + 20*86400
    stop_callback = ContinuousCallback(
        (x,t,integrator) -> geodetic_altitude(x, model.body, model.elem) - hstop,
        integrator -> terminate!(integrator),
        interp_points=0
    )
    problem = ODEProblem{false,SciMLBase.FullSpecialize}(
        (x, p, t) -> state_dynamics(model, x, p, t),
        x0, (t0, tmax), p,
        callback=stop_callback)
    sol = solve(problem, Tsit5();
        abstol=1e-11,
        reltol=1e-11
    )

    tf = sol.t[end]
    @assert tf < tmax "Propagation did not reach stop condition before time limit"
    xf = sol.u[end]
    return mce2reentry(xf, tf, model.body)
end

function nb_revolutions(model, X)
    x0 = convert_orbit(apsalt2mce, X[SOneTo(7)], model.body)
    xr = reentry_shooting(model, X)
    return (xr[3] - x0[6]) / 2π
end