function to_sigma_points(μ::T, Σ) where {T}
    L = sqrt(Σ)
    N = size(L, 2)
    s = Vector{T}(undef, 2N + 1)
    w0 = 0.5
    w = fill((1-w0) / 2N, 2N + 1)
    w[1] = w0
    s[1] = μ
    for i in 1:N
        s[1+i] = μ .+ sqrt(N) .* L[:, i]
        s[1+N+i] = μ .- sqrt(N) .* L[:, i]
    end
    return s, w
end

function from_sigma_points(s, w)
    μ = sum(i -> w[i] * s[i], eachindex(s, w))
    Σ = sum(i -> (δ = s[i] - μ; w[i] * δ * δ'), eachindex(s, w))
    Σ = (Σ + Σ') / 2
    return μ, Σ
end

function unscented_transform(f, μ::T, Σ) where {T}
    x, w = to_sigma_points(μ, Σ)
    y = Vector{T}(undef, length(x))
    Threads.@threads for i in eachindex(x, y)
        y[i] = f(x[i])
    end
    return from_sigma_points(y, w)
end

function unscented_transform(f, μ, Σ, ::Type{T}) where {T}
    x, w = to_sigma_points(μ, Σ)
    y = Vector{T}(undef, length(x))
    Threads.@threads for i in eachindex(x, y)
        y[i] = f(x[i])
    end
    return from_sigma_points(y, w)
end