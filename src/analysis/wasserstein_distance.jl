function w2(u1, u2; ε=0.5, maxiter=10^5)
    @assert size(u1, 1) == size(u2, 1)
    C = pairwise(sqeuclidean, u1, u2; dims=size(u1, 1))
    μ = fill(1 / size(u1, 2), size(u1, 2))
    ν = fill(1 / size(u2, 2), size(u2, 2))
    return sinkhorn2(μ, ν, C, ε; maxiter=maxiter)
end

function w2(u, dist::Distribution; sample_size=10^4, kwargs...)
    u2 = rand(dist, sample_size)
    return w2(u, u2; kwargs...)
end