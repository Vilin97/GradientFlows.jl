function w2(u1, u2; ε, maxiter=10^4, tol=1e-2)
    @assert size(u1, 1) == size(u2, 1)
    C = pairwise(sqeuclidean, u1, u2; dims=2)
    μ = fill(1 / size(u1, 2), size(u1, 2))
    ν = fill(1 / size(u2, 2), size(u2, 2))
    return sinkhorn2(μ, ν, C, ε; maxiter=maxiter, tol=tol)
end

function w2(u, dist::Distribution; sample_size=10^4, kwargs...)
    u2 = rand(dist, sample_size)
    return w2(u, u2; kwargs...)
end