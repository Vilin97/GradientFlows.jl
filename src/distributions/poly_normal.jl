# Distribution (2π * K)^(-d / 2) * exp(-x² / (2K)) * (P + Q * x²)
struct PolyNormal{F} <: ContinuousMultivariateDistribution
    d::Int
    K::F
    function PolyNormal(d, K)
        dist = new{typeof(K)}(d, K)
        if get_P(dist) < 0
            error("P = $(get_P(dist)) < 0 for d = $d, K = $K. Make sure K > d/(d+2).")
        end
        if get_Q(dist) < 0
            error("Q = $(get_Q(dist)) < 0 for d = $d, K = $K. Make sure K < 1.")
        end
        return dist
    end
end

get_P(dist::PolyNormal) = ((dist.d + 2) * dist.K - dist.d) / (2dist.K)
get_Q(dist::PolyNormal) = (1 - dist.K) / (2dist.K^2)

function pdf(dist::PolyNormal, x::AbstractVector)
    @unpack K, d = dist
    P, Q = get_P(dist), get_Q(dist)
    x² = sum(abs2, x)
    return (2π * K)^(-d / 2) * exp(-x² / (2K)) * (P + Q * x²)
end

function marginal_pdf(dist::PolyNormal, x::Number)
    @unpack K, d = dist
    P, Q = get_P(dist), get_Q(dist)
    return (2π * K)^(-1 / 2) * exp(-x^2 / (2K)) * (P + Q * x^2 + (d - 1) * Q * K)
end

function Random.rand(dist::PolyNormal, n::Int)
    return rand(DEFAULT_RNG, dist, n)
end

function Random.rand(rng::Random.AbstractRNG, dist::PolyNormal, n::Int)
    d = dist.d
    K = dist.K
    β = 1.5
    proposal = MvNormal(K * I(d) * β)
    xs = [[x, zeros(typeof(K), d - 1)...] for x in 0:0.01:5]
    M = maximum(x -> pdf(dist, x) / pdf(proposal, x), xs) + 1
    u = zeros(typeof(K), d, n)
    for i in 1:n
        u[:, i] = rejection_sample(dist, proposal, M, rng)
    end
    return u
end

function gradlogpdf(dist::PolyNormal, x)
    K = dist.K
    P, Q = get_P(dist), get_Q(dist)
    return x .* (-1 / K + 2Q / (P + Q * sum(abs2, x)))
end

mean(dist::PolyNormal) = zeros(dist.d)
cov(dist::PolyNormal) = I(dist.d)

"E |X|^k"
function abs_moment(dist::PolyNormal, k::Int)
    @unpack K, d = dist
    P, Q = get_P(dist), get_Q(dist)
    if k == 2
        return d
    elseif k == 4
        return P * K^2 * (d^2 + 2d) + Q * K^3 * d * (d + 2) * (d + 4)
    else
        error("abs_moment(dist, k) is not implemented for k = $k.")
    end
end