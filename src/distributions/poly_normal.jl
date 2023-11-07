# Distribution (2π * K)^(-d / 2) * exp(-x² / (2K)) * (P + Q * x²)
struct PolyNormal{F} <: ContinuousMultivariateDistribution
    d::Int
    K::F
end

function pdf(dist::PolyNormal, x::AbstractVector)
    @unpack K, d = dist
    P = ((d + 2) * K - d) / (2K)
    Q = (1 - K) / (2K^2)
    x² = sum(abs2, x)
    return (2π * K)^(-d / 2) * exp(-x² / (2K)) * (P + Q * x²)
end

function marginal_pdf(dist::PolyNormal, x::Number)
    @unpack K, d = dist
    P = ((d + 2) * K - d) / (2K)
    Q = (1 - K) / (2K^2)
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

function rejection_sample(target_dist, proposal_dist, M, rng=DEFAULT_RNG)
    f(x) = pdf(target_dist, x)
    g(x) = pdf(proposal_dist, x)
    while true
        x = rand(rng, proposal_dist)
        if M * g(x) < f(x)
            error("M = $M is too low: $(M*g(x)) = Mg(x) < f(x) = $(f(x)) for x = $x.")
        end
        if rand(rng) * M * g(x) < f(x) # accept with probability f(x)/Mg(x)
            return x
        end
    end
end

function gradlogpdf(dist::PolyNormal, x)
    K = dist.K
    d = dist.d
    P = ((d + 2) * K - d) / (2K)
    Q = (1 - K) / (2K^2)
    return x .* (-1 / K + 2Q / (P + Q * sum(abs2, x)))
end

mean(dist::PolyNormal) = zeros(dist.d)
cov(dist::PolyNormal) = I(dist.d)
