# Distribution S⁻²exp(-S(|x|-σ)²/σ²)
struct Rosenbluth{F} <: ContinuousMultivariateDistribution
    d::Int # dimension
    σ::F
    S::F
end

function pdf(dist::Rosenbluth, x)
    @unpack d, σ, S = dist
    return 1/(S^2) * exp(-S * (sqrt(sum(abs2, x)) - σ)^2 / σ^2)
end
logpdf(dist::Rosenbluth, x::AbstractArray{<:Real, M}) where M = -dist.S * (sqrt(sum(abs2, x)) - dist.σ)^2 / dist.σ^2
gradlogpdf(dist::Rosenbluth, x) = gradient(x_ -> logpdf(dist, x_), x)[1]

# sample |x| and the y ~ Unif(Sᵈ⁻¹) and multiply them. To sample from Sᵈ⁻¹, sample from d-dimensional Gaussian and normalize. To sample from |x|, use rejection sampling with a Gaussian prior.
function Random.rand(rng::Random.AbstractRNG, dist::Rosenbluth{F}, n::Int) where {F}
    @unpack d, σ, S = dist
    u = zeros(F, d, n)

    f(x) = 1/(S^2) * exp(-S * (abs(x) - σ)^2 / σ^2)
    prior = Normal()
    g(x) = pdf(prior, x)
    
    M = max(1/S^2.5 * σ * exp(-σ^2) * sqrt(2π), maximum(x -> f(x) / g(x), -4*σ:0.01:4*σ) + 1)
    
    for i in 1:n
        direction = F.(rand(rng, MvNormal(I(d))))
        x = rejection_sample(f, prior, M, rng)
        u[:, i] = x .* direction
    end
    return u
end
Random.rand(dist::Rosenbluth, n::Int) = rand(DEFAULT_RNG, dist, n)

mean(dist::Rosenbluth) = zeros(dist.d)
cov(dist::Rosenbluth{F}) where F = fill(F(NaN), (dist.d, dist.d))
abs_moment(::Rosenbluth{F}) where F = F(NaN)