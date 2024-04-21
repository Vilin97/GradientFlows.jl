# Distribution S⁻²exp(-S(|x|-σ)²/σ²)
struct Rosenbluth{F} <: ContinuousMultivariateDistribution
    d::Int
    σ::F
    S::F
end

# sample |x| and the y ~ Unif(Sᵈ⁻¹) and multiply them. To sample from Sᵈ⁻¹, sample from d-dimensional Gaussian and normalize.
function Random.rand(rng::Random.AbstractRNG, dist::Rosenbluth{F}, n::Int) where {F}
    u = zeros(F, d, n)
    for i in 1:n
        direction = rand(rng, MvNormal(I(F, d)))

        # rejection sample |x|
        f(x) = 1/(dist.S^2) * exp(-dist.S * (abs(x) - dist.σ)^2 / dist.σ^2)
        M = 7
        prior = Normal()
        while true
            x = rand(rng, prior)
            if M * pdf(prior, x) < pdf(dist, x, direction)
                error("M = $M is too low: $(M * pdf(prior, x)) = M * pdf(prior, x) < pdf(dist, x, direction) = $(pdf(dist, x, direction)) for x = $x.")
            end
            if rand(rng) * M * pdf(prior, x) < f(x) # accept with probability f(x)/Mg(x)
                u[:, i] = x * direction
                break
            end
        end
    end
    return u
end
