# Distribution S⁻²exp(-S(|x|-σ)²/σ²)
struct Rosenbluth{F} <: ContinuousMultivariateDistribution
    d::Int
    σ::F
    S::F
end

# sample |x| and the y ~ Unif(Sᵈ⁻¹) and multiply them. To sample from Sᵈ⁻¹, sample from d-dimensional Gaussian and normalize. To sample from |x|, use rejection sampling with a Gaussian prior.
function Random.rand(rng::Random.AbstractRNG, dist::Rosenbluth{F}, n::Int) where {F}
    d = dist.d
    S = dist.S
    σ = dist.σ
    u = zeros(F, d, n)

    f(x) = 1/(S^2) * exp(-S * (abs(x) - σ)^2 / σ^2)
    prior = Normal()
    g(x) = pdf(prior, x)
    M = max(1/S^2.5 * σ * exp(-σ^2) * sqrt(2π), [f(x)/g(x) for x in -4*σ:0.01:4*σ]...)
    println("M = $M")
    
    for i in 1:n
        direction = F.(rand(rng, MvNormal(I(d))))
        
        # rejection sample |x|
        while true
            x = rand(rng, prior)
            if M * g(x) < f(x)
                error("M = $M is too low: $(M * g(x)) = M * g(x) < f(x) = $(f(x)) for x = $x.")
            end
            if rand(rng) * M * g(x) < f(x) # accept with probability f(x)/Mg(x)
                u[:, i] = x .* direction
                break
            end
        end
    end
    return u
end
Random.rand(dist::Rosenbluth, n::Int) = rand(DEFAULT_RNG, dist, n)