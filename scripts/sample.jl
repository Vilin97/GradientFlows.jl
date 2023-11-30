using AdvancedHMC, ForwardDiff
using LogDensityProblems
using LinearAlgebra
using AbstractMCMC, LogDensityProblemsAD

struct LogTargetDensity
    dim::Int
end
LogDensityProblems.logdensity(p::LogTargetDensity, θ) = -sum(abs2, θ) / 2  # standard multivariate normal
LogDensityProblems.dimension(p::LogTargetDensity) = p.dim
LogDensityProblems.capabilities(::Type{LogTargetDensity}) = LogDensityProblems.LogDensityOrder{0}()

D = 3; initial_θ = rand(D)
ℓπ = LogTargetDensity(D)

model = AdvancedHMC.LogDensityModel(LogDensityProblemsAD.ADgradient(Val(:ForwardDiff), ℓπ))

n_samples, n_adapts, δ = 10_000, 2_000, 0.8
sampler = NUTS(δ)

# Now sample
samples = AbstractMCMC.sample(
      model,
      sampler,
      n_adapts + n_samples;
      nadapts = n_adapts,
      initial_params = initial_θ,
  )

# Extract the samples
x_samples = [sample.z.θ for sample in samples]

using Plots
histogram([x[1] for x in x_samples]; normalize=true);
plot!(x -> (2π)^(-1/2) * exp(-x^2/2))










using Distributions
# Distribution (2π * K)^(-d / 2) * exp(-x² / (2K)) * (P + Q * x²)
struct PolyNormal{F} <: ContinuousMultivariateDistribution
    d::Int
    K::F
end

get_P(dist::PolyNormal) = ((dist.d + 2) * dist.K - dist.d) / (2dist.K)
get_Q(dist::PolyNormal) = (1 - dist.K) / (2dist.K^2)

function pdf(dist::PolyNormal, x::AbstractVector)
    K, d = dist.K, dist.d
    P, Q = get_P(dist), get_Q(dist)
    x² = sum(abs2, x)
    return (2π * K)^(-d / 2) * exp(-x² / (2K)) * (P + Q * x²)
end

LogDensityProblems.logdensity(p::PolyNormal, x) = log(pdf(p, x))
LogDensityProblems.dimension(p::PolyNormal) = p.d
LogDensityProblems.capabilities(::Type{PolyNormal}) = LogDensityProblems.LogDensityOrder{0}()

D = 3; initial_θ = rand(D)
K = 0.6
ρ = PolyNormal(D, K)

model = AdvancedHMC.LogDensityModel(LogDensityProblemsAD.ADgradient(Val(:ForwardDiff), ρ))

n_samples, n_adapts, δ = 10_000, 2_000, 0.8
sampler = NUTS(δ)

# Now sample
samples = AbstractMCMC.sample(
      model,
      sampler,
      n_adapts + n_samples;
      nadapts = n_adapts,
      initial_params = initial_θ,
  )

# Extract the samples
x_samples = [sample.z.θ for sample in samples]


function marginal_pdf(dist::PolyNormal, x::Number)
    K, d = dist.K, dist.d
    P, Q = get_P(dist), get_Q(dist)
    return (2π * K)^(-1 / 2) * exp(-x^2 / (2K)) * (P + Q * x^2 + (d - 1) * Q * K)
end
using Plots
histogram([x[1] for x in x_samples]; normalize=true);
plot!(x -> marginal_pdf(ρ, x))



function LogDensityModel(logpdf, D)
    struct LogTargetDensity
        dim::Int
    end
    LogDensityProblems.logdensity(p::LogTargetDensity, θ) = logpdf(θ)
    LogDensityProblems.dimension(p::LogTargetDensity) = p.dim
    LogDensityProblems.capabilities(::Type{LogTargetDensity}) = LogDensityProblems.LogDensityOrder{0}()
    
    model = AdvancedHMC.LogDensityModel(LogDensityProblemsAD.ADgradient(Val(:ForwardDiff), LogTargetDensity(D))) 
end