using GradientFlows, SafeTestsets, Test, StableRNGs
using LinearAlgebra, Distributions, Zygote
using GradientFlows: LandauParams, LandauDistribution

rng = StableRNG(123)

d = 3
B = 1 / 24
params = LandauParams(d, B)

# test pdf
for t in 5.5:0.1:6.5
    K = params.K(t)
    @test K == 1 − params.C * exp(-2params.B * (d − 1) * t)
    P = ((d + 2)K − d) / (2K)
    Q = (1 − K) / (2K^2)
    dist = LandauDistribution(d, K)
    x = rand(rng, d)
    @test pdf(dist, x) ≈ pdf(MvNormal(K * I(d)), x) * (P + Q * sum(abs2, x))
    @test pdf(dist, x) ≈ (2π * K)^(-d / 2) * exp(-sum(abs2, x) / (2K)) * (P + Q * sum(abs2, x))
    @test gradlogpdf(dist, x) ≈ Zygote.gradient(x -> log(pdf(dist, x)), x)[1]
end

# test sampling
dist = LandauDistribution(d, params.K(5.5))
n = 10^4
u = rand(rng, dist, n)
@test mean(dist) == zeros(d)
@test cov(dist) == I(d)
@test emp_mean(u) ≈ mean(dist) atol = 0.05
@test emp_cov(u) ≈ cov(dist) atol = 0.05
@test Lp_error(u, dist; p=2) ≈ 0 atol = 0.05

# test that the distribution satisfies the PDE
# divergence(f, u) = tr(jacobian(f, u)[1])
# landau_pdf(t,x) = pdf(LandauDistribution(d, params.K(t)), x)

# Zygote.gradient(landau_pdf, t, x)