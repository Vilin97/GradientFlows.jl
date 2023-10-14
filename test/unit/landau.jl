using GradientFlows, SafeTestsets, Test, StableRNGs 
using LinearAlgebra, Distributions, Zygote
using GradientFlows: LandauParams, LandauDistribution

rng = StableRNG(123)

d = 3
B = 1/24
t = 5.5
params = LandauParams(3, B)
K = params.K(t)
@test K == 1 − params.C * exp(-2params.B*(d − 1)*t)

P = ((d + 2)K − d) / (2K)
Q = (1 − K)/(2K^2)

# test pdf
dist = LandauDistribution(d, K)
x = rand(rng, d)
@test pdf(dist, x) ≈ pdf(MvNormal(K * I(d)), x) * (P + Q * sum(abs2, x))
@test gradlogpdf(dist, x) ≈ Zygote.gradient(x -> log(pdf(dist, x)), x)[1]

# test sampling
n = 10^4
u = rand(rng, dist, n)
@test emp_mean(u) ≈ mean(dist) atol = 0.05
@test emp_cov(u) ≈ cov(dist) atol = 0.05
@test Lp_error(u, dist; p=2) ≈ 0 atol = 0.05


