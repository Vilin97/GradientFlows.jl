using GradientFlows, Test, StableRNGs, Distributions, LinearAlgebra

rng = StableRNG(123)

@testset "Lp_error tests" begin
    dist = MvNormal(I(2))
    true_pdf(x) = pdf(dist, x)
    u = rand(rng, dist, 10^4)
    @test Lp_error(u, true_pdf; p=2) ≈ 0 atol = 0.1
    @test Lp_error(u, true_pdf; p=2) == Lp_error(u, dist; p=2)
end