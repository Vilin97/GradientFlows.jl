using GradientFlows, Test, StableRNGs
using HCubature: hcubature

@testset "kde" begin
    rng = StableRNG(123)
    d = 2
    n = 100
    u = randn(rng, d, n)
    result, error = hcubature(x -> kde(x, u), fill(-10, d), fill(10, d))
    @test result ≈ 1 atol = 4 * error
end