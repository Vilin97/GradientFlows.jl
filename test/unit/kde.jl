using GradientFlows, Test
using HCubature: hcubature

@testset "kde" begin
    d = 2
    n = 100
    u = randn(d, n)
    result, error = hcubature(x -> kde(x, u), fill(-10, d), fill(10, d))
    @test result ≈ 1 atol=2*error
end