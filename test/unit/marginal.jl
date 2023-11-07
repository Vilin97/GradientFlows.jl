using GradientFlows, Test, LinearAlgebra, Distributions
using HCubature: hcubature
using GradientFlows: PolyNormal

@testset "Marginal pdf" begin
    dist = MvNormal(I(2))
    result, error = hcubature(x -> marginal_pdf(dist, x[1]), [-10], [10])
    @test result ≈ 1 atol = 2 * error

    dist = PolyNormal(5, 1 / 5)
    result, error = hcubature(x -> marginal_pdf(dist, x[1]), [-10], [10])
    @test result ≈ 1 atol = 2 * error
end