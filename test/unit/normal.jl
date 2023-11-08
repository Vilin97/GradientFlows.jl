using GradientFlows, Test, StableRNGs
using LinearAlgebra, Distributions
using GradientFlows: abs_moment, emp_abs_moment
using HCubature: hcubature

@testset "MvNormal distribution" begin
    @testset "moments" begin
        rng = StableRNG(123)
        d = 3
        dist = MvNormal(I(d))
        n = 10^4
        u = rand(rng, dist, n)
        @test emp_mean(u) ≈ mean(dist) atol = 0.05
        @test emp_cov(u) ≈ cov(dist) atol = 0.05
        @test Lp_error(u, dist; p=2) ≈ 0 atol = 0.05
        @test abs_moment(dist, 2) ≈ emp_abs_moment(u, 2) rtol = 0.05
        @test abs_moment(dist, 4) ≈ emp_abs_moment(u, 4) rtol = 0.05
    end

    @testset "marginal" begin
        dist = MvNormal(I(2))
        result, error = hcubature(x -> marginal_pdf(dist, x[1]), [-10], [10])
        @test result ≈ 1 atol = 2 * error
    end
end