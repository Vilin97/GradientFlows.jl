using GradientFlows, Test, StableRNGs
using LinearAlgebra, Distributions, Zygote
using GradientFlows: LandauParams, PolyNormal, t0, abs_moment, emp_abs_moment

@testest "PolyNormal distribution" begin
    rng = StableRNG(123)

    d = 3
    B = 1 / 24
    params = LandauParams(d, B)

    @test t0(params) == 5.5

    @tetset "pdf" begin
        for t in 5.5:0.1:6.5
            K = params.K(t)
            @test K == 1 − params.C * exp(-2params.B * (d − 1) * t)
            P = ((d + 2)K − d) / (2K)
            Q = (1 − K) / (2K^2)
            dist = PolyNormal(d, K)
            x = rand(rng, d)
            @test pdf(dist, x) ≈ pdf(MvNormal(K * I(d)), x) * (P + Q * sum(abs2, x))
            @test pdf(dist, x) ≈ (2π * K)^(-d / 2) * exp(-sum(abs2, x) / (2K)) * (P + Q * sum(abs2, x))
            @test gradlogpdf(dist, x) ≈ Zygote.gradient(x -> log(pdf(dist, x)), x)[1]
        end
    end

    @testset "sampling and moments" begin
        dist = PolyNormal(d, params.K(5.5))
        n = 10^4
        u = rand(rng, dist, n)
        @test mean(dist) == zeros(d)
        @test cov(dist) == I(d)
        @test emp_mean(u) ≈ mean(dist) atol = 0.05
        @test emp_cov(u) ≈ cov(dist) atol = 0.05
        @test Lp_error(u, dist; p=2) ≈ 0 atol = 0.05
        @test abs_moment(dist, 2) ≈ emp_abs_moment(u, 2) rtol = 0.05
        @test abs_moment(dist, 4) ≈ emp_abs_moment(u, 4) rtol = 0.05
    end

    @testset "marginal" begin
        dist = PolyNormal(5, 1 / 5)
        result, error = hcubature(x -> marginal_pdf(dist, x[1]), [-10], [10])
        @test result ≈ 1 atol = 2 * error
    end
end