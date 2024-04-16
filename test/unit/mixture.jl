using GradientFlows, Test, StableRNGs, Distributions, LinearAlgebra

@testset "Mixture model" begin
    rng = StableRNG(123)
    N = 10^6

    mm = MixtureModel(MvNormal[MvNormal([-1.,0.], I(2)), MvNormal([1., 0.], I(2))], [1/2, 1/2])
    u = rand(rng, mm, N)
    @test mean(mm) ≈ zeros(2)
    @test mean(mm) ≈ emp_mean(u) atol = 0.01

    @test cov(mm) ≈ I(2) + 0.5*([-1,0] * [-1,0]' + [1,0] * [1,0]')
    @test cov(mm) ≈ emp_cov(u) atol = 0.1
    
    
    X = randn(rng, 3, 3)
    mm = MixtureModel(MvNormal[MvNormal([1.,2.,3.], I(3)), MvNormal([3.,2.,1.], 3*I(3)), MvNormal([-4.,-5.,-6.], X'*X)], [1/3, 1/3, 1/3])
    u = rand(rng, mm, N)
    @test mean(mm) ≈ emp_mean(u) atol = 0.01
    @test cov(mm) ≈ emp_cov(u) atol = 0.1
end