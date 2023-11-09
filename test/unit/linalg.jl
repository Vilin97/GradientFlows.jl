using LinearAlgebra, Test, StableRNGs, GradientFlows

@testset "norm" begin
    rng = StableRNG(0)
    for _ in 1:10
        x, y = rand(rng, 5), rand(rng, 5)
        @test GradientFlows.norm(x, y) ≈ LinearAlgebra.norm(x .- y)
        @test GradientFlows.normsq(x, y) ≈ sum(abs2, x .- y)
    end
end