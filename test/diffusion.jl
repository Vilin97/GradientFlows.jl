using GradientFlows, Test, StableRNGs, Distributions, Random

@time @safetestset "Diffusion Tests" begin
    d = 2
    n = 2000
    for solver in [Exact(), SBTM(mlp(d, rng=StableRNG(321)); logger=Logger(0)), Blob(0.16)]
        problem = diffusion_problem(d, n, solver; rng=StableRNG(123))
        test_prob(problem)
    end
end