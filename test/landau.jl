using GradientFlows, StableRNGs
include("testutils.jl")

n = 2000
for d in [3, 5]
    for solver in [Exact(), SBTM(mlp(d, rng=StableRNG(321), depth=2); logger=Logger(0)), Blob(0.16)]
        problem = landau_problem(d, n, solver; rng=StableRNG(123))
        test_prob(problem)
    end
end