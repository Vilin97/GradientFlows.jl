using GradientFlows, StableRNGs
include("testutils.jl")

d = 3
n = 2000
# TODO: resolve the instability
for solver in [Exact(), SBTM(mlp(d, rng=StableRNG(321), depth=2)), Blob(0.16)]
    problem = landau_problem(d, n, solver; rng=StableRNG(123))
    test_prob(problem)
end