using GradientFlows, Test, StableRNGs, Distributions, CUDA

if CUDA.functional()
    d = 2
    n = 2000
    problem = diffusion_problem(d, n, Exact(); rng=rng)
    cu_problem = cu(problem)
    @test cu_problem.u0 isa CuArray
    @test solve(problem)[end] ≈ Array(solve(cu_problem)[end])
end