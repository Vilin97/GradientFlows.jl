using GradientFlows, Test, StableRNGs, Distributions, CUDA

d = 2
n = 5000
problem = diffusion_problem(d, n, Exact(); rng=rng)
cu_problem = cu(problem)
if CUDA.functional()
    @test cu_problem.u0 isa CuArray
else
    @test cu_problem.u0 isa Array
end