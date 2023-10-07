using GradientFlows, Test, StableRNGs, Distributions, CUDA
using GradientFlows: normsq

function test_prob(problem; debug=false, time=false)
    solution = solve(problem; saveat=problem.tspan[2])
    u = solution[end]
    end_dist = true_dist(problem, problem.tspan[2])
    if debug
        @show problem
        @show normsq(emp_mean(u))
        @show normsq(emp_cov(u), cov(end_dist))
        @show Lp_error(u, end_dist; p=2)
    end
    if time
        @time solve(problem; saveat=problem.tspan[2])
    end
    @test emp_mean(u) ≈ mean(end_dist) atol = 0.05
    @test emp_cov(u) ≈ cov(end_dist) rtol = 0.1
    @test Lp_error(u, end_dist; p=2) ≈ 0 atol = 0.05
    @test "$problem" isa String # just to increase coverage
end

d = 2
n = 2000
for solver in [Exact(), Blob(0.16)]
    rng = StableRNG(123)
    problem = diffusion_problem(d, n, solver; rng=rng)
    test_prob(problem)
end
