using GradientFlows, Test

function test_prob(problem; p=2, mean_atol=0.05, cov_atol=1., Lp_atol=0.05)
    u = solve(problem; saveat=problem.tspan[2])[end]
    end_dist = true_dist(problem, problem.tspan[2])

    @test emp_mean(u) ≈ mean(end_dist) atol = mean_atol
    @test emp_cov(u) ≈ cov(end_dist) atol = cov_atol
    @test Lp_error(u, end_dist; p=p) ≈ 0 atol = Lp_atol
end

function debug_prob(problem)
    @time solution = solve(problem; saveat=problem.tspan[2])
    u = solution[end]
    end_dist = true_dist(problem, problem.tspan[2])

    @show emp_mean(u)
    @show emp_cov(u), cov(end_dist)
    @show Lp_error(u, end_dist; p=2)
end