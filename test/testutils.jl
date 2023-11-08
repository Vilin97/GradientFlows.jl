using GradientFlows, Test
using Statistics: mean, cov
using GradientFlows: BlobAllocMemCPU, Solver
import Base.==

==(a::T, b::T) where {T<:Union{Solver,GradFlowProblem,BlobAllocMemCPU}} = all(f -> getfield(a, f) == getfield(b, f), fieldnames(T))

function test_prob(problem; p=2, mean_atol=0.05, cov_atol=1.0, Lp_atol=0.05)
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

    @show Lp_error(u, end_dist; p=2)
    @show emp_mean(u)
    @show emp_cov(u)
    @show cov(end_dist)
end