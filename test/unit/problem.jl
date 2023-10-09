using GradientFlows, StableRNGs, Test, Distributions
using GradientFlows: BlobAllocMemCPU
import Base.==

function ==(p1::T, p2::T) where {T <: Union{Solver, GradFlowProblem, BlobAllocMemCPU}}
    for field in fieldnames(T)
        if getfield(p1, field) != getfield(p2, field)
            return false
        end
    end
    return true
end

@testset "GradFlowProblem tests" begin
    problem = diffusion_problem(2, 10, Exact(); rng = StableRNG(123))
    @test "$problem" isa String
    @test true_dist(problem, problem.tspan[1]) isa MvNormal
    @test set_solver(problem, Blob()) == diffusion_problem(2, 10, Blob(); rng = StableRNG(123))
end