using GradientFlows, StableRNGs, Test, Distributions
using GradientFlows: BlobAllocMemCPU
import Base.==

==(a::T, b::T) where {T<:Union{Solver,GradFlowProblem,BlobAllocMemCPU}} = all(f -> getfield(a, f) == getfield(b, f), fieldnames(T))

@testset "GradFlowProblem tests" begin
    problem = diffusion_problem(2, 10, Exact(); rng=StableRNG(123))
    @test "$problem" isa String
    @test true_dist(problem, problem.tspan[1]) isa MvNormal
    @test set_solver(problem, Blob()) == diffusion_problem(2, 10, Blob(); rng=StableRNG(123))
end