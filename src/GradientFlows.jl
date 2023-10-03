module GradientFlows

using OrdinaryDiffEq, DiffEqCallbacks
using LinearAlgebra, Distributions, Random
import OrdinaryDiffEq.solve
import Statistics.mean, Statistics.cov

const DEFAULT_RNG = Random.default_rng()

include("score.jl")
include("problem.jl")
include("solver.jl")
include("solve.jl")

include("analysis/moments.jl")

export GradFlowProblem, Solver
export Exact
export solve
export update!
export diffusion_problem

export true_dist
export emp_mean, emp_cov

end
