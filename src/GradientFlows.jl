module GradientFlows

using UnPack
using OrdinaryDiffEq, DiffEqCallbacks
using LinearAlgebra, Distributions, Random
import OrdinaryDiffEq.solve
using CUDA

import Statistics.mean, Statistics.cov
using HCubature

const DEFAULT_RNG = Random.default_rng()

include("score.jl")
include("problem.jl")
include("solver.jl")
include("solve.jl")

include("analysis/linalg.jl")
include("analysis/moments.jl")
include("analysis/kde.jl")
include("analysis/Lp.jl")


export GradFlowProblem, Solver
export Exact
export solve
export update!
export diffusion_problem

export true_dist
export emp_mean, emp_cov
export kde
export Lp_distance, Lp_error

end
