module GradientFlows

using UnPack: @unpack
using OrdinaryDiffEq: solve, ODEProblem, Euler, u_modified!
using DiffEqCallbacks: PresetTimeCallback
using HCubature: hcubature
using Zygote: withgradient
using Optimisers: Leaf, Optimisers
using Flux
using LinearAlgebra, Random, LoopVectorization

import Distributions: MvNormal, Distribution, MultivariateDistribution, mean, cov, gradlogpdf, pdf
import OrdinaryDiffEq.solve

import Statistics.mean, Statistics.cov


const DEFAULT_RNG = Random.default_rng()

include("score.jl")
include("solve.jl")

include("problems/problem.jl")
include("problems/diffusion.jl")

include("solvers/solver.jl")
include("solvers/exact.jl")
include("solvers/blob.jl")
include("solvers/sbtm.jl")

include("analysis/moments.jl")
include("analysis/kde.jl")
include("analysis/Lp.jl")


export GradFlowProblem, Solver
export Exact, Blob, SBTM
export mlp
export Logger
export set_solver
export solve
export diffusion_problem

export true_dist
export emp_mean, emp_cov
export kde
export Lp_distance, Lp_error
export score

end
