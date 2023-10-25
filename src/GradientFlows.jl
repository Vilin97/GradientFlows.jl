module GradientFlows

using UnPack: @unpack
using OrdinaryDiffEq: solve, ODEProblem, Euler, u_modified!
using DiffEqCallbacks: PresetTimeCallback
using HCubature: hcubature
using Zygote: withgradient
using Optimisers: Leaf, Optimisers
using Flux
using LinearAlgebra, Random, LoopVectorization
using TimerOutputs
using JLD2

import Distributions: MvNormal, Distribution, MultivariateDistribution, mean, cov, gradlogpdf, pdf, rand, ContinuousMultivariateDistribution
import OrdinaryDiffEq.solve
import Statistics.mean, Statistics.cov


const DEFAULT_RNG = Random.default_rng()
const DEFAULT_TIMER = TimerOutput()

const SOLVER_NAME_WIDTH = 5
const PROBLEM_NAME_WIDTH = 9
const n_WIDTH = 6

include("linalg.jl")

include("problems/problem.jl")
include("problems/diffusion.jl")
include("problems/landau.jl")

include("score.jl")
include("solve.jl")

include("solvers/solver.jl")
include("solvers/exact.jl")
include("solvers/blob.jl")
include("solvers/sbtm.jl")

include("analysis/moments.jl")
include("analysis/kde.jl")
include("analysis/Lp.jl")

include("experiments/experiment.jl")
include("experiments/io.jl")

export GradFlowProblem, Solver
export Exact, Blob, SBTM
export mlp, blob_eps
export Logger
export set_solver, resample!
export solve
export diffusion_problem, landau_problem

export GradFlowExperiment
export solve!, compute_errors!
export run_experiment_set!
export DEFAULT_TIMER
export save, load, model_filename, best_model, experiment_filename

export true_dist
export emp_mean, emp_cov
export kde
export Lp_distance, Lp_error
export score

end
