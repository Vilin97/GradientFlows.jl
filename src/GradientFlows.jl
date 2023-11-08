module GradientFlows

using UnPack: @unpack
using OrdinaryDiffEq: solve, ODEProblem, Euler, u_modified!
using DiffEqCallbacks: PresetTimeCallback
using HCubature: hcubature
using Zygote: withgradient
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
const PLOT_WINDOW_SIZE = (2400, 1800)

include("distributions/poly_normal.jl")
include("distributions/score.jl")
include("distributions/normal.jl")

include("linalg.jl")

include("problems/problem.jl")
include("problems/diffusion.jl")
include("problems/landau.jl")

include("solve.jl")

include("solvers/solver.jl")
include("solvers/exact.jl")
include("solvers/blob.jl")
include("solvers/sbtm.jl")

include("analysis/empirical_moments.jl")
include("analysis/kde.jl")
include("analysis/Lp.jl")

include("experiments/experiment.jl")
include("experiments/io.jl")

export GradFlowProblem
export Exact, Blob, SBTM
export mlp, blob_epsilon
export Logger
export set_u0!
export solve
export diffusion_problem, landau_problem

export GradFlowExperiment, GradFlowExperimentResult
export solve!, compute_errors!
export save, load, model_filename, experiment_filename, timer_filename, best_model
export train_s!
export DEFAULT_TIMER, PLOT_WINDOW_SIZE

export true_dist, pdf, marginal_pdf
export emp_mean, emp_cov, mean, cov
export kde, kde_epsilon
export Lp_distance, Lp_error
export score

end
