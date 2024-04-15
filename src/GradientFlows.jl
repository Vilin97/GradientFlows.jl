module GradientFlows

using UnPack: @unpack
using OrdinaryDiffEq: solve, ODEProblem, Euler, u_modified!
using DiffEqCallbacks: PresetTimeCallback
using HCubature: hcubature
using Zygote: withgradient, gradient
using Flux
using LinearAlgebra, Random, LoopVectorization
using TimerOutputs
using JLD2
using StableRNGs: StableRNG

import Distributions: MvNormal, Distribution, MultivariateDistribution, mean, cov, gradlogpdf, logpdf, pdf, rand, ContinuousMultivariateDistribution, MixtureModel
import OrdinaryDiffEq.solve
import Statistics.mean, Statistics.cov

const DEFAULT_RNG = Random.default_rng()
const DEFAULT_TIMER = TimerOutput()

const SOLVER_NAME_WIDTH = 5
const PROBLEM_NAME_WIDTH = 18
const n_WIDTH = 6
const PLOT_WINDOW_SIZE = (3000, 1800)
const PLOT_LINE_WIDTH = 4
const PLOT_MARGIN = (13, :mm)

include("distributions/poly_normal.jl")
include("distributions/score.jl")
include("distributions/normal.jl")
include("distributions/mixture.jl")

include("linalg.jl")

include("problems/problem.jl")
include("problems/diffusion.jl")
include("problems/landau.jl")
include("problems/fpe.jl")

include("solve.jl")

include("solvers/solver.jl")
include("solvers/logger.jl")
include("solvers/exact.jl")
include("solvers/blob.jl")
include("solvers/sbtm.jl")

include("analysis/empirical_moments.jl")
include("analysis/kde.jl")
include("analysis/Lp.jl")

include("experiments/experiment.jl")
include("experiments/experiment_result.jl")
include("experiments/io.jl")

const ALL_PROBLEMS = [(diffusion_problem, 2), (diffusion_problem, 5), (diffusion_problem, 10), (fpe_problem, 2), (fpe_problem, 5), (fpe_problem, 10), (landau_problem, 3), (landau_problem, 5), (landau_problem, 10), (anisotropic_landau_problem, 3), (anisotropic_landau_problem, 5), (anisotropic_landau_problem, 10)]
const ALL_SOLVERS = [Exact(), SBTM(), Blob()]

export GradFlowProblem
export set_u0!
export diffusion_problem, fpe_problem, landau_problem, anisotropic_landau_problem, coulomb_landau_normal_problem, coulomb_landau_mixture_problem
export Exact, Blob, SBTM
export Logger
export mlp, train_s!
export name
export solve

export Experiment, GradFlowExperimentResult
export run_experiments, save_results, train_nn
export save, load, model_filename, experiment_filename, experiment_result_filename, load_metric, load_all_experiment_runs, timer_filename, best_model
export DEFAULT_TIMER, PLOT_WINDOW_SIZE, PLOT_LINE_WIDTH, PLOT_MARGIN
export ALL_PROBLEMS, ALL_SOLVERS

export true_dist, have_true_dist, true_score, pdf, marginal_pdf
export emp_mean, emp_cov, mean, cov
export kde, kde_bandwidth
export Lp_distance, Lp_error
export score

end
