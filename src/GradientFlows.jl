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
using Distributions
using OptimalTransport: sinkhorn2
using Distances: sqeuclidean, pairwise
using Zygote: pullback 
using Flux.OneHotArrays: onehot
using Plots, Polynomials, LinearAlgebra, LaTeXStrings, Logging, Dates, LoggingExtras, Telegram, Telegram.API, ConfigEnv
using Plots: plot, savefig, scatter

import Distributions: mean, cov, gradlogpdf, logpdf, pdf, rand
import OrdinaryDiffEq.solve
import Statistics.mean, Statistics.cov

const DEFAULT_RNG = Random.default_rng()
const DEFAULT_TIMER = TimerOutput()

const SOLVER_NAME_WIDTH = 5
const PROBLEM_NAME_WIDTH = 25
const n_WIDTH = 6
const PLOT_WINDOW_SIZE = (3000, 1800)
const PLOT_SMALL_WINDOW_SIZE = (600, 400)
const PLOT_LINE_WIDTH = 4
const PLOT_MARGIN = (13, :mm)
const PLOT_FONT_SIZE = 15
const PLOT_COLOR_TRUTH = :black

include("distributions/rejection_sample.jl")
include("distributions/poly_normal.jl")
include("distributions/score.jl")
include("distributions/normal.jl")
include("distributions/mixture.jl")


include("problems/problem.jl")
include("problems/diffusion.jl")
include("problems/landau.jl")
include("problems/fpe.jl")

include("linalg.jl")
include("solve.jl")

include("solvers/solver.jl")
include("solvers/logger.jl")
include("solvers/exact.jl")
include("solvers/blob.jl")
include("solvers/sbtm.jl")
include("solvers/asbtm.jl")

include("analysis/empirical_moments.jl")
include("analysis/kde.jl")
include("analysis/Lp.jl")
include("analysis/wasserstein_distance.jl")
include("analysis/plot.jl")

include("experiments/experiment.jl")
include("experiments/experiment_result.jl")
include("io/save_load.jl")
include("io/logging.jl")
include("io/telegram_alerts.jl")

const ALL_SOLVERS = [Exact(), SBTM(), ASBTM(), Blob()]

export GradFlowProblem
export set_u0!
export diffusion_problem, fpe_problem, landau_problem
export landau_problem_factory
export Exact, Blob, SBTM, ASBTM
export Logger
export mlp, train_s!
export name
export solve

export Experiment, GradFlowExperimentResult
export run_experiments, save_results, train_nn, train_nns
export save, load, model_filename, experiment_filename, experiment_result_filename, load_metric, load_all_experiment_runs, num_runs, timer_filename, best_model
export DEFAULT_TIMER, PLOT_WINDOW_SIZE, PLOT_SMALL_WINDOW_SIZE, PLOT_LINE_WIDTH, PLOT_MARGIN, PLOT_FONT_SIZE, PLOT_COLOR_TRUTH
export ALL_SOLVERS

export true_dist, have_true_dist, true_score, pdf, marginal_pdf
export emp_mean, emp_cov, mean, cov
export kde, kde_bandwidth
export Lp_distance, Lp_error, w2
export score
export plot_metric_over_n, scatter_plot, marginal_pdf_plot, slice_pdf_plot, plot_score_error, plot_covariance_trajectory, plot_entropy_production_rate, plot_w2, plot_L2, plot_all
export @log, @trySendTelegramMessage

end
