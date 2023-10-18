struct GradFlowExperiment{P, V, S, T}
	problem :: P
    saveat :: V
    # TODO: remove solutions from here and add them to GradFlowExperimentSolution
	solutions :: Vector{S} # solutions[run][time_index][d, n] isa Float
    timer :: T
end

"Solve `problem` `num_solutions` times with different u0."
function GradFlowExperiment(problem::GradFlowProblem, saveat, num_solutions :: Int)
    solutions = Vector{Vector{typeof(problem.u0)}}(undef, num_solutions)
    reset_timer!()
    @timeit "$(problem.name) $(problem.solver)" for i in 1:num_solutions
        resample!(problem; rng=StableRNG(i))
        sol = solve(problem, saveat=saveat)
        solutions[i] = sol.u
    end
    return GradFlowExperiment(problem, saveat, solutions, TimerOutputs.get_defaulttimer())
end

function Lp_error(experiment; kwargs...)
    return avg_metric(Lp_error, experiment; kwargs...)
end
function mean_norm_error(experiment; kwargs...)
    return avg_metric((u,dist) -> sqrt(normsq(emp_mean(u), mean(dist))), experiment; kwargs...)
end
function cov_norm_error(experiment; kwargs...)
    return avg_metric((u,dist) -> sqrt(normsq(emp_cov(u), cov(dist))), experiment; kwargs...)
end

function avg_metric(error, experiment::GradFlowExperiment; t_idx = length(experiment.saveat), kwargs...)
    @unpack problem, saveat, solutions = experiment
    dist = true_dist(problem, saveat[t_idx])
    return mean([error(sol[t_idx], dist; kwargs...) for sol in solutions])
end
    
struct GradFlowExperimentSet{S,N,E,T}
    solvers::S
    num_particles::N
    experiments::Matrix{E} # experiments[num_particles, solver]
    Lp_errors::Matrix{T}
    mean_norm_errors::Matrix{T}
    cov_norm_errors::Matrix{T}
end

# TODO: solve!(experiment_set::GradFlowExperimentSet; kwargs...) = solve!.(experiment_set.experiments; kwargs...)
# TODO: analyze!(experiment_set::GradFlowExperimentSet; kwargs...) = # compute metrics

Lp_error(experiment_set::GradFlowExperimentSet; kwargs...) = Lp_error.(experiment_set.experiments; kwargs...)
mean_norm_error(experiment_set::GradFlowExperimentSet; kwargs...) = mean_norm_error.(experiment_set.experiments; kwargs...)
cov_norm_error(experiment_set::GradFlowExperimentSet; kwargs...) = cov_norm_error.(experiment_set.experiments; kwargs...)