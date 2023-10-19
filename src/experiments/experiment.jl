mutable struct GradFlowExperiment{P, V, S, F}
	problem :: P
    saveat :: V
    num_solutions :: Int
	solutions :: Vector{S} # solutions[run][time_index][d, n] isa Float
    L2_error :: F
    mean_norm_error :: F
    cov_norm_error :: F
    cov_trace_error :: F
end

short_string(float, digits, width) = rpad(round(float, digits=digits), width)

function Base.show(io::IO, experiment::GradFlowExperiment)
    @unpack problem, saveat, num_solutions, solutions, L2_error, mean_norm_error, cov_norm_error, cov_trace_error = experiment
    digits = 3
    width = 5
    d,n = size(problem.u0)
    print(io, "\n$(problem.name)(d=$d,n=$(rpad(n,6))) $(rpad(problem.solver, 5)) $(num_solutions) runs: |ρ∗ϕ - ρ*|₂ = $(short_string(L2_error,digits,width)) |E(ρ)-E(ρ*)|₂ = $(short_string(mean_norm_error,digits,width)) |Σ-Σ'|₂ = $(short_string(cov_norm_error,digits,width)) |tr(Σ-Σ')| = $(short_string(cov_trace_error,digits,width))")
end

"Solve `problem` `num_solutions` times with different u0."
function GradFlowExperiment(problem::GradFlowProblem, num_solutions :: Int; saveat = problem.tspan[2])
    solutions = Vector{Vector{typeof(problem.u0)}}(undef, 0)
    F = eltype(problem.u0)
    return GradFlowExperiment(problem, saveat, num_solutions, solutions, zero(F), zero(F), zero(F), zero(F))
end

function solve!(experiment::GradFlowExperiment)
    @unpack problem, saveat, num_solutions, solutions = experiment
    d,n = size(problem.u0)
    for _ in 1:num_solutions
        resample!(problem)
        @timeit DEFAULT_TIMER "d=$d n=$(rpad(n,6)) $(rpad(problem.name, 10)) $(problem.solver)" sol = solve(problem, saveat=saveat)
        push!(solutions, sol.u)
    end
    # TODO: if save_to_file, save(experiment) 
    nothing
end

function compute_errors!(experiment::GradFlowExperiment)
    d,n = size(experiment.problem.u0)
    @timeit DEFAULT_TIMER "d=$d n=$(rpad(n,6)) Lp" experiment.L2_error = Lp_error(experiment;p=2)
    experiment.mean_norm_error = mean_norm_error(experiment)
    experiment.cov_norm_error = cov_norm_error(experiment)
    experiment.cov_trace_error = cov_trace_error(experiment)
    nothing
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
function cov_trace_error(experiment; kwargs...)
    return avg_metric((u,dist) -> abs(tr(emp_cov(u) .- cov(dist))), experiment; kwargs...)
end

function avg_metric(error, experiment::GradFlowExperiment; t_idx = length(experiment.saveat), kwargs...)
    @unpack problem, saveat, solutions = experiment
    dist = true_dist(problem, saveat[t_idx])
    return mean([error(sol[t_idx], dist; kwargs...) for sol in solutions])
end
    
struct GradFlowExperimentSet{E}
    experiments::E # a collection of `GradFlowExperiment`s
end

function GradFlowExperimentSet(problems, num_solutions; kwargs...)
    experiments = [GradFlowExperiment(problem, num_solutions; kwargs...) for problem in problems]
    return GradFlowExperimentSet(experiments)
end

function run_experiment_set!(experiment_set)
    reset_timer!(DEFAULT_TIMER)
    solve!.(experiment_set.experiments)
    show(DEFAULT_TIMER)

    compute_errors!.(experiment_set.experiments)
    show(DEFAULT_TIMER)
end

Base.show(io::IO, experiment_set::GradFlowExperimentSet) = Base.show(io, experiment_set.experiments)