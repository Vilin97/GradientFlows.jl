struct GradFlowExperimentResult{F}
    update_score_time::F
    L2_error::F
	true_mean_error::F
	true_cov_trace_error::F
	true_cov_norm_error::F
	true_fourth_moment_error::F
	sample_mean_error::F
	sample_cov_trace_error::F
end

function GradFlowExperimentResult(experiment :: GradFlowExperiment)
    return GradFlowExperimentResult(
        update_score_time(experiment.timer),
        Lp_error(experiment; p=2),
        true_mean_error(experiment),
        true_cov_trace_error(experiment),
        true_cov_norm_error(experiment),
        true_fourth_moment_error(experiment),
        sample_mean_error(experiment),
        sample_cov_trace_error(experiment)
    )
end

function update_score_time(timer)
    # the key that does not contain "Lp"
    key = first(filter(x -> !occursin("Lp", x), keys(timer.inner_timers)))
    return TimerOutputs.time(timer[key].inner_timers["update score"]) / 10^9
end

sample_mean_error(experiment; t_idx=length(experiment.saveat)) = norm(emp_mean(experiment.solution[t_idx]), emp_mean(experiment.solution[1]))
sample_cov_trace_error(experiment; t_idx=length(experiment.saveat)) = abs(tr(emp_cov(experiment.solution[t_idx])) - tr(emp_cov(experiment.solution[1])))

Lp_error(experiment; kwargs...) = true_metric(Lp_error, experiment; kwargs...)
true_mean_error(experiment; kwargs...) = true_metric((u, dist) -> norm(emp_mean(u), mean(dist)), experiment; kwargs...)
true_cov_norm_error(experiment; kwargs...) = true_metric((u, dist) -> norm(emp_cov(u), cov(dist)), experiment; kwargs...)
true_cov_trace_error(experiment; kwargs...) = true_metric((u, dist) -> abs(tr(emp_cov(u)) - tr(cov(dist))), experiment; kwargs...)
true_fourth_moment_error(experiment; kwargs...) = true_metric((u, dist) -> abs(emp_abs_moment(u, 4) - abs_moment(dist, 4)), experiment; kwargs...)

function true_metric(metric, experiment; t_idx=length(experiment.saveat), kwargs...)
    @unpack problem, saveat, solution = experiment
    dist = true_dist(problem, saveat[t_idx])
    return metric(solution[t_idx], dist; kwargs...)
end
