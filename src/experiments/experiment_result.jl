struct GradFlowExperimentResult{F,M1,M2}
    true_covariance::M1
    empirical_covariance::M2

    # Errors
    update_score_time::F
    true_cov_trace_error::F
    true_cov_norm_error::F
    sample_mean_error::F
    sample_cov_trace_error::F

    # Computable only if have_true_dist(experiment)
    true_mean_error::F
    true_fourth_moment_error::F
    L2_error::F
end

function GradFlowExperimentResult(experiment::Experiment)
    d = size(experiment.solution[1], 1)
    F = eltype(experiment.solution[1])
    lp_error = d <= 5 ? Lp_error(experiment; p=2) : F(NaN)
    return GradFlowExperimentResult{Float64}(
        experiment.true_cov[end],
        emp_cov(experiment.solution[end]),

        update_score_time(experiment.timer),
        true_cov_trace_error(experiment),
        true_cov_norm_error(experiment),
        sample_mean_error(experiment),
        sample_cov_trace_error(experiment),
        
        have_true_dist(experiment) ? true_mean_error(experiment) : F(NaN),
        have_true_dist(experiment) ? true_fourth_moment_error(experiment) : F(NaN),
        have_true_dist(experiment) ? lp_error : F(NaN),
    )
end

update_score_time(timer) = TimerOutputs.time(timer["update score"]) / 10^9

sample_mean_error(experiment; t_idx=length(experiment.saveat)) = norm(emp_mean(experiment.solution[t_idx]), emp_mean(experiment.solution[1]))
sample_cov_trace_error(experiment; t_idx=length(experiment.saveat)) = abs(tr(emp_cov(experiment.solution[t_idx])) - tr(emp_cov(experiment.solution[1])))

Lp_error(experiment; kwargs...) = true_metric(Lp_error, experiment; kwargs...)
true_mean_error(experiment; kwargs...) = true_metric((u, dist) -> norm(emp_mean(u), mean(dist)), experiment; kwargs...)
true_cov_norm_error(experiment; kwargs...) = true_metric((u, dist) -> norm(emp_cov(u), cov(dist)), experiment; kwargs...)
true_cov_trace_error(experiment; kwargs...) = true_metric((u, dist) -> abs(tr(emp_cov(u)) - tr(cov(dist))), experiment; kwargs...)
true_fourth_moment_error(experiment; kwargs...) = true_metric((u, dist) -> abs(emp_abs_moment(u, 4) - abs_moment(dist, 4)), experiment; kwargs...)

function true_metric(metric, experiment; t_idx=length(experiment.saveat), kwargs...)
    @unpack solution = experiment
    dist = experiment.true_dist[t_idx]
    return metric(solution[t_idx], dist; kwargs...)
end

function Base.show(io::IO, result::GradFlowExperimentResult)
    @unpack update_score_time, L2_error, true_mean_error, true_cov_trace_error, true_cov_norm_error, true_fourth_moment_error, sample_mean_error, sample_cov_trace_error = result
    println(io, "$(rpad("L2 error:", 25)) $(round(L2_error,digits=4))")
    println(io, "$(rpad("true mean error:", 25)) $(round(true_mean_error,digits=4))")
    println(io, "$(rpad("true cov trace error:", 25)) $(round(true_cov_trace_error,digits=4))")
    println(io, "$(rpad("true cov norm error:", 25)) $(round(true_cov_norm_error,digits=4))")
    println(io, "$(rpad("true fourth moment error:", 25)) $(round(true_fourth_moment_error,digits=4))")
    println(io, "$(rpad("sample mean error:", 25)) $(round(sample_mean_error,digits=4))")
    println(io, "$(rpad("sample cov trace error:", 25)) $(round(sample_cov_trace_error,digits=4))")
    println(io, "$(rpad("update score time:", 25)) $(round(update_score_time,digits=4))")
end

have_true_dist(experiment) = !isnothing(experiment.true_mean_error)