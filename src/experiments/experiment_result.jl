struct GradFlowExperimentResult{F}
    # Errors
    update_score_time::F
    top_eigenvalue_error::F
    bottom_eigenvalue_error::F
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
    F = Float64
    lp_error = d <= 5 && have_true_dist(experiment) ? Lp_error(experiment; p=2) : F(NaN)
    return GradFlowExperimentResult{Pair{F, String}}(
        update_score_time(experiment.timer) => "update score time, s",
        top_eigenvalue_error(experiment) => "λ₁ - λ₁*",
        bottom_eigenvalue_error(experiment) => "λₖ - λₖ*",
        true_cov_trace_error(experiment) => "E |Xₜ|² - E |Xₜ*|²",
        true_cov_norm_error(experiment) => "|Cov(Xₜ) - Cov(Xₜ*)|₂",
        sample_mean_error(experiment) => "|E(Xₜ) - E(X₀)|₂",
        sample_cov_trace_error(experiment) => "E |Xₜ|² - E |X₀|²",
        
        (have_true_dist(experiment) ? true_mean_error(experiment) : F(NaN)) => "|E(Xₜ)-E(Xₜ*)|₂",
        (have_true_dist(experiment) ? true_fourth_moment_error(experiment) : F(NaN)) => "E |Xₜ|⁴ - E |Xₜ*|⁴",
        lp_error => "|ρₜ∗ϕ - ρₜ*|₂",
    )
end

update_score_time(timer) = TimerOutputs.time(timer["update score"]) / 10^9

top_eigenvalue_error(experiment; t_idx=length(experiment.saveat)) = maximum(eigvals(emp_cov(experiment.solution[t_idx]))) - maximum(eigvals(experiment.true_cov[t_idx]))
bottom_eigenvalue_error(experiment; t_idx=length(experiment.saveat)) = minimum(eigvals(emp_cov(experiment.solution[t_idx]))) - minimum(eigvals(experiment.true_cov[t_idx]))

sample_mean_error(experiment; t_idx=length(experiment.saveat)) = norm(emp_mean(experiment.solution[t_idx]), emp_mean(experiment.solution[1]))
sample_cov_trace_error(experiment; t_idx=length(experiment.saveat)) = tr(emp_cov(experiment.solution[t_idx])) - tr(emp_cov(experiment.solution[1]))

true_cov_norm_error(experiment; t_idx=length(experiment.saveat)) = norm(emp_cov(experiment.solution[t_idx]), experiment.true_cov[t_idx])
true_cov_trace_error(experiment; t_idx=length(experiment.saveat)) = tr(emp_cov(experiment.solution[t_idx])) - tr(experiment.true_cov[t_idx])

Lp_error(experiment; kwargs...) = true_metric(Lp_error, experiment; kwargs...)
true_mean_error(experiment; kwargs...) = true_metric((u, dist) -> norm(emp_mean(u), mean(dist)), experiment; kwargs...)
true_fourth_moment_error(experiment; kwargs...) = true_metric((u, dist) -> emp_abs_moment(u, 4) - abs_moment(dist, 4), experiment; kwargs...)

function true_metric(metric, experiment; t_idx=length(experiment.saveat), kwargs...)
    @unpack solution = experiment
    dist = experiment.true_dist[t_idx]
    return metric(solution[t_idx], dist; kwargs...)
end

function Base.show(io::IO, result::GradFlowExperimentResult)
    @unpack update_score_time, L2_error, true_mean_error, true_cov_trace_error, true_cov_norm_error, true_fourth_moment_error, sample_mean_error, sample_cov_trace_error = result
    println(io, "$(rpad("L2 error:", 25)) $(round(L2_error[1],digits=4))")
    println(io, "$(rpad("true mean error:", 25)) $(round(true_mean_error[1],digits=4))")
    println(io, "$(rpad("true cov trace error:", 25)) $(round(true_cov_trace_error[1],digits=4))")
    println(io, "$(rpad("true cov norm error:", 25)) $(round(true_cov_norm_error[1],digits=4))")
    println(io, "$(rpad("true fourth moment error:", 25)) $(round(true_fourth_moment_error[1],digits=4))")
    println(io, "$(rpad("sample mean error:", 25)) $(round(sample_mean_error[1],digits=4))")
    println(io, "$(rpad("sample cov trace error:", 25)) $(round(sample_cov_trace_error[1],digits=4))")
    println(io, "$(rpad("update score time:", 25)) $(round(update_score_time[1],digits=4))")
end

have_true_dist(experiment) = !isnothing(experiment.true_mean_error)