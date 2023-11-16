using GradientFlows, Plots, Polynomials, TimerOutputs

"metric_matrix[i,j] is the metric for the i-th value of n and the j-th solver"
function plot_metric(problem_name, d, ns, solver_names, metric_name, metric_matrix; scale=:log10)
    log_slope(x, y) = Polynomials.fit(log.(x), log.(y), 1).coeffs[2]
    p = Plots.plot(title="$problem_name, d=$d, $metric_name", xlabel="number of patricles, n", ylabel=metric_name, size=PLOT_WINDOW_SIZE)
    for (j, solver_name) in enumerate(solver_names)
        slope = round(log_slope(ns, metric_matrix[:, j]), digits=2)
        Plots.plot!(p, ns, metric_matrix[:, j], label="$solver_name, log-slope=$slope", marker=:circle, yscale=scale, xscale=scale, markerstrokewidth=0.4)
    end
    return p
end

function scatter_plot(problem_name, d, n, solver_names; num_samples=min(2000, n))
    p = Plots.plot(title="$problem_name, d=$d, $num_samples/$n samples"; size=(PLOT_WINDOW_SIZE[2], PLOT_WINDOW_SIZE[2]))
    for solver in solver_names
        experiment = load(experiment_filename(problem_name, d, n, solver, 1))
        u = experiment.solution[end][:, 1:num_samples]
        scatter!(p, u[1, :], u[2, :], label=solver, markersize=4, markerstrokewidth=0.4)
    end
    return p
end

"experiment.saveat[t_idx] is the time at which to plot the pdf"
function pdf_plot(problem_name, d, n, solver_names; t_idx, xrange=range(-5, 5, length=200))
    experiment = load(experiment_filename(problem_name, d, n, "exact", 1))
    saveat = experiment.saveat
    dist = true_dist(experiment.problem, saveat[t_idx])
    p_marginal = Plots.plot(title="marginal $problem_name, d=$d, n=$n, ε=$(round(kde_epsilon(1,n),digits=4)), t=$(saveat[t_idx])", size=PLOT_WINDOW_SIZE)
    p_slice = Plots.plot(title="slice $problem_name, d=$d, n=$n, ε=$(round(kde_epsilon(d,n),digits=4)), t=$(saveat[t_idx])", size=PLOT_WINDOW_SIZE)
    slice(x::Number) = [x, zeros(typeof(x), d - 1)...]
    for solver in solver_names
        experiment = load(experiment_filename(problem_name, d, n, solver, 1))
        u = experiment.solution[t_idx]
        u_marginal = reshape(u[1, :], 1, :)
        plot!(p_marginal, xrange, x -> kde(x, u_marginal), label=solver)
        plot!(p_slice, xrange, x -> kde(slice(x), u), label=solver)
    end
    plot!(p_marginal, xrange, x -> marginal_pdf(dist, x), label="true")
    plot!(p_slice, xrange, x -> pdf(dist, slice(x)), label="true")
    return p_marginal, p_slice
end

function plot_all(problem_name, d, ns, solver_names; save=true,
    metrics=[
        (:update_score_time, "update score time, s"),
        (:L2_error, "|ρₜ∗ϕ - ρₜ*|₂"),
        (:true_mean_error, "|E(Xₜ)-E(Xₜ*)|₂"),
        (:true_cov_trace_error, "|E |Xₜ|² - E |Xₜ*|²|"),
        (:true_cov_norm_error, "|Cov(Xₜ)-Cov(Xₜ*)|₂"),
        (:true_fourth_moment_error, "|E |Xₜ|⁴ - E |Xₜ*|⁴|"),
        (:sample_mean_error, "|E(X₀)-E(Xₜ)|₂|"),
        (:sample_cov_trace_error, "|E |X₀|² - E |Xₜ|²|")])
    plots = []
    p_marginal_start, p_slice_start = pdf_plot(problem_name, d, ns[end], solver_names, t_idx=1)
    p_marginal_end, p_slice_end = pdf_plot(problem_name, d, ns[end], solver_names, t_idx=2)
    push!(plots, p_marginal_start, p_marginal_end, p_slice_start, p_slice_end)
    for (metric, metric_name) in metrics
        metric_matrix = load_metric(problem_name, d, ns, solver_names, metric)
        p = plot_metric(problem_name, d, ns, solver_names, metric_name, metric_matrix)
        push!(plots, p)
    end
    push!(plots, scatter_plot(problem_name, d, ns[end], solver_names))
    plt_all = Plots.plot(plots[1:end-1]..., size=PLOT_WINDOW_SIZE, margin=(10, :mm)) # don't include the scatter plot

    if save
        path = joinpath("data", "plots", problem_name, "d_$d")
        mkpath(path)
        metric_filenames = [string(metric) for (metric, _) in metrics]
        filenames = ["marginal_start", "marginal_end", "slice_start", "slice_end", metric_filenames..., "scatter"]
        for (plt, filename) in zip(plots, filenames)
            savefig(plt, joinpath(path, filename))
        end
        savefig(plt_all, joinpath(path, "all"))
    end
    return plt_all
end

