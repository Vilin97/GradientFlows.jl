using GradientFlows, Plots, Polynomials, TimerOutputs

function update_score_time(timer)
    # it's the key that does not contain "Lp"
    key = first(filter(x -> !occursin("Lp", x), keys(timer.inner_timers)))
    return TimerOutputs.time(timer[key].inner_timers["update score"]) / 10^9
end

function load_metric(problem_name, d, ns, solver_names, metric::Symbol)
    f(m) = metric == :timer ? update_score_time(m) : m
    metric_matrix = zeros(length(ns), length(solver_names))
    for (i, n) in enumerate(ns), (j, solver_name) in enumerate(solver_names)
        dir = dirname(experiment_filename(problem_name, d, n, solver_name, 1))
        filenames = joinpath.(dir, readdir(dir))
        # use the mean of all the runs
        metric_matrix[i, j] = mean(f, [getfield(load(f), metric) for f in filenames])
    end
    return metric_matrix
end

"metric_matrix[i,j] is the metric for the i-th value of n and the j-th solver"
function plot_metric(problem_name, d, ns, solver_names, metric_name, metric_matrix; scale=:log10)
    log_slope(x, y) = Polynomials.fit(log.(x), log.(y), 1).coeffs[2]
    p = Plots.plot(title="$problem_name, d=$d, $metric_name", xlabel="number of patricles, n", ylabel=metric_name, size=PLOT_WINDOW_SIZE)
    for (j, solver_name) in enumerate(solver_names)
        slope = round(log_slope(ns, metric_matrix[:, j]), digits=2)
        Plots.plot!(p, ns, metric_matrix[:, j], label="$solver_name, log-slope=$slope", marker=:circle, yscale=scale, xscale=scale)
    end
    return p
end

function scatter_plot(problem_name, d, n, solver_names; num_samples=min(2000, n))
    p = Plots.plot(title="$problem_name, d=$d, $num_samples/$n samples"; size=(PLOT_WINDOW_SIZE[2], PLOT_WINDOW_SIZE[2]))
    for solver in solver_names
        experiment = load(experiment_filename(problem_name, d, n, solver, 1))
        u = experiment.solution[end][:, 1:num_samples]
        scatter!(p, u[1, :], u[2, :], label=solver, markersize=4)
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

function plot_all(problem_name, d, ns, solver_names; precomputed_metrics=[(:L2_error, "|ρ∗ϕ - ρ*|₂"), (:mean_norm_error, "|E(ρ)-E(ρ*)|₂"), (:cov_norm_error, "|Σ(ρ)-Σ(ρ*)|₂"), (:cov_trace_error, "|tr(Σ(ρ))-tr(Σ(ρ*))|"), (:timer, "update score time, s")], save=true)
    plots = []
    p_marginal_start, p_slice_start = pdf_plot(problem_name, d, ns[end], solver_names, t_idx=1)
    p_marginal_end, p_slice_end = pdf_plot(problem_name, d, ns[end], solver_names, t_idx=2)
    push!(plots, p_marginal_start, p_marginal_end, p_slice_start, p_slice_end)
    for (metric, metric_name) in precomputed_metrics
        metric_matrix = load_metric(problem_name, d, ns, solver_names, metric)
        p = plot_metric(problem_name, d, ns, solver_names, metric_name, metric_matrix)
        push!(plots, p)
    end
    push!(plots, scatter_plot(problem_name, d, ns[end], solver_names))
    plt_all = Plots.plot(plots..., size=PLOT_WINDOW_SIZE, margin=(10, :mm))

    if save
        path = joinpath("data", "plots", problem_name, "d_$d")
        mkpath(path)
        metric_filenames = [string(metric) for (metric, _) in precomputed_metrics]
        filenames = ["marginal_start", "marginal_end", "slice_start", "slice_end", metric_filenames..., "scatter"]
        for (plt, filename) in zip(plots, filenames)
            savefig(plt, joinpath(path, filename))
        end
        savefig(plt_all, joinpath(path, "all"))
    end
    return plt_all
end

ns = 100 * 2 .^ (0:8)
solver_names = ["exact", "sbtm", "blob"]
problems = [(2, "diffusion"), (5, "diffusion"), (3, "landau"), (5, "landau"), (10, "landau")]
for (d, problem_name) in problems
    @show d, problem_name
    @time plot_all(problem_name, d, ns, solver_names)
end