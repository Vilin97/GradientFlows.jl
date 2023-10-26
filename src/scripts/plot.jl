using GradientFlows, Plots, Polynomials

function load_metric(problem_name, d, ns, solver_names, metric::Symbol)
    metric_matrix = zeros(length(ns), length(solver_names))
    for (i,n) in enumerate(ns), (j,solver_name) in enumerate(solver_names)
        dir = dirname(experiment_filename(problem_name, d, n, solver_name, 1))
        filenames = joinpath.(dir, readdir(dir))
        # use the mean of all the runs
        metric_matrix[i, j] = mean([getfield(load(f), metric) for f in filenames])
    end
    return metric_matrix
end

function plot_metric(problem_name, d, ns, solver_names, metric_name, metric_matrix; scale = :log10)
    log_slope(x, y) = Polynomials.fit(log.(x), log.(y), 1).coeffs[2]
    p = Plots.plot(title = "$problem_name, d=$d, $metric_name", xlabel = "number of patricles, n", ylabel = metric_name)
    for (j,solver_name) in enumerate(solver_names)
        slope = round(log_slope(ns, metric_matrix[:,j]), digits=2)
        Plots.plot!(p, ns, metric_matrix[:,j], label="$solver_name, log-slope=$slope", marker=:circle, yscale=scale, xscale=scale)
    end
    return p
end

function plot_metrics(problem_name, d, ns, solver_names; metrics = [(:L2_error, "|ρ∗ϕ - ρ*|₂"), (:mean_norm_error, "|E(ρ)-E(ρ*)|₂"), (:cov_norm_error, "|Σ(ρ)-Σ(ρ*)|₂"), (:cov_trace_error, "|tr(Σ(ρ))-tr(Σ(ρ*))|")])
    plots = []
    for (metric, metric_name) in metrics
        metric_matrix = load_metric(problem_name, d, ns, solver_names, metric)
        p = plot_metric(problem_name, d, ns, solver_names, metric_name, metric_matrix)
        push!(plots, p)
    end
    Plots.plot(plots..., size=PLOT_WINDOW_SIZE, margin=(5, :mm))
end

function scatter_plot(problem_name, d, n, solver_names; num_samples=min(5000, n))
    p = Plots.plot(title = "$problem_name, d=$d, $num_samples/$n samples"; size=(PLOT_WINDOW_SIZE[2], PLOT_WINDOW_SIZE[2]))
    for solver in solver_names
        experiment = load(experiment_filename(problem_name, d, n, solver, 1))
        u = experiment.solution[end][:, 1:num_samples]
        scatter!(p, u[1,:], u[2,:], label=solver, markersize=4)
    end
    return p
end

"experiment.saveat[t_idx] is the time at which to plot the pdf"
function pdf_plot(problem_name, d, n, solver_names; t_idx, xrange=range(-5,5,length=200))
    experiment = load(experiment_filename(problem_name, d, n, "exact", 1))
    saveat = experiment.saveat
    dist = true_dist(experiment.problem, saveat[t_idx])
    p = Plots.plot(title = "$problem_name, d=$d, ε=$(kde_epsilon(d,n)), t=$(saveat[t_idx])")
    for solver in solver_names
        experiment = load(experiment_filename(problem_name, d, n, solver, 1))
        u = reshape(experiment.solution[t_idx][1,:], 1, :)
        plot!(p, xrange, x -> kde(x, u), label=solver)
    end
    plot!(p, xrange, x -> marginal_pdf(dist, x), label="true")
    return p
end

d = 5
ns = 100 * 2 .^ (0:4)
problem_name = "landau"
solver_names = ["exact", "sbtm", "blob"]
plot_metrics(problem_name, d, ns, solver_names)
scatter_plot(problem_name, d, ns[end], solver_names)
pdf_plot(problem_name, d, ns[end], solver_names, t_idx=1)