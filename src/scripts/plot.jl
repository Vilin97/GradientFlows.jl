using GradientFlows, Plots, Polynomials

function solve_time(timer)
    # it's the key that does not contain "Lp"
    key = first(filter(x -> !occursin("Lp",x), keys(timer.inner_timers)))
    return TimerOutputs.time(timer[key])/10^9
end

function load_metric(problem_name, d, ns, solver_names, metric::Symbol)
    f(m) = metric == :timer ? solve_time(m) : m
    metric_matrix = zeros(length(ns), length(solver_names))
    for (i,n) in enumerate(ns), (j,solver_name) in enumerate(solver_names)
        dir = dirname(experiment_filename(problem_name, d, n, solver_name, 1))
        filenames = joinpath.(dir, readdir(dir))
        # use the mean of all the runs
        metric_matrix[i, j] = mean(f, [getfield(load(f), metric) for f in filenames])
    end
    return metric_matrix
end

function plot_metric(problem_name, d, ns, solver_names, metric_name, metric_matrix; scale = :log10)
    log_slope(x, y) = Polynomials.fit(log.(x), log.(y), 1).coeffs[2]
    p = Plots.plot(title = "$problem_name, d=$d, $metric_name", xlabel = "number of patricles, n", ylabel = metric_name, size=PLOT_WINDOW_SIZE)
    for (j,solver_name) in enumerate(solver_names)
        slope = round(log_slope(ns, metric_matrix[:,j]), digits=2)
        Plots.plot!(p, ns, metric_matrix[:,j], label="$solver_name, log-slope=$slope", marker=:circle, yscale=scale, xscale=scale)
    end
    return p
end

function scatter_plot(problem_name, d, n, solver_names; num_samples=min(2000, n))
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
    p = Plots.plot(title = "$problem_name, d=$d, n=$n, ε=$(round(kde_epsilon(d,n),digits=4)), t=$(saveat[t_idx])", size=PLOT_WINDOW_SIZE)
    for solver in solver_names
        experiment = load(experiment_filename(problem_name, d, n, solver, 1))
        u = reshape(experiment.solution[t_idx][1,:], 1, :)
        plot!(p, xrange, x -> kde(x, u), label=solver)
    end
    plot!(p, xrange, x -> marginal_pdf(dist, x), label="true")
    return p
end

function plot_all(problem_name, d, ns, solver_names; metrics = [(:L2_error, "|ρ∗ϕ - ρ*|₂"), (:mean_norm_error, "|E(ρ)-E(ρ*)|₂"), (:cov_norm_error, "|Σ(ρ)-Σ(ρ*)|₂"), (:cov_trace_error, "|tr(Σ(ρ))-tr(Σ(ρ*))|"), (:timer, "solve time, s")], save=true)
    plots = []
    push!(plots, scatter_plot(problem_name, d, ns[end], solver_names))
    push!(plots, pdf_plot(problem_name, d, ns[end], solver_names, t_idx=1))
    push!(plots, pdf_plot(problem_name, d, ns[end], solver_names, t_idx=2))
    for (metric, metric_name) in metrics
        metric_matrix = load_metric(problem_name, d, ns, solver_names, metric)
        p = plot_metric(problem_name, d, ns, solver_names, metric_name, metric_matrix)
        push!(plots, p)
    end
    plt_all = Plots.plot(plots..., size=PLOT_WINDOW_SIZE, margin=(10, :mm))
    
    if save
        path = joinpath("data", "plots", problem_name, "d_$d")
        mkpath(path)
        metric_filenames = [string(metric) for (metric, _) in metrics]
        filenames = ["scatter", "pdf_start", "pdf_end", metric_filenames...]
        for (plt, filename) in zip(plots, filenames)
            savefig(plt, joinpath(path, filename))
        end
        savefig(plt_all, joinpath(path, "all"))
    end
    return plt_all
end

ns = 100 * 2 .^ (0:7)
solver_names = ["exact", "sbtm", "blob"]
for (d,problem_name) in [(2,"diffusion"), (5,"diffusion"), (3,"landau"), (5,"landau")]
    @show d, problem_name
    plot_all(problem_name, d, ns, solver_names)
end