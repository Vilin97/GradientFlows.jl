using GradientFlows, Plots, Polynomials, TimerOutputs, LinearAlgebra
ENV["GKSwstype"] = "nul" # no GUI
default(display_type=:inline)

"metric_matrix[i,j] is the metric for the i-th value of n and the j-th solver"
function plot_metric(problem_name, d, ns, solver_names, metric_name, metric_math_name, metric_matrix; scale=:log10, kwargs...)
    if scale == :log10
        metric_matrix = abs.(metric_matrix)
    end
    log_slope(x, y) = Polynomials.fit(log.(abs.(x)), log.(abs.(y)), 1).coeffs[2]
    p = Plots.plot(title=metric_name, xlabel="number of patricles, n", ylabel=metric_math_name, size=PLOT_WINDOW_SIZE)
    for (j, solver_name) in enumerate(solver_names)
        slope = round(log_slope(ns, metric_matrix[:, j]), digits=2)
        Plots.plot!(p, ns, metric_matrix[:, j], label="$solver_name, log-slope=$slope", marker=:circle, yscale=scale, xscale=:log10, markerstrokewidth=0.4)
    end
    return p
end

function scatter_plot(problem_name, d, n, solver_names; num_samples=min(2000, n), dir="data")
    p = Plots.plot(title="$problem_name, d=$d, $num_samples/$n samples"; size=(PLOT_WINDOW_SIZE[2], PLOT_WINDOW_SIZE[2]))
    for solver in solver_names
        experiment = load(experiment_filename(problem_name, d, n, solver, 1; dir=dir))
        u = experiment.solution[end][:, 1:num_samples]
        scatter!(p, u[1, :], u[2, :], label=solver, markersize=4, markerstrokewidth=0.4)
    end
    return p
end

"experiment.saveat[t_idx] is the time at which to plot the pdf"
function pdf_plot(problem_name, d, n, solver_names; t_idx, xrange=range(-5, 5, length=200), dir="data")
    slice(x::Number) = [x, zeros(typeof(x), d - 1)...]
    experiment = load(experiment_filename(problem_name, d, n, solver_names[1], 1; dir=dir))
    saveat = experiment.saveat
    dist = experiment.true_dist[t_idx]
    p_marginal = Plots.plot(size=PLOT_WINDOW_SIZE, xlabel="x", ylabel="Σᵢϕ(x - Xᵢ[1])/n", title="marginal density n=$n t=$(saveat[t_idx])")
    p_slice = Plots.plot(size=PLOT_WINDOW_SIZE, xlabel="x", ylabel="Σᵢϕ([x,0...] - Xᵢ)/n", title="slice density n=$n t=$(saveat[t_idx])")
    for solver in solver_names
        experiments = load_all_experiment_runs(problem_name, d, n, solver; dir=dir)
        u = hcat([exp.solution[t_idx] for exp in experiments]...)
        u_marginal = reshape(u[1, :], 1, :)
        plot!(p_marginal, xrange, x -> kde([x], u_marginal), label="$solver h=$(round.(kde_bandwidth(u_marginal)[1],digits=3))")
        plot!(p_slice, xrange, x -> kde(slice(x), u), label="$solver h=$(round.((det(kde_bandwidth(u))^(1/d)), digits=3))")
    end
    plot!(p_marginal, xrange, x -> marginal_pdf(dist, x), label="true")
    plot!(p_slice, xrange, x -> pdf(dist, slice(x)), label="true")
    return p_marginal, p_slice
end

function cov_plot(problem_name, d, ns, solver_names; t_idx, dir="data")
    experiment = load(experiment_filename(problem_name, d, ns[1], solver_names[1], 1; dir=dir))
    saveat = experiment.saveat
    p1 = Plots.plot(size=PLOT_WINDOW_SIZE, xlabel="n", ylabel="Σ₁₁", title="Cov(Xₜ)₁₁ at t = $(saveat[t_idx])", xscale=:log10)
    p2 = Plots.plot(size=PLOT_WINDOW_SIZE, xlabel="n", ylabel="Σ₂₂", title="Cov(Xₜ)₂₂ at t = $(saveat[t_idx])", xscale=:log10)
    for solver in solver_names
        covs_1 = [mean([emp_cov(exp.solution[t_idx])[1,1] for exp in load_all_experiment_runs(problem_name, d, n, solver; dir=dir)]) for n in ns]
        covs_2 = [mean([emp_cov(exp.solution[t_idx])[2,2] for exp in load_all_experiment_runs(problem_name, d, n, solver; dir=dir)]) for n in ns]
        plot!(p1, ns, covs_1, label=solver)
        plot!(p2, ns, covs_2, label=solver)
    end
    plot!(p1, ns, fill(experiment.true_cov[t_idx][1,1], length(ns)), label="true")
    plot!(p2, ns, fill(experiment.true_cov[t_idx][2,2], length(ns)), label="true")
    return p1, p2
end

function plot_all(problem_name, d, ns, solver_names; save=true, dir="data",
    metrics=[
        :update_score_time,
        :L2_error,
        :top_eigenvalue_error,
        :bottom_eigenvalue_error,
        :true_cov_trace_error,
        :true_cov_norm_error, 
        :sample_mean_error,
        :sample_cov_trace_error], kwargs...)
    println("Plotting $problem_name, d=$d")
    dt = load(experiment_filename(problem_name, d, ns[1], solver_names[1], 1; dir=dir)).dt
    plots = []
    plot_filenames = String[]
    try
        p_marginal_start, p_slice_start = pdf_plot(problem_name, d, ns[end], solver_names, t_idx=1)
        p_marginal_end, p_slice_end = pdf_plot(problem_name, d, ns[end], solver_names, t_idx=2)
        push!(plots, p_marginal_start, p_marginal_end, p_slice_start, p_slice_end)
        push!(plot_filenames, "marginal_start", "marginal_end", "slice_start", "slice_end")
    catch e
        cov_plot1, cov_plot2 = cov_plot(problem_name, d, ns, solver_names, t_idx=2)
        push!(plots, cov_plot1, cov_plot2)
        push!(plot_filenames, "cov1", "cov2")
    end
    for metric in metrics
        metric_matrix, metric_math_name = load_metric(problem_name, d, ns, solver_names, metric; dir=dir)
        metric_name = string(metric)
        any(isnan, metric_matrix) && continue
        p = plot_metric(problem_name, d, ns, solver_names, metric_name, metric_math_name, metric_matrix; kwargs...)
        push!(plots, p)
        push!(plot_filenames, metric_name)
    end
    push!(plots, scatter_plot(problem_name, d, ns[end], solver_names))
    push!(plot_filenames, "scatter")
    plt_all = Plots.plot(plots[1:end-1]..., size=PLOT_WINDOW_SIZE, margin=(13, :mm), plot_title="$problem_name, d=$d, $(ns[1])≤n≤$(ns[end]), dt=$dt", linewidth=3)

    if save
        path = joinpath(dir, "plots", problem_name, "d_$d")
        mkpath(path)
        for (plt, filename) in zip(plots, plot_filenames)
            savefig(plt, joinpath(path, filename))
        end
        path = joinpath(dir, "plots", "all")
        mkpath(path)
        savefig(plt_all, joinpath(path, "$(problem_name)_d_$d"))
    end
    return plt_all
end

function plot_all(problems, ns, solvers; kwargs...)
    solver_names = name.(solvers)
    for (problem, d) in problems
        prob_name = problem(d, ns[1], solvers[1]).name
        plot_all(prob_name, d, ns, solver_names; kwargs...)
    end
    nothing
end