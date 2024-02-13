using GradientFlows, Plots, Polynomials, TimerOutputs, LinearAlgebra
ENV["GKSwstype"] = "nul" # no GUI
default(display_type=:inline)

"metric_matrix[i,j] is the metric for the i-th value of n and the j-th solver"
function plot_metric_over_n(problem_name, d, ns, solver_names, metric_name, metric_math_name, metric_matrix; scale=:log10, kwargs...)
    metric_matrix = abs.(metric_matrix)
    if scale == :log10
        metric_matrix .+= 1e-18
    else
        metric_matrix = round.(metric_matrix, digits=13)
    end
    log_slope(x, y) = Polynomials.fit(log.(abs.(x)), log.(abs.(y)), 1).coeffs[2]
    p = Plots.plot(title=metric_name, xlabel="number of patricles, n", ylabel=metric_math_name, margin=PLOT_MARGIN)
    for (j, solver_name) in enumerate(solver_names)
        slope = round(log_slope(ns, metric_matrix[:, j]), digits=2)
        Plots.plot!(p, ns, metric_matrix[:, j], label="$solver_name, log-slope=$slope", marker=:circle, yscale=scale, xscale=:log10, markerstrokewidth=0.4, lw=PLOT_LINE_WIDTH)
    end
    return p
end

function scatter_plot(problem_name, d, n, solver_names; num_samples=min(2000, n), dir="data")
    p = Plots.plot(title="$problem_name, d=$d, $num_samples/$n samples")
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
    t_idx = t_idx < 1 ? length(saveat) + t_idx : t_idx
    dist = experiment.true_dist[t_idx]
    p_marginal = Plots.plot(xlabel="x", ylabel="Σᵢϕ(x - Xᵢ[1])/n", title="marginal density n=$n t=$(saveat[t_idx])", margin=PLOT_MARGIN)
    p_slice = Plots.plot(xlabel="x", ylabel="Σᵢϕ([x,0...] - Xᵢ)/n", title="slice density n=$n t=$(saveat[t_idx])", margin=PLOT_MARGIN)
    for solver in solver_names
        experiments = load_all_experiment_runs(problem_name, d, n, solver; dir=dir)
        u = hcat([exp.solution[t_idx] for exp in experiments]...)
        u_marginal = reshape(u[1, :], 1, :)
        plot!(p_marginal, xrange, x -> kde([x], u_marginal), label="$solver h=$(round.(kde_bandwidth(u_marginal)[1],digits=3))", lw=PLOT_LINE_WIDTH)
        plot!(p_slice, xrange, x -> kde(slice(x), u), label="$solver h=$(round.((det(kde_bandwidth(u))^(1/d)), digits=3))", lw=PLOT_LINE_WIDTH)
    end
    plot!(p_marginal, xrange, x -> marginal_pdf(dist, x), label="true", lw=PLOT_LINE_WIDTH)
    plot!(p_slice, xrange, x -> pdf(dist, slice(x)), label="true", lw=PLOT_LINE_WIDTH)
    return p_marginal, p_slice
end

"metric(experiment) isa Vector of length(experiment.saveat)"
function plot_metric_over_t(problem_name, d, n, solver_names, metric, metric_name, metric_math_name; kwargs...)
    p = Plots.plot(title="$metric_name n=$n", xlabel="simulated time", ylabel=metric_math_name, margin=PLOT_MARGIN)
    for solver_name in solver_names
        experiments = load_all_experiment_runs(problem_name, d, n, solver_name; kwargs...)
        saveat = round.(experiments[1].saveat, digits=3)
        metric_ = mean([metric(exp) for exp in experiments])
        plot!(p, saveat, metric_, label=solver_name, lw=PLOT_LINE_WIDTH)
    end
    return p
end

function plot_score_error(problem_name, d, n, solver_names; kwargs...)
    function score_error(experiment)
        true_score_values = [score(dist, u) for (dist, u) in zip(experiment.true_dist, experiment.solution)]
        return sum.(abs2, experiment.score_values .- true_score_values) ./ sum.(abs2, true_score_values)
    end
    metric_name = "score_error"
    metric_math_name = "∑ᵢ |s(xᵢ) - ∇log ρ*(xᵢ)|² / ∑ᵢ |∇log ρ*(xᵢ)|²"
    return plot_metric_over_t(problem_name, d, n, solver_names, score_error, metric_name, metric_math_name; kwargs...)
end

function plot_covariance_trajectory(problem_name, d, n, solver_names; row, column, kwargs...)
    cov_(experiment) = [emp_cov(u)[row, column] for u in experiment.solution]
    plt = plot_metric_over_t(problem_name, d, n, solver_names, cov_, "covariance($row,$column)", "Σ$row$column"; kwargs...)
    experiment = load(experiment_filename(problem_name, d, n, solver_names[1], 1; kwargs...))
    plot!(plt, experiment.saveat, getindex.(experiment.true_cov, row, column), label="true", lw=PLOT_LINE_WIDTH, linestyle=:dash)
    return plt
end

function plot_all(problem_name, d, ns, solver_names; save=true, dir="data",
    metrics=[
        :top_eigenvalue_error,
        :update_score_time,
        :L2_error,
        :true_cov_trace_error,
        :true_cov_norm_error,
        :sample_mean_error,
        :sample_cov_trace_error], kwargs...)
    println("Plotting $problem_name, d=$d")
    any_experiment = load(experiment_filename(problem_name, d, ns[1], solver_names[1], 1; dir=dir))
    dt = any_experiment.dt
    have_true_distribution = have_true_dist(any_experiment)

    plots = []
    if save
        path = joinpath(dir, "plots", problem_name, "d_$d")
        mkpath(path)
        saveplot(plt, plot_name) = savefig(plt, joinpath(path, plot_name))
        saveplots(plts, plot_names) =
            for (plt, name) in zip(plts, plot_names)
                saveplot(plt, name)
            end
    end

    ### plot ###
    save && saveplot(scatter_plot(problem_name, d, ns[end], solver_names), "scatter")
    if have_true_distribution
        p_marginal_start, p_slice_start = pdf_plot(problem_name, d, ns[end], solver_names, t_idx=1)
        p_marginal_end, p_slice_end = pdf_plot(problem_name, d, ns[end], solver_names, t_idx=0)
        p_cov_trajectory_1 = plot_covariance_trajectory(problem_name, d, ns[end], solver_names; row=1, column=1, dir=dir)
        p_score_error = plot_score_error(problem_name, d, ns[end], solver_names; dir=dir)
        plts_ = [p_marginal_start, p_marginal_end, p_slice_start, p_slice_end, p_cov_trajectory_1, p_score_error]
        push!(plots, plts_...)
        save && saveplots(plts_, ["marginal_start", "marginal_end", "slice_start", "slice_end", "cov_trajectory_1", "score_error"])
    else
        p_cov_trajectory_1 = plot_covariance_trajectory(problem_name, d, ns[end], solver_names; row=1, column=1, dir=dir)
        p_cov_trajectory_2 = plot_covariance_trajectory(problem_name, d, ns[end], solver_names; row=2, column=2, dir=dir)
        plts_ = [p_cov_trajectory_1, p_cov_trajectory_2]
        push!(plots, plts_...)
        save && saveplots(plts_, ["cov_trajectory_1", "cov_trajectory_2"])
        insert!(metrics, 2, :bottom_eigenvalue_error)
    end
    for metric in metrics
        metric_matrix, metric_math_name = load_metric(problem_name, d, ns, solver_names, metric; dir=dir)
        metric_name = string(metric)
        any(isnan, metric_matrix) && continue
        p = plot_metric_over_n(problem_name, d, ns, solver_names, metric_name, metric_math_name, metric_matrix; kwargs...)
        push!(plots, p)
        save && saveplot(p, metric_name)
    end
    plt_all = Plots.plot(plots..., size=PLOT_WINDOW_SIZE, margin=PLOT_MARGIN, plot_title="$problem_name, d=$d, $(ns[1])≤n≤$(ns[end]), dt=$dt", linewidth=PLOT_LINE_WIDTH)

    ### save ###
    if save
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