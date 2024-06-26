using GradientFlows, Plots, Polynomials, LinearAlgebra, LaTeXStrings
ENV["GKSwstype"] = "nul" # no GUI
default(display_type=:inline)

"metric_matrix[i,j] is the metric for the i-th value of n and the j-th trajectory"
function plot_metric_over_n(ns, trajectory_names, plot_title, metric_math_name, metric_matrix; scale=:log10, kwargs...)
    metric_matrix = abs.(metric_matrix)
    if scale == :log10
        metric_matrix .+= 1e-18
    else
        metric_matrix = round.(metric_matrix, digits=13)
    end
    log_slope(x, y) = Polynomials.fit(log.(abs.(x)), log.(abs.(y)), 1).coeffs[2]
    p = Plots.plot(title=plot_title, xlabel="number of patricles, n", ylabel=metric_math_name, margin=PLOT_MARGIN)
    for (j, trajectory_name) in enumerate(trajectory_names)
        slope = round(log_slope(ns, metric_matrix[:, j]), digits=2)
        Plots.plot!(p, ns, metric_matrix[:, j], label="$trajectory_name, log-slope=$slope", marker=:circle, yscale=scale, xscale=:log10, markerstrokewidth=0.4, lw=PLOT_LINE_WIDTH, size=PLOT_SMALL_WINDOW_SIZE)
    end
    return p
end

function scatter_plot(problem_name, d, n, solver_names; num_samples=min(2000, n), dir="data")
    p = Plots.plot(title="$problem_name, d=$d, $num_samples/$n samples")
    for solver in solver_names
        experiment = load(experiment_filename(problem_name, d, n, solver, 1; dir=dir))
        u = experiment.solution[end][:, 1:num_samples]
        scatter!(p, u[1, :], u[2, :], label=solver, markersize=4, markerstrokewidth=0.4, size = (1000, 1000))
    end
    return p
end

"experiment.saveat[t_idx] is the time at which to plot the pdf"
function marginal_pdf_plot(problem_name, d, n, solver_names; t_idx, xrange=range(-5, 5, length=200), dir="data", bandwidth=kde_bandwidth, marginal_coordinate=1)
    experiment = load(experiment_filename(problem_name, d, n, solver_names[1], 1; dir=dir))
    saveat = experiment.saveat
    t_idx = t_idx < 1 ? length(saveat) + t_idx : t_idx
    dist = experiment.true_dist[t_idx]
    p_marginal = Plots.plot(xlabel="x", ylabel="Σᵢϕ(x - Xᵢ[1])/n", title="marginal density n=$n t=$(saveat[t_idx]) coordinate=$marginal_coordinate", margin=PLOT_MARGIN,  size=PLOT_SMALL_WINDOW_SIZE)
    for solver in solver_names
        experiments = load_all_experiment_runs(problem_name, d, n, solver; dir=dir)
        us = [reshape(exp.solution[t_idx][marginal_coordinate,:], 1, :) for exp in experiments]
        bandwidths = [bandwidth(u) for u in us]
        pdf_val(x) = mean([kde([x], u; h=bandwidth(u)) for u in us]) 
        plot!(p_marginal, xrange, pdf_val, label="$solver h=$(round.(mean(bandwidths)[1], digits=3))", lw=PLOT_LINE_WIDTH)
    end
    plot!(p_marginal, xrange, x -> marginal_pdf(dist, x), label="true", lw=PLOT_LINE_WIDTH, linestyle=:dash, color=PLOT_COLOR_TRUTH)
    return p_marginal
end

function slice_pdf_plot(problem_name, d, n, solver_names; t_idx, xrange=range(-5, 5, length=200), dir="data", bandwidth=kde_bandwidth, slice_coordinate=1)
    slice(x::Number) = [zeros(typeof(x), slice_coordinate-1)..., x, zeros(typeof(x), d - slice_coordinate)...]
    experiment = load(experiment_filename(problem_name, d, n, solver_names[1], 1; dir=dir))
    saveat = experiment.saveat
    t_idx = t_idx < 1 ? length(saveat) + t_idx : t_idx
    dist = experiment.true_dist[t_idx]
    p_slice = Plots.plot(xlabel="x", ylabel="Σᵢϕ([0..., x, 0...] - Xᵢ)/n", title="slice density n=$n t=$(saveat[t_idx]) coordinate=$slice_coordinate", margin=PLOT_MARGIN,  size=PLOT_SMALL_WINDOW_SIZE)
    for solver in solver_names
        experiments = load_all_experiment_runs(problem_name, d, n, solver; dir=dir)
        us = [exp.solution[t_idx] for exp in experiments]
        bandwidths = [bandwidth(u) for u in us]
        pdf_val(x) = mean([kde(slice(x), u; h=bandwidth(u)) for u in us]) 
        plot!(p_slice, xrange, pdf_val, label="$solver h=$(round.(eigmax(mean(bandwidths)), digits=3))", lw=PLOT_LINE_WIDTH)
    end
    plot!(p_slice, xrange, x -> pdf(dist, slice(x)), label="true", lw=PLOT_LINE_WIDTH, linestyle=:dash, color=PLOT_COLOR_TRUTH)
    return p_slice
end

function save_pdfs_over_n(problem_name, d, ns, solver_names; dir="data")
    for n in ns
        p_marginal_start = marginal_pdf_plot(problem_name, d, n, solver_names, t_idx=1; dir=dir)
        p_marginal_end = marginal_pdf_plot(problem_name, d, n, solver_names, t_idx=0; dir=dir)
        p_slice_start = slice_pdf_plot(problem_name, d, n, solver_names, t_idx=1; dir=dir)
        p_slice_end = slice_pdf_plot(problem_name, d, n, solver_names, t_idx=0; dir=dir)
        plt = plot(p_marginal_start, p_marginal_end, p_slice_start, p_slice_end, size=PLOT_WINDOW_SIZE, margin=PLOT_MARGIN, plot_title="$problem_name, d=$d, n=$n", linewidth=PLOT_LINE_WIDTH, legendfontsize=PLOT_FONT_SIZE)
        path = joinpath(dir, "plots", problem_name, "d_$d", "pdf")
        plot_name = "n_$n"
        mkpath(path)
        savefig(plt, joinpath(path, plot_name))
    end
    nothing
end

"metric(experiment, step) returns a Vector of metric values with given step"
function plot_metric_over_t(problem_name, d, ns, solver_names, metric, metric_name, metric_math_name; step=1, kwargs...)
    p = Plots.plot(title="$metric_name n = $ns", xlabel="simulated time", ylabel=metric_math_name, margin=PLOT_MARGIN)
    for n in ns, solver_name in solver_names
        experiments = load_all_experiment_runs(problem_name, d, n, solver_name; kwargs...)
        saveat = round.(experiments[1].saveat, digits=3)
        metric_values = mean([metric(exp, step) for exp in experiments])
        plot!(p, saveat[1:step:end], metric_values, label="$solver_name, n=$n", lw=PLOT_LINE_WIDTH, size=PLOT_SMALL_WINDOW_SIZE)
    end
    return p
end

function plot_score_error(problem_name, d, ns, solver_names; kwargs...)
    function score_error(experiment, step)
        true_score_values = [score(dist, u) for (dist, u) in zip(experiment.true_dist[1:step:end], experiment.solution[1:step:end])]
        return sum.(abs2, experiment.score_values .- true_score_values) ./ sum.(abs2, true_score_values)
    end
    metric_name = "score_error"
    metric_math_name = "∑ᵢ |s(xᵢ) - ∇log ρ*(xᵢ)|² / ∑ᵢ |∇log ρ*(xᵢ)|²"
    return plot_metric_over_t(problem_name, d, ns, solver_names, score_error, metric_name, metric_math_name; kwargs...)
end

function plot_covariance_trajectory(problem_name, d, ns, solver_names; row, column, kwargs...)
    cov_(experiment, step) = [emp_cov(u)[row, column] for u in experiment.solution[1:step:end]]
    plt = plot_metric_over_t(problem_name, d, ns, solver_names, cov_, "covariance($row,$column)", "Σ$row$column"; kwargs...)
    experiment = load(experiment_filename(problem_name, d, ns[1], solver_names[1], 1; kwargs...))
    plot!(plt, experiment.saveat, getindex.(experiment.true_cov, row, column), label="true", lw=PLOT_LINE_WIDTH, linestyle=:dash, color=PLOT_COLOR_TRUTH)
    return plt
end

function plot_entropy_production_rate(problem_name, d, ns, solver_names; kwargs...)
    metric_name = "entropy_production_rate"
    metric_math_name = "d/dt ∫ρ(x)logρ(x)dx ≈ ∑ᵢ v[s](xᵢ)⋅s(xᵢ) / n"
    entropy_production_rate(experiment, step) = [dot(experiment.velocity_values[i], experiment.score_values[i]) / size(experiment.solution[i], 2) for i in 1:step:length(experiment.score_values)]
    return plot_metric_over_t(problem_name, d, ns, solver_names, entropy_production_rate, metric_name, metric_math_name; kwargs...)
    
end

function plot_w2(problem_name, d, ns, solver_names; step=10, ε=0.2, kwargs...)
    p = Plots.plot(title="Wasserstein dist n=$(ns[1]), $(ns[end])", xlabel="simulated time", ylabel=L"W_2(\rho^{%$(ns[1])}, \rho^{%$(ns[end])})", margin=PLOT_MARGIN)
    for solver_name in solver_names
        experiments_1 = load_all_experiment_runs(problem_name, d, ns[1], solver_name; kwargs...)
        experiments_2 = load_all_experiment_runs(problem_name, d, ns[end], solver_name; kwargs...)
        saveat = round.(experiments_1[1].saveat, digits=3)
        metric_values = mean([[w2(u1, u2; ε=ε) for (u1,u2) in zip(exp1.solution[1:step:end], exp2.solution[1:step:end])] for (exp1, exp2) in zip(experiments_1, experiments_2)])
        plot!(p, saveat[1:step:end], metric_values, label="$solver_name", lw=PLOT_LINE_WIDTH-1,size=PLOT_SMALL_WINDOW_SIZE)
    end
    return p
end

function plot_L2(problem_name, d, ns, solver_names; kwargs...)
    get_L2(experiment, step) = [Lp_error(u, dist;p=2) for (u, dist) in zip(experiment.solution[1:step:end], experiment.true_dist[1:step:end])]
    plot_metric_over_t(problem_name, d, ns, solver_names, get_L2, "L2_distance", "L²(ρᴺ, ρ*)"; step=20, kwargs...)
end

"""
    plot_all(problem_name, d, ns, solver_names; save=true, save_dir=joinpath("data","plots"), dir="data",
    metrics=..., kwargs...)
"""
function plot_all(problem_name, d, ns, solver_names; save=true, save_dir=joinpath("data","plots"), dir="data",
    metrics=[
        :L2_error,
        :true_cov_norm_error,
        :sample_cov_trace_error,
        :sample_mean_error], kwargs...)
    @info "Plotting $problem_name, d=$d"
    any_experiment = load(experiment_filename(problem_name, d, ns[1], solver_names[1], 1; dir=dir))
    dt = any_experiment.dt
    have_true_distribution = have_true_dist(any_experiment)
    ns_low_high = ns[[1,end]]

    ### plot ###
    plots = []
    plot_names = []
    if have_true_distribution
        # slice pdfs
        p_slice_start = slice_pdf_plot(problem_name, d, ns[end], solver_names, t_idx=1; dir=dir)
        p_slice_end = slice_pdf_plot(problem_name, d, ns[end], solver_names, t_idx=0; dir=dir)
        p_slice_start_low_n = slice_pdf_plot(problem_name, d, ns[1], solver_names, t_idx=1; dir=dir)
        p_slice_end_low_n = slice_pdf_plot(problem_name, d, ns[1], solver_names, t_idx=0; dir=dir)
        # scores
        p_score_error = plot_score_error(problem_name, d, ns_low_high, solver_names; dir=dir)
        # @time p_w2 = plot_w2(problem_name, d, ns_low_high, solver_names; dir=dir)
        push!(plots, p_slice_start, p_slice_end, p_slice_start_low_n, p_slice_end_low_n, p_score_error)#, p_w2)
        push!(plot_names, "slice_start", "slice_end", "slice_start_low_n", "slice_end_low_n", "score_error")#, "wasserstein_2_distance")
        # if d < 5
        #     @time p_L2 = plot_L2(problem_name, d, ns_low_high, solver_names; dir=dir)
        #     push!(plots, p_L2)
        #     push!(plot_names, "L2_distance")
        # end
        if d > 3 # plot marginal pdfs only for d > 3
            p_marginal_start = marginal_pdf_plot(problem_name, d, ns[end], solver_names; t_idx=1, dir=dir)
            p_marginal_end = marginal_pdf_plot(problem_name, d, ns[end], solver_names; t_idx=0, dir=dir)
            p_marginal_start_low_n = marginal_pdf_plot(problem_name, d, ns[1], solver_names, t_idx=1; dir=dir)
            p_marginal_end_low_n = marginal_pdf_plot(problem_name, d, ns[1], solver_names, t_idx=0; dir=dir)
            pushfirst!(plots, p_marginal_start, p_marginal_end, p_marginal_start_low_n, p_marginal_end_low_n)
            pushfirst!(plot_names, "marginal_start", "marginal_end", "marginal_start_low_n", "marginal_end_low_n")
        end
    end
    # covariance trajectory
    p_cov_trajectory_1 = plot_covariance_trajectory(problem_name, d, ns_low_high, solver_names; row=1, column=1, dir=dir)
    p_cov_trajectory_2 = plot_covariance_trajectory(problem_name, d, ns_low_high, solver_names; row=2, column=2, dir=dir)
    # entropy trajectory
    entropy_plot = plot_entropy_production_rate(problem_name, d, ns_low_high, solver_names; dir=dir)
    push!(plots, p_cov_trajectory_1, p_cov_trajectory_2, entropy_plot)
    push!(plot_names, "cov_trajectory_1", "cov_trajectory_2", "entropy_production_rate")
    # other metrics, against n
    for metric in metrics
        metric_matrix, metric_math_name = load_metric(problem_name, d, ns, solver_names, metric; dir=dir)
        metric_name = string(metric)
        any(isnan, metric_matrix) && continue
        p = plot_metric_over_n(ns, solver_names, metric_name, metric_math_name, metric_matrix; kwargs...)
        push!(plots, p)
        push!(plot_names, metric_name)
    end
    num_runs_ = GradientFlows.num_runs(problem_name, d, ns[1], solver_names[1]; dir=dir)
    plt_all = Plots.plot(plots..., plot_title="$problem_name, d=$d, $(ns[1])≤n≤$(ns[end]), dt=$dt, $num_runs_ runs", size=PLOT_WINDOW_SIZE, margin=PLOT_MARGIN, linewidth=PLOT_LINE_WIDTH, legendfontsize=PLOT_FONT_SIZE)
    
    ### save ###
    if save
        mkpath(joinpath(save_dir, "all"))
        path = joinpath(save_dir, problem_name, "d_$d")
        mkpath(path)
        saveplot(plt, plot_name) = savefig(plt, joinpath(path, plot_name))
        
        for (plt, name) in zip(plots, plot_names)
            saveplot(plt, name)
        end
        savefig(plt_all, joinpath(save_dir, "all", "$(problem_name)_d_$d"))
        # saveplot(scatter_plot(problem_name, d, ns[end], solver_names; dir=dir), "scatter")
        # save_pdfs_over_n(problem_name, d, ns, solver_names; dir=dir)
    end
    return plt_all
end

function plot_all(problems, ns, solvers; kwargs...)
    solver_names = name.(solvers)
    plots = []
    for (problem, d) in problems
        prob_name = problem(d, ns[1], solvers[1]).name
        push!(plots, plot_all(prob_name, d, ns, solver_names; kwargs...))
    end
    plots
end