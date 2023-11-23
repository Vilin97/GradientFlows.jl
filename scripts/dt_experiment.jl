using GradientFlows, StableRNGs, Plots

function get_results(problem_name, d, n, dts)
    results = Matrix{GradFlowExperimentResult}(undef, length(dts), 3)
    for (i,dt) in enumerate(dts)
        solvers = [Exact(), SBTM(best_model(problem_name, d)), Blob(blob_epsilon(d,n))]
        for (j,solver) in enumerate(solvers)
            @show dt, solver
            problem = diffusion_problem(d, n, solver; rng=StableRNG(123), dt=dt)
            result = GradFlowExperimentResult(Experiment(problem))
            results[i,j] = result
        end
    end
    results
end

function plot_metric(problem_name, d, n, dts, solver_names, metric_name, metric_matrix; scale=:log10)
    log_slope(x, y) = Polynomials.fit(log.(x), log.(y), 1).coeffs[2]
    p = Plots.plot(title="$problem_name, d=$d, n = $n, $metric_name", xlabel="time step, dt", ylabel=metric_name, size=PLOT_WINDOW_SIZE)
    for (j, solver_name) in enumerate(solver_names)
        slope = round(log_slope(ns, metric_matrix[:, j]), digits=2)
        Plots.plot!(p, dts, metric_matrix[:, j], label="$solver_name, log-slope=$slope", marker=:circle, yscale=scale, xscale=scale, markerstrokewidth=0.4)
    end
    return p
end

function plot_all(results, problem_name, d, n, dts, solver_names, metrics=[
    (:L2_error, "|ρₜ∗ϕ - ρₜ*|₂"),
    (:true_mean_error, "|E(Xₜ)-E(Xₜ*)|₂"),
    (:true_cov_trace_error, "|E |Xₜ|² - E |Xₜ*|²|"),
    (:true_cov_norm_error, "|Cov(Xₜ)-Cov(Xₜ*)|₂"),
    (:true_fourth_moment_error, "|E |Xₜ|⁴ - E |Xₜ*|⁴|")])
    for (metric, metric_name) in metrics
        metric_matrix = getfield.(results, metric)
        p = plot_metric(problem_name, d, n, dts, solver_names, metric_name, metric_matrix)
        push!(plots, p)
    end
    plt_all = Plots.plot(plots..., size=PLOT_WINDOW_SIZE, margin=(10, :mm))
    return plt_all
end

d = 2
n = 2000
dts = 2. .^ (-10:-1)
results = get_results("diffusion", d, n, dts)
plot_all(rsults, "diffusion", d, n, dts, ALL_SOLVER_NAMES)