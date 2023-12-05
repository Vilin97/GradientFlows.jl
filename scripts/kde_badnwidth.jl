using GradientFlows, Plots, LinearAlgebra

function pdf_plot(problem_name, d, n, solver_names; t_idx, xrange=range(-5, 5, length=200), dir="data", bandwidth_coeff=1)
    slice(x::Number) = [x, zeros(typeof(x), d - 1)...]
    experiment = load(experiment_filename(problem_name, d, n, "exact", 1; dir=dir))
    saveat = experiment.saveat
    dist = experiment.true_dist[t_idx]
    p_marginal = Plots.plot(size=PLOT_WINDOW_SIZE)
    p_slice = Plots.plot(size=PLOT_WINDOW_SIZE)
    for solver in solver_names
        experiments = load_all_experiment_runs(problem_name, d, n, solver; dir=dir)
        u = hcat([exp.solution[t_idx] for exp in experiments]...)
        u_marginal = reshape(u[1, :], 1, :)
        plot!(p_marginal, xrange, x -> kde([x], u_marginal; h=kde_bandwidth(u_marginal*bandwidth_coeff)), label=solver, title="marginal $problem_name, d=$d, n=$n, h=$(round(bandwidth_coeff,digits=3)) * $(round.(kde_bandwidth(u_marginal)[1],digits=3)), t=$(saveat[t_idx]), dt=$(experiment.dt)")
        plot!(p_slice, xrange, x -> kde(slice(x), u; h=kde_bandwidth(u*bandwidth_coeff)), label=solver, title="slice $problem_name, d=$d, n=$n, h=$(round(bandwidth_coeff,digits=3)) * $(round.((det(kde_bandwidth(u))^(1/d)), digits=3)), t=$(saveat[t_idx]), dt=$(experiment.dt)")
    end
    plot!(p_marginal, xrange, x -> marginal_pdf(dist, x), label="true")
    plot!(p_slice, xrange, x -> pdf(dist, slice(x)), label="true")
    return p_marginal, p_slice
end

problem_name = "landau"
d = 5
n = 25600
solver_names = ["blob", "sbtm", "exact"]
marginal_plots = []
slice_plots = []
for bandwidth_coeff in 0.5:0.5:3
    @show bandwidth_coeff
    @time p_marginal_end, p_slice_end = pdf_plot(problem_name, d, n, solver_names, t_idx=2, bandwidth_coeff=bandwidth_coeff)
    push!(marginal_plots, p_marginal_end)
    push!(slice_plots, p_slice_end)
end
plot(marginal_plots...)
plot(slice_plots...)
