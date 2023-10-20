using GradientFlows, Plots, Polynomials
d = 5
ns = [1000, 2000, 5000]
problem_name = "landau"
solver_names = ["blob", "sbtm", "exact"]
num_runs = 10
function load_metric(problem_name, solver_names, d, ns, num_runs, metric)
    metric_matrix = zeros(length(ns), length(solver_names))
    for (i,n) in enumerate(ns), (j,solver_name) in enumerate(solver_names)
        filename = experiment_filename(problem_name, solver_name, d, n, num_runs)
        experiment = load(filename)
        metric_matrix[i, j] = getfield(experiment, metric)
    end
    return metric_matrix
end

L2_errors = load_metric(problem_name, solver_names, d, ns, num_runs, :L2_error)
mean_norm_errors = load_metric(problem_name, solver_names, d, ns, num_runs, :mean_norm_error)
cov_norm_errors = load_metric(problem_name, solver_names, d, ns, num_runs, :cov_norm_error)
cov_trace_errors = load_metric(problem_name, solver_names, d, ns, num_runs, :cov_trace_error)

log_slope(x, y) = Polynomials.fit(log.(x), log.(y), 1).coeffs[2]

function plot_metric(problem_name, solver_names, d, ns, metric_name, metric_matrix; scale = :log)
    p = Plots.plot(title = "$problem_name, d=$d, $metric_name", xlabel = "number of patricles, n", ylabel = metric_name)
    for (j,solver_name) in enumerate(solver_names)
        slope = round(log_slope(ns, metric_matrix[:,j]), digits=2)
        Plots.plot!(p, ns, metric_matrix[:,j], label="$solver_name, log-slope=$slope", marker=:circle, yscale=scale, xscale=scale)
    end
    return p
end

p1 = plot_metric(problem_name, solver_names, d, ns, "|ρ∗ϕ - ρ*|₂", L2_errors);
p2 = plot_metric(problem_name, solver_names, d, ns, "|E(ρ)-E(ρ*)|₂", mean_norm_errors);
p3 = plot_metric(problem_name, solver_names, d, ns, "|Σ(ρ)-Σ(ρ*)|₂", cov_norm_errors);
p4 = plot_metric(problem_name, solver_names, d, ns, "|tr(Σ(ρ)-Σ(ρ*))|", cov_trace_errors);

Plots.plot(p1, p2, p3, p4, size=(1200, 800))