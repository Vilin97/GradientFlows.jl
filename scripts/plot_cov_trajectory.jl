using GradientFlows, Plots, Polynomials, LinearAlgebra, LaTeXStrings
ENV["GKSwstype"] = "nul" # no GUI
default(display_type=:inline)

function plot_covariance_11(problem_name, d, solver_names; step=1, dir="data")
    ns = [100, 200, 400, 800, 1600, 3200, 6400, 12800,25600]
    row, column = 1, 1
    cov_(experiment, step) = [emp_cov(u)[row, column] for u in experiment.solution[1:step:end]]
    plt = Plots.plot(title="Covariance (1,1) over time", xlabel="Simulated time", ylabel="Σ₁₁")
    
    for solver_name in solver_names
        for n in ns
            experiments = load_all_experiment_runs(problem_name, d, n, solver_name; dir=dir)
            saveat = round.(experiments[1].saveat, digits=3)
            cov_values = mean([cov_(exp, step) for exp in experiments])
            plot!(plt, saveat[1:step:end], cov_values, label="$solver_name n=$n", lw=2)
        end
    end
    
    # Plot the true covariance trajectory
    experiment = load(experiment_filename(problem_name, d, ns[1], solver_names[1], 1; dir=dir))
    plot!(plt, experiment.saveat, getindex.(experiment.true_cov, row, column), label="true", lw=2, linestyle=:dash, color=:black)
    
    return plt
end

for solver in ["blob", "sbtm"]
    plt = plot_covariance_11("maxwell_landau_normal", 3, [solver]; step=1, dir="data")
    savefig(plt, "covariance_11_$solver.png")
end