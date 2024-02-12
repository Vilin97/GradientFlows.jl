using GradientFlows, StableRNGs, LinearAlgebra, Plots
using GradientFlows: blob_bandwidth, long_name
using TimerOutputs

verbose = 0
n = 2000
ds = [3, 5]
problems = [diffusion_problem, fpe_problem]
plot_data = Matrix{Vector{Any}}(undef, length(ds), length(problems)) # plot_data[d, p] = (score_diff_vector, label)
for (i, d) in enumerate(ds)
    println("            d=$d")
    for (j, problem) in enumerate(problems)
        plot_data[i, j] = []
        prob_ = problem(d, n, Exact())
        println("        $(prob_.name)")
        ts = collect(prob_.tspan[1]:prob_.dt:prob_.tspan[2])

        solvers = [
            NPF(mlp(d, depth=1); verbose=verbose),
            NPF(mlp(d, depth=2); verbose=verbose),
            NPF(mlp(d, depth=3); verbose=verbose),
        ]

        for solver in solvers
            println("    $(long_name(solver))")
            prob = problem(d, n, solver)
            set_u0!(prob, prob.u0)
            @time (@timeit DEFAULT_TIMER "$problem $d $(long_name(solver))" sol = solve(prob))

            logged_score_values = prob.solver.logger.score_values
            true_score_values = [true_score(prob, t, u) for (t, u) in zip(ts, sol.u)]
            score_val_diff = sum.(abs2, logged_score_values .- true_score_values) ./ sum.(abs2, true_score_values)
            label = "$(long_name(prob.solver))"
            push!(plot_data[i, j], (logged_score_values, true_score_values, label))
        end
    end
end

@show DEFAULT_TIMER

plots = []
for (i, d) in enumerate(ds), (j, problem) in enumerate(problems)
    prob_name = problem(d, n, Exact()).name
    plt_ = plot(xlabel="time step", title="$prob_name d=$d, n=$n", ylabel="Σᵢ|s(xᵢ)|^2")
    mean_true_score_values = mean([sum.(abs2, triple[2]) for triple in plot_data[i, j]])
    plot!(plt_, mean_true_score_values, label="true", linestyle=:dash)
    for (logged_score_values, true_score_values, label) in plot_data[i, j]
        plot!(plt_, sum.(abs2, logged_score_values), label=label)
    end
    push!(plots, plt_)
end

plt = plot(plots..., plot_title="Score trajectories", size=PLOT_WINDOW_SIZE, linewidth=4, margin=(13, :mm))
savefig(plt, "data/plots/score_plots/npf_score_trajectories_nn_sizes")