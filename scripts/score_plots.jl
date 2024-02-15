using GradientFlows, StableRNGs, LinearAlgebra, Plots
using GradientFlows: blob_bandwidth, long_name, l2_error_normalized
using TimerOutputs

ENV["GKSwstype"] = "nul" # no GUI
default(display_type=:inline)

function get_plot_data(problems, ds, n; verbose=1)
    plot_data = Matrix{Vector{Any}}(undef, length(problems), length(ds)) # plot_data[p, d] = (logged_score_values, true_score_values, label)
    for (i, problem) in enumerate(problems), (j, d) in enumerate(ds)
        plot_data[i,j] = []
        prob_exact = problem(d, n, Exact())
        println("        $(prob_exact.name) d = $d")
        ts = collect(prob_exact.tspan[1]:prob_exact.dt:prob_exact.tspan[2])
        u0 = prob_exact.u0

        ε = blob_bandwidth(u0)
        solvers = [
            SBTM(learning_rate=1e-4; verbose=verbose),
            SBTM(learning_rate=4e-4; verbose=verbose),
            Blob(ε; verbose=verbose),
            Blob(ε * 5; verbose=verbose),
        ]

        for solver in solvers
            println("    $(long_name(solver))")
            prob = problem(d, n, solver)
            set_u0!(prob, u0)
            @time (@timeit DEFAULT_TIMER "$problem $d $(long_name(solver))" sol = solve(prob))

            logged_score_values = prob.solver.logger.score_values
            true_score_values = [true_score(prob, t, u) for (t, u) in zip(ts, sol.u)]
            label = "$(long_name(prob.solver))"
            push!(plot_data[i,j], (logged_score_values, true_score_values, label))
        end
    end
    return plot_data
end

n = 2000
ds = [3, 10]
problems = [diffusion_problem, fpe_problem, landau_problem]
plot_data = get_plot_data(problems, ds, n; verbose=1)

plots = []
for (i, problem) in enumerate(problems), (j, d) in enumerate(ds)
    prob_name = problem(d, n, Exact()).name
    plt_ = plot(xlabel="time step", title="$prob_name d=$d, n=$n", ylabel="Σᵢ|s(xᵢ) - ∇log u*(Xᵢ)|^2")
    for (logged_score_values, true_score_values, label) in plot_data[i, j]
        score_val_diff = l2_error_normalized.(logged_score_values, true_score_values)
        plot!(plt_, score_val_diff, label=label)
    end
    push!(plots, plt_)
end

plt = plot(plots..., plot_title="Score error", size=PLOT_WINDOW_SIZE, linewidth=4, margin=PLOT_MARGIN, layout=size(plot_data), legendfont=font(15), thickness_scaling=1.5);
savefig(plt, "data/plots/score_plots/score_errors")