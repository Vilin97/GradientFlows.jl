using GradientFlows, StableRNGs, LinearAlgebra, Plots
using GradientFlows: update!, score

function pdf_plot(experiment; t_idx=2, xrange=range(-5, 5, length=200), h_coefs=0.5:0.5:2.0)
    slice(x::Number) = [x, zeros(typeof(x), d - 1)...]
    solver = experiment.solver_name
    saveat = experiment.saveat
    dist = experiment.true_dist[t_idx]
    p_marginal = Plots.plot(size=PLOT_WINDOW_SIZE, xlabel="x", ylabel="Σᵢϕ(x - Xᵢ[1])/n", title="marginal density t=$(saveat[t_idx])")
    p_slice = Plots.plot(size=PLOT_WINDOW_SIZE, xlabel="x", ylabel="Σᵢϕ([x,0...] - Xᵢ)/n", title="slice density t=$(saveat[t_idx])")
    u = experiment.solution[t_idx]
    u_marginal = reshape(u[1, :], 1, :)
    plot!(p_marginal, xrange, x -> marginal_pdf(dist, x), label="true")
    plot!(p_slice, xrange, x -> pdf(dist, slice(x)), label="true")
    for h_coef in h_coefs
        h = h_coef * kde_bandwidth(u_marginal)
        plot!(p_marginal, xrange, x -> kde([x], u_marginal; h=h, label="$solver h=$(round.(h[1],digits=4))"))
        h = h_coef * kde_bandwidth(u)
        plot!(p_slice, xrange, x -> kde(slice(x), u; h=h), label="$solver h=$(round.((det(h)^(1/d)), digits=4))")
    end
    return p_marginal, p_slice
end

using Plots
n = 2000
plots = []
verbose = 0
for d in [3, 5, 10]
    println("            d=$d")
    plot_data = []
    labels = []
    for problem in [diffusion_problem, landau_problem, fpe_problem]

        # exact
        prob1 = problem(d, n, Exact())
        println("        $(prob1.name)")
        ts = collect(prob1.tspan[1]:prob1.dt:prob1.tspan[2])

        # sbtm
        println("    SBTM")
        prob2 = problem(d, n, SBTM(; verbose=verbose, logger=Logger(1)))
        set_u0!(prob2, prob1.u0)
        sol2 = solve(prob2)

        logged_score_values2 = prob2.solver.logger.score_values
        true_score_values = [true_score(prob2, t, u) for (t, u) in zip(ts, sol2.u)]
        score_val_diff = [sum(abs2, logged_score_values2[i] .- true_score_values[i]) / sum(abs2, true_score_values[i]) for i in 1:length(true_score_values)]
        push!(plot_data, score_val_diff)
        push!(labels, "$(prob2.name) $(name(prob2.solver))")

        # blob
        println("    Blob")
        prob3 = problem(d, n, Blob(; verbose=verbose, logger=Logger(1)))
        set_u0!(prob3, prob1.u0)
        sol3 = solve(prob3)

        logged_score_values3 = prob3.solver.logger.score_values
        true_score_values = [true_score(prob3, t, u) for (t, u) in zip(ts, sol3.u)]
        score_val_diff = [sum(abs2, logged_score_values3[i] .- true_score_values[i]) / sum(abs2, true_score_values[i]) for i in 1:length(true_score_values)]
        push!(plot_data, score_val_diff)
        push!(labels, "$(prob3.name) $(name(prob3.solver))")
    end
    plt = plot(plot_data, label=reshape(labels, 1, :), xlabel="time step", title="d=$d, n=$n", ylabel="Σᵢ|s(xᵢ) - ∇log(xᵢ)|² / Σᵢ|∇log(xᵢ)|²")
    push!(plots, plt)
end
plt = plot(plots..., plot_title="Score approximation error", size=(1200, 800), linewidth=3, margin=(13, :mm))
savefig(plt, "data/plots/score_approximation_error_learning_rate_1e-4")


# d = 3
# n = 1000
# verbose=1
# using Flux: Adam
# prob2 = diffusion_problem(d, n, SBTM(mlp(d,depth=1);verbose=verbose, logger=Logger(1), learning_rate=1f-4))
# ts = collect(prob2.tspan[1]:prob2.dt:prob2.tspan[2])
# sol2 = solve(prob2)

# logged_score_values2 = prob2.solver.logger.score_values
# true_score_values = [true_score(prob2, t, u) for (t, u) in zip(ts, sol2.u)]
# score_val_diff = [sum(abs2, logged_score_values2[i] .- true_score_values[i]) / sum(abs2, true_score_values[i]) for i in 1:length(true_score_values)]
# plot!(plt, score_val_diff, label="D=$(prob2.params.D), training rate = 10^-4", title="SBTM score approximation in diffusion d=$d n=$n")