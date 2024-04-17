using GradientFlows
include("plot.jl")

problems = [(coulomb_landau_normal_problem, 2), (coulomb_landau_mixture_problem, 2), (anisotropic_landau_problem, 2)]
num_runs = 5
ns = 100 * 2 .^ (0:6)
solvers = [SBTM(), Blob()]

### train nn ###
# for problem in problems
#     train_nn(problem..., 20000, mlp(2, depth=1))
# end

### generate data ###
run_experiments(problems, ns, num_runs, solvers)
plot_all(problems, ns, solvers)

# d = 2
# dir="data"
# solver_names = ["SBTM", "Blob"]
for problem_name in ["coulomb_landau_normal", "coulomb_landau_mixture"], n in ns
    @show problem_name, n
    p_marginal_start, p_slice_start = pdf_plot(problem_name, d, n, solver_names, t_idx=1; dir=dir)
    p_marginal_end, p_slice_end = pdf_plot(problem_name, d, n, solver_names, t_idx=0; dir=dir)
    plt = plot(p_marginal_start, p_marginal_end, p_slice_start, p_slice_end, size=PLOT_WINDOW_SIZE, margin=PLOT_MARGIN, plot_title="$problem_name, d=$d, n=$n", linewidth=PLOT_LINE_WIDTH)
    path = joinpath("data", "plots", "pdf", problem_name, "d_$d")
    plot_name = "n_$n"
    mkpath(path)
    savefig(plt, joinpath(path, plot_name))
end