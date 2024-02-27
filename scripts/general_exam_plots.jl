using GradientFlows
include("plot.jl")

function plot_slice_pdfs(problem_name, d, n, solver_names; dir="data")
    p_cov_trajectory_1 = plot_covariance_trajectory(problem_name, d, n, solver_names; row=1, column=1, dir=dir)
    p_cov_trajectory_2 = plot_covariance_trajectory(problem_name, d, n, solver_names; row=2, column=2, dir=dir)
    path = joinpath(dir, "plots", problem_name, "d_$d")
    mkpath(path)
    for (plt, name) in zip([p_cov_trajectory_1, p_cov_trajectory_2], ["cov_trajectory_1_n_$n", "cov_trajectory_2_n_$n"])
        savefig(plt, joinpath(path, name))
    end
end

for n in 100 .* 2 .^ (0:8)
    plot_slice_pdfs("anisotropic_landau", 3, n, ["blob", "sbtm"])
end