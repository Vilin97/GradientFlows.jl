using GradientFlows
include("plot.jl")

problems = [(coulomb_landau_problem, 2), (anisotropic_landau_problem, 2)]
num_runs = 1
ns = 100 * 2 .^ (0:6)
solvers = [SBTM(), Blob()]

dts = [1.0, 0.2, 0.05, 0.01]
dirs = [joinpath("data","dt_$(round(Int, dt * 1000))ms") for dt in dts]

### train nn ###
for dir_ in dirs
    for problem in problems
        train_nn(problem..., 20000, mlp(2, depth=1); dir=dir_)
    end
end

### generate data ###
for (dt, dir_) in zip(dts, dirs)
    @show dt
    run_experiments(problems, ns, num_runs, solvers; dir=dir_, dt=dt)
end
for (dt, dir_) in zip(dts, dirs)
    @time plot_all(problems, ns, solvers; dir=dir_);
end
