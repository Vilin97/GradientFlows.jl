using GradientFlows
include("plot.jl")

problems = [(coulomb_landau_problem, 2), (anisotropic_landau_problem, 2)]
num_runs = 1
ns = 100 * 2 .^ (0:6)
solvers = [SBTM(), Blob()]

dts = [1.0, 0.2, 0.05, 0.01]
dirs = [joinpath("data","dt_$(round(Int, dt * 1000))ms") for dt in dts]

### train nn ###

### copy models ###
for dir in dirs, problem_name in ["anisotropic_landau", "coulomb_landau"]
    source = joinpath("data", "models", problem_name, "d_2", "n_20000.jld2")
    destination = joinpath(dir, "models", problem_name, "d_2", "n_20000.jld2")
    mkpath(dirname(destination))
    cp(source, destination)
end

### generate data ###
for dt in dts
    dt_ms = round(Int, dt * 1000)
    dir = "data_$(dt_ms)ms"
    run_experiments(problems, ns, num_runs, solvers; dir=dir, dt=dt)
    @time plot_all(problems, ns, solvers; dir=dir);
end
