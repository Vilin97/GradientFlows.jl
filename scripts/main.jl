using GradientFlows

dir = joinpath("data", "long_time")
problems = []
ds = [2]
# for IC in ["normal"], γ in [0], d in ds
#     push!(problems, (landau_problem_factory(d; IC=IC, γ=γ), d))
# end
for d in ds
    push!(problems, (landau_problem, d))
end

runs = 1:10
ns = 100 * 2 .^ (0:7)
solvers = [SBTM(), Blob()]

### train nn ###
# @log @trySendTelegramMessage train_nns(problems, 80000; nn_depth=2, verbose=1)

### generate data ###
@log @trySendTelegramMessage run_experiments(problems, ns, runs, solvers;dir=dir)

### plot ###
@log @trySendTelegramMessage plot_all(problems, ns, solvers;dir=dir, save_dir=joinpath(dir,"plots"))