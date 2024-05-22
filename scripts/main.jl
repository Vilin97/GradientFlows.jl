using GradientFlows

problems = []
ds = [2,3]
# for IC in ["normal"], γ in [0, -3], d in ds
#     push!(problems, (landau_problem_factory(d; IC=IC, γ=γ), d))
# end
for d in ds
    push!(problems, (landau_problem, d))
end

runs = 1:10
ns = 100 * 2 .^ (0:7)
solvers = [SBTM(), Blob()]

### train nn ###
@log @trySendTelegramMessage train_nns([(landau_problem, d) for d in ds], 80000; nn_depth=2, verbose=1)

### generate data ###
@log @trySendTelegramMessage run_experiments(problems, ns, runs, solvers)

### plot ###
@log @trySendTelegramMessage plot_all(problems, ns, solvers)