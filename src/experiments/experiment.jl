struct GradFlowExperiment{P, V, S, T}
	problem :: P
    saveat :: V
	solutions :: Vector{S}
    timer :: T
end

"Solve `problem` `num_solutions` times with different u0."
function GradFlowExperiment(problem::GradFlowProblem, saveat, num_solutions :: Int)
    solutions = Vector{Vector{typeof(problem.u0)}}(undef, num_solutions)
    reset_timer!()
    @timeit "" for i in 1:num_solutions
        resample!(problem; rng=StableRNG(i))
        sol = solve(problem, saveat=saveat)
        solutions[i] = sol.u
    end
    return GradFlowExperiment(problem, saveat, solutions, TimerOutputs.get_defaulttimer())
end
