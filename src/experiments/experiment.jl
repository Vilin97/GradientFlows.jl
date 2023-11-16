mutable struct GradFlowExperiment{P,V,S,T}
    problem::P
    saveat::V
    solution::Vector{S} # solution[time_index][d, n] isa Float
    timer::T
end

function Base.show(io::IO, experiment::GradFlowExperiment)
    @unpack problem = experiment
    d, n = size(problem.u0)
    print(io, "$(problem.name) d=$d n=$(rpad(n,n_WIDTH)) $(lpad(problem.solver, SOLVER_NAME_WIDTH))")
end

function GradFlowExperiment(problem::GradFlowProblem; saveat=problem.tspan)
    solution = Vector{typeof(problem.u0)}(undef, 0)
    return GradFlowExperiment(problem, saveat, solution, TimerOutput())
end

function solve!(experiment::GradFlowExperiment)
    @unpack problem, saveat, solution = experiment
    reset_timer!(DEFAULT_TIMER)
    experiment.solution = solve(problem, saveat=saveat).u
    experiment.timer = deepcopy(DEFAULT_TIMER)
    nothing
end
