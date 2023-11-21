mutable struct Experiment{D,S,P,V,T}
    true_dist::Vector{D}
    solution::Vector{S}
    params::P
    saveat::V
    dt::Float64
    timer::T
    problem_name::String
    solver_name::String
end

function Experiment(problem::GradFlowProblem; saveat=problem.tspan)
    solution = Vector{typeof(problem.u0)}(undef, 0)
    true_dist_ = [true_dist(problem, t) for t in saveat]
    return Experiment(true_dist_, solution, problem.params, saveat, Float64(problem.dt), TimerOutput(), problem.name, name(problem.solver))
end

function solve!(experiment::Experiment)
    @unpack problem, saveat, solution = experiment
    reset_timer!(DEFAULT_TIMER)
    experiment.solution = solve(problem, saveat=saveat).u
    experiment.timer = deepcopy(DEFAULT_TIMER)
    nothing
end

function Base.show(io::IO, experiment::Experiment)
    @unpack solution, problem_name, solver_name = experiment
    d, n = size(solution[1])
    print(io, "$problem_name d=$d n=$(rpad(n,n_WIDTH)) $(lpad(solver_name, SOLVER_NAME_WIDTH))")
end