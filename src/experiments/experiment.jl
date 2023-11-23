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
    reset_timer!(DEFAULT_TIMER)
    solution = solve(problem, saveat=saveat).u
    timer = deepcopy(DEFAULT_TIMER)
    true_dist_ = [true_dist(problem, t) for t in saveat]
    return Experiment(true_dist_, solution, problem.params, saveat, Float64(problem.dt), timer, problem.name, name(problem.solver))
end

function Base.show(io::IO, experiment::Experiment)
    @unpack solution, problem_name, solver_name = experiment
    d, n = size(solution[1])
    print(io, "$problem_name d=$d n=$(rpad(n,n_WIDTH)) $(lpad(solver_name, SOLVER_NAME_WIDTH))")
end